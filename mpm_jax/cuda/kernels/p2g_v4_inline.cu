// Cell-major inline P2G scatter kernel (cuda_v4_inline).
//
// Combines two ideas:
//   * `p2g_inline.cu` (cuda_v1_inline): one thread per particle, inline
//     B-spline weights + 27-stencil scatter computed in registers. No
//     (N, 27, *) tensor is ever materialised in HBM.
//   * `p2g_scatter_smem.cu` (cuda_v4): particles are sorted by their home
//     cell on the JAX side; one CUDA block per grid cell aggregates its
//     particles' contributions into a 4x4x4 shared-memory tile via fast
//     shmem atomics, then flushes the tile to global memory with one
//     atomicAdd per node.
//
// The hypothesis is that the smem aggregation amortises the 27 global
// atomicAdds per particle (108 floats) down to ~64 global atomicAdds per
// occupied cell, regardless of how many particles live in the cell.
// `cuda_v4` already did this for the scatter-only path, but its inputs
// were the (N, 27, *) momentum/mass/index tensors precomputed by JAX —
// which cost more than the smem aggregation saved. With inline weight
// computation those tensors disappear from HBM and only x/v/C/stress are
// loaded once per particle.
//
// Inputs (all float32 unless noted):
//   x:          (N, 3)        particle positions (SORTED by home cell)
//   v:          (N, 3)        particle velocities
//   C:          (N, 9)        APIC affine matrix (row-major)
//   stress:     (N, 9)        Kirchhoff stress (row-major)
//   cell_start: (G^3 + 1,) int32  CSR boundaries into the sorted arrays
//
// Outputs:
//   grid_mv: (G^3, 3)
//   grid_m:  (G^3,)
//
// Scalar attributes: dt, vol, p_mass, inv_dx, dx
//
// Home cell convention (matches `cuda_p2g_scatter_smem` Python wrapper):
//   For a particle at x, the stencil base node is
//     base = floor(x * inv_dx - 0.5)
//   and the center stencil node (offset (1,1,1)) is
//     home = base + 1
//   So home in [0, G). Sorting by `home` puts every particle whose center
//   stencil node is the same cell together, and the 27 stencil indices for
//   any such particle are home + (-1..+1, -1..+1, -1..+1) — a 3^3 box
//   contained entirely in the 4x4x4 tile centred at (home-1).

#include "xla/ffi/api/ffi.h"

#define TILE_DIM 4
#define TILE_SIZE (TILE_DIM * TILE_DIM * TILE_DIM)  // 64 nodes
#define STENCIL 27
#define BLOCK_SIZE 128  // threads per cell-block

namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

__global__ void zero_kernel(float* __restrict__ buf, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x)
        buf[i] = 0.0f;
}

// ---------------------------------------------------------------------------
// Cell-major inline kernel
// ---------------------------------------------------------------------------
// One block per grid cell. Each thread in the block processes one or more
// particles from that cell (grid-stride loop over the in-cell particle
// range). Per particle:
//   - load x, v, C, stress
//   - compute base node + per-axis B-spline weight/dweight tables
//   - loop over 27 stencil offsets, atomicAdd momentum + mass into the
//     shared-memory tile (or fall back to global atomicAdd when the stencil
//     clips outside the tile — only happens at grid boundary).
// After all threads finish, the block flushes the 64 tile entries to global
// memory with one atomicAdd per (mv, m).

__global__ void p2g_v4_inline_kernel(
    const float* __restrict__ x,          // (N, 3) sorted
    const float* __restrict__ v,          // (N, 3) sorted
    const float* __restrict__ C,          // (N, 9) sorted, row-major
    const float* __restrict__ stress,     // (N, 9) sorted, row-major
    const int*   __restrict__ cell_start, // (G^3 + 1,)
    float*       __restrict__ grid_mv,    // (G^3, 3)
    float*       __restrict__ grid_m,     // (G^3,)
    int G,
    float dt, float vol, float p_mass, float inv_dx, float dx
) {
    int cell_id = blockIdx.x;
    int G3 = G * G * G;
    if (cell_id >= G3) return;

    int p_start = cell_start[cell_id];
    int p_end   = cell_start[cell_id + 1];
    int n_particles = p_end - p_start;

    // Cell 3D coords from flat index (matches the searchsorted CSR build).
    int ci = cell_id / (G * G);
    int cj = (cell_id / G) % G;
    int ck = cell_id % G;

    // Tile origin: (home - 1) so that the 3^3 stencil of any home-centred
    // particle lands at tile-local indices (0..2). The 4th slot is slack.
    int tile_i = ci - 1;
    int tile_j = cj - 1;
    int tile_k = ck - 1;

    // Shared-memory tile: 64 nodes, each (mv_x, mv_y, mv_z, mass).
    // 256 floats = 1 KB. Fits even on tiny chips.
    __shared__ float tile[TILE_SIZE * 4];

    // Cooperatively zero the tile.
    for (int t = threadIdx.x; t < TILE_SIZE * 4; t += blockDim.x)
        tile[t] = 0.0f;
    __syncthreads();

    // If the cell is empty, skip straight to the flush — but everyone has to
    // hit the same __syncthreads() so we can't early-return here. We just
    // let the per-particle loop be a no-op.

    // ---- Per-particle inline scatter into the tile ----
    for (int p = threadIdx.x; p < n_particles; p += blockDim.x) {
        int pid = p_start + p;

        // Register-resident particle state.
        float px[3], pv[3];
        for (int d = 0; d < 3; d++) {
            px[d] = x[pid * 3 + d];
            pv[d] = v[pid * 3 + d];
        }
        float pC[9], pS[9];
        for (int i = 0; i < 9; i++) {
            pC[i] = C[pid * 9 + i];
            pS[i] = stress[pid * 9 + i];
        }

        // Base + fractional offset for quadratic B-spline.
        float fx[3];
        int base[3];
        for (int d = 0; d < 3; d++) {
            float fpx = px[d] * inv_dx;
            base[d] = (int)floorf(fpx - 0.5f);
            fx[d] = fpx - (float)base[d];
        }

        // Per-axis weight + weight-gradient tables (3 entries each).
        float w[3][3], dw[3][3];
        for (int d = 0; d < 3; d++) {
            w[d][0] = 0.5f * (1.5f - fx[d]) * (1.5f - fx[d]);
            w[d][1] = 0.75f - (fx[d] - 1.0f) * (fx[d] - 1.0f);
            w[d][2] = 0.5f * (fx[d] - 0.5f) * (fx[d] - 0.5f);
            dw[d][0] = fx[d] - 1.5f;
            dw[d][1] = -2.0f * (fx[d] - 1.0f);
            dw[d][2] = fx[d] - 0.5f;
        }

        // 27-stencil register-resident loop.
        for (int di = 0; di < 3; di++)
        for (int dj = 0; dj < 3; dj++)
        for (int dk = 0; dk < 3; dk++) {
            float weight = w[0][di] * w[1][dj] * w[2][dk];

            float dweight[3];
            dweight[0] = inv_dx * dw[0][di] * w[1][dj]  * w[2][dk];
            dweight[1] = inv_dx * w[0][di]  * dw[1][dj] * w[2][dk];
            dweight[2] = inv_dx * w[0][di]  * w[1][dj]  * dw[2][dk];

            float dpos[3];
            dpos[0] = ((float)di - fx[0]) * dx;
            dpos[1] = ((float)dj - fx[1]) * dx;
            dpos[2] = ((float)dk - fx[2]) * dx;

            // Global grid index (with clip to match solver._single_particle_weights).
            int gi_raw = base[0] + di;
            int gj_raw = base[1] + dj;
            int gk_raw = base[2] + dk;
            int gi = max(0, min(gi_raw, G - 1));
            int gj = max(0, min(gj_raw, G - 1));
            int gk = max(0, min(gk_raw, G - 1));

            // Momentum contribution.
            float mv[3];
            for (int d = 0; d < 3; d++) {
                float s_dw = 0.0f;
                float c_dp = 0.0f;
                for (int j = 0; j < 3; j++) {
                    s_dw += pS[d * 3 + j] * dweight[j];
                    c_dp += pC[d * 3 + j] * dpos[j];
                }
                mv[d] = -dt * vol * s_dw + p_mass * weight * (pv[d] + c_dp);
            }
            float m_contrib = weight * p_mass;

            // Try the tile first. Tile-local index uses the (possibly
            // clipped) global index, so a clipped stencil that lands back
            // inside the tile is still tile-local and a stencil that lands
            // outside the tile (or outside the grid) goes via global atomic.
            int ti = gi - tile_i;
            int tj = gj - tile_j;
            int tk = gk - tile_k;
            if (ti >= 0 && ti < TILE_DIM &&
                tj >= 0 && tj < TILE_DIM &&
                tk >= 0 && tk < TILE_DIM) {
                int tile_idx = (ti * TILE_DIM * TILE_DIM + tj * TILE_DIM + tk) * 4;
                atomicAdd(&tile[tile_idx + 0], mv[0]);
                atomicAdd(&tile[tile_idx + 1], mv[1]);
                atomicAdd(&tile[tile_idx + 2], mv[2]);
                atomicAdd(&tile[tile_idx + 3], m_contrib);
            } else {
                // Out-of-tile fallback: direct global atomic. In practice this
                // only triggers near the grid boundary (cells where the
                // stencil clips). The body of the simulation never falls here.
                int grid_idx = gi * G * G + gj * G + gk;
                atomicAdd(&grid_mv[grid_idx * 3 + 0], mv[0]);
                atomicAdd(&grid_mv[grid_idx * 3 + 1], mv[1]);
                atomicAdd(&grid_mv[grid_idx * 3 + 2], mv[2]);
                atomicAdd(&grid_m[grid_idx], m_contrib);
            }
        }
    }

    __syncthreads();

    // ---- Flush the smem tile to global memory ----
    for (int t = threadIdx.x; t < TILE_SIZE; t += blockDim.x) {
        float smv0 = tile[t * 4 + 0];
        float smv1 = tile[t * 4 + 1];
        float smv2 = tile[t * 4 + 2];
        float sm   = tile[t * 4 + 3];

        // Skip entries no particle touched.
        if (sm == 0.0f && smv0 == 0.0f && smv1 == 0.0f && smv2 == 0.0f)
            continue;

        int ti = t / (TILE_DIM * TILE_DIM);
        int tj = (t / TILE_DIM) % TILE_DIM;
        int tk = t % TILE_DIM;
        int gi = tile_i + ti;
        int gj = tile_j + tj;
        int gk = tile_k + tk;

        if (gi < 0 || gi >= G || gj < 0 || gj >= G || gk < 0 || gk >= G)
            continue;

        int gid = gi * G * G + gj * G + gk;
        atomicAdd(&grid_mv[gid * 3 + 0], smv0);
        atomicAdd(&grid_mv[gid * 3 + 1], smv1);
        atomicAdd(&grid_mv[gid * 3 + 2], smv2);
        atomicAdd(&grid_m[gid],          sm);
    }
}

// ---------------------------------------------------------------------------
// XLA FFI handler
// ---------------------------------------------------------------------------

ffi::Error P2GV4InlineImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> x,
    ffi::Buffer<ffi::F32> v,
    ffi::Buffer<ffi::F32> C,
    ffi::Buffer<ffi::F32> stress,
    ffi::Buffer<ffi::S32> cell_start,
    ffi::ResultBuffer<ffi::F32> grid_mv,
    ffi::ResultBuffer<ffi::F32> grid_m,
    float dt, float vol, float p_mass, float inv_dx, float dx
) {
    // cell_start is (G^3 + 1,) ints.
    int G3_plus_1 = static_cast<int>(cell_start.dimensions()[0]);
    int G3 = G3_plus_1 - 1;
    int G = 1;
    while (G * G * G < G3) G++;
    if (G * G * G != G3) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "cell_start size is not G^3 + 1 for integer G");
    }

    int grid_mv_size = G3 * 3;
    int grid_m_size = G3;

    // Zero the grid (the kernel only adds into it).
    int zero_blocks = (grid_mv_size + 255) / 256;
    zero_kernel<<<zero_blocks, 256, 0, stream>>>(grid_mv->typed_data(), grid_mv_size);
    zero_kernel<<<(grid_m_size + 255) / 256, 256, 0, stream>>>(grid_m->typed_data(), grid_m_size);

    // One block per cell.
    p2g_v4_inline_kernel<<<G3, BLOCK_SIZE, 0, stream>>>(
        x.typed_data(),
        v.typed_data(),
        C.typed_data(),
        stress.typed_data(),
        reinterpret_cast<const int*>(cell_start.typed_data()),
        grid_mv->typed_data(),
        grid_m->typed_data(),
        G, dt, vol, p_mass, inv_dx, dx
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    P2GV4Inline, P2GV4InlineImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()   // x (sorted)
        .Arg<ffi::Buffer<ffi::F32>>()   // v (sorted)
        .Arg<ffi::Buffer<ffi::F32>>()   // C (sorted)
        .Arg<ffi::Buffer<ffi::F32>>()   // stress (sorted)
        .Arg<ffi::Buffer<ffi::S32>>()   // cell_start
        .Ret<ffi::Buffer<ffi::F32>>()   // grid_mv
        .Ret<ffi::Buffer<ffi::F32>>()   // grid_m
        .Attr<float>("dt")
        .Attr<float>("vol")
        .Attr<float>("p_mass")
        .Attr<float>("inv_dx")
        .Attr<float>("dx")
);
