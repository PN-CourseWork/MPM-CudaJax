// Inline P2G scatter kernel with warp-shuffle reduction (cuda_v2_inline).
//
// Identical to cuda_v1_inline (p2g_inline.cu) for the per-particle setup —
// one thread per particle, register-resident state, inline B-spline weights,
// no (N, 27, *) materialisation in HBM. The ONLY difference is the 27-stencil
// scatter: before each atomicAdd, lanes inside the warp that target the same
// grid_idx reduce their contributions via __match_any_sync + __shfl_xor_sync,
// and only the leader lane issues the atomic.
//
// Question this kernel answers: now that the (N, 27, *) materialisation is
// gone (cuda_v1_inline structure), does warp-shuffle reduction on top of
// atomicAdd help, hurt, or wash?
//
// Requires sm_70+ for __match_any_sync (the A10 / sm_86 is fine).
//
// Inputs/outputs identical to p2g_inline.cu — see that file for layout.

#include "xla/ffi/api/ffi.h"

#define BLOCK_SIZE 256
#define FULL_MASK 0xFFFFFFFFu

namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------
// Warp-level reduction helper for coalescing matching stencil targets.
// ---------------------------------------------------------------------------

// Reduce `val` across all lanes in `mask` using butterfly shuffle.
// Returns the sum in ALL lanes of the group (not just the leader).
__device__ __forceinline__ float warp_reduce_masked(float val, unsigned mask) {
    for (int delta = 16; delta >= 1; delta >>= 1) {
        float other = __shfl_xor_sync(mask, val, delta);
        // Only add if the other lane is actually in our group.
        if (mask & (1u << ((threadIdx.x & 31) ^ delta)))
            val += other;
    }
    return val;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Zero a float buffer (grid-stride loop).
__global__ void zero_kernel(float* __restrict__ buf, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x)
        buf[i] = 0.0f;
}

__global__ void p2g_v2_inline_kernel(
    const float* __restrict__ x,        // (N, 3)
    const float* __restrict__ v,        // (N, 3)
    const float* __restrict__ C,        // (N, 9) row-major
    const float* __restrict__ stress,   // (N, 9) row-major
    float*       __restrict__ grid_mv,  // (G^3, 3)
    float*       __restrict__ grid_m,   // (G^3,)
    int N, int G,
    float dt, float vol, float p_mass, float inv_dx, float dx
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;

    // Out-of-range threads must still participate in __match_any_sync (they
    // contribute a sentinel grid_idx of -1, so they form their own group and
    // skip the atomics in the leader check). Otherwise we deadlock the warp.
    bool active = pid < N;

    float px[3], pv[3];
    float pC[9], pS[9];

    if (active) {
        for (int d = 0; d < 3; d++) {
            px[d] = x[pid * 3 + d];
            pv[d] = v[pid * 3 + d];
        }
        for (int i = 0; i < 9; i++) {
            pC[i] = C[pid * 9 + i];
            pS[i] = stress[pid * 9 + i];
        }
    }

    // Base node + fractional offset for quadratic B-spline.
    float fpx[3], fx[3];
    int base[3];
    if (active) {
        for (int d = 0; d < 3; d++) {
            fpx[d] = px[d] * inv_dx;
            base[d] = (int)floorf(fpx[d] - 0.5f);
            fx[d] = fpx[d] - (float)base[d];
        }
    }

    // Per-axis weight + weight-gradient tables (3 entries each).
    float w[3][3], dw[3][3];
    if (active) {
        for (int d = 0; d < 3; d++) {
            w[d][0] = 0.5f * (1.5f - fx[d]) * (1.5f - fx[d]);
            w[d][1] = 0.75f - (fx[d] - 1.0f) * (fx[d] - 1.0f);
            w[d][2] = 0.5f * (fx[d] - 0.5f) * (fx[d] - 0.5f);
            dw[d][0] = fx[d] - 1.5f;
            dw[d][1] = -2.0f * (fx[d] - 1.0f);
            dw[d][2] = fx[d] - 0.5f;
        }
    }

    // Scatter to 27 stencil nodes. All warp lanes must execute the loop
    // (they all participate in __match_any_sync); inactive lanes carry
    // grid_idx = -1 so they form their own match group and the atomic guard
    // (active && lane == leader) keeps them from writing.
    for (int di = 0; di < 3; di++)
    for (int dj = 0; dj < 3; dj++)
    for (int dk = 0; dk < 3; dk++) {
        float weight = 0.0f;
        float dweight[3] = {0.0f, 0.0f, 0.0f};
        float dpos[3] = {0.0f, 0.0f, 0.0f};
        int grid_idx = -1;

        if (active) {
            weight = w[0][di] * w[1][dj] * w[2][dk];

            dweight[0] = inv_dx * dw[0][di] * w[1][dj]  * w[2][dk];
            dweight[1] = inv_dx * w[0][di]  * dw[1][dj] * w[2][dk];
            dweight[2] = inv_dx * w[0][di]  * w[1][dj]  * dw[2][dk];

            dpos[0] = ((float)di - fx[0]) * dx;
            dpos[1] = ((float)dj - fx[1]) * dx;
            dpos[2] = ((float)dk - fx[2]) * dx;

            int gi = max(0, min(base[0] + di, G - 1));
            int gj = max(0, min(base[1] + dj, G - 1));
            int gk = max(0, min(base[2] + dk, G - 1));
            grid_idx = gi * G * G + gj * G + gk;
        }

        // mv = -dt*vol*stress @ dweight + p_mass*weight*(v + C @ dpos)
        // Matches solver._single_particle_p2g exactly.
        float mv0 = 0.0f, mv1 = 0.0f, mv2 = 0.0f;
        if (active) {
            float s_dw[3], c_dp[3];
            for (int d = 0; d < 3; d++) {
                float s = 0.0f, c = 0.0f;
                for (int j = 0; j < 3; j++) {
                    s += pS[d * 3 + j] * dweight[j];
                    c += pC[d * 3 + j] * dpos[j];
                }
                s_dw[d] = s;
                c_dp[d] = c;
            }
            mv0 = -dt * vol * s_dw[0] + p_mass * weight * (pv[0] + c_dp[0]);
            mv1 = -dt * vol * s_dw[1] + p_mass * weight * (pv[1] + c_dp[1]);
            mv2 = -dt * vol * s_dw[2] + p_mass * weight * (pv[2] + c_dp[2]);
        }
        float mass_contrib = active ? (weight * p_mass) : 0.0f;

        // Warp coalescing: find lanes in this warp targeting the same grid_idx.
        // Inactive lanes (grid_idx = -1) form their own group and don't write.
        unsigned peers = __match_any_sync(FULL_MASK, grid_idx);

        // Sum contributions across matching lanes.
        mv0 = warp_reduce_masked(mv0, peers);
        mv1 = warp_reduce_masked(mv1, peers);
        mv2 = warp_reduce_masked(mv2, peers);
        mass_contrib = warp_reduce_masked(mass_contrib, peers);

        // Only the leader (lowest lane in group) does the atomic.
        int leader = __ffs(peers) - 1;  // __ffs returns 1-indexed
        if (active && lane == leader) {
            atomicAdd(&grid_mv[grid_idx * 3 + 0], mv0);
            atomicAdd(&grid_mv[grid_idx * 3 + 1], mv1);
            atomicAdd(&grid_mv[grid_idx * 3 + 2], mv2);
            atomicAdd(&grid_m[grid_idx],          mass_contrib);
        }
    }
}

// ---------------------------------------------------------------------------
// XLA FFI handler
// ---------------------------------------------------------------------------

ffi::Error P2GV2InlineImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> x,
    ffi::Buffer<ffi::F32> v,
    ffi::Buffer<ffi::F32> C,
    ffi::Buffer<ffi::F32> stress,
    ffi::ResultBuffer<ffi::F32> grid_mv,
    ffi::ResultBuffer<ffi::F32> grid_m,
    int32_t N,
    int32_t G,
    float dt, float vol, float p_mass, float inv_dx, float dx
) {
    int grid_mv_size = G * G * G * 3;
    int grid_m_size = G * G * G;

    int zero_blocks = (grid_mv_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    zero_kernel<<<zero_blocks, BLOCK_SIZE, 0, stream>>>(
        grid_mv->typed_data(), grid_mv_size);
    zero_kernel<<<zero_blocks, BLOCK_SIZE, 0, stream>>>(
        grid_m->typed_data(), grid_m_size);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    p2g_v2_inline_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        x.typed_data(),
        v.typed_data(),
        C.typed_data(),
        stress.typed_data(),
        grid_mv->typed_data(),
        grid_m->typed_data(),
        N, G,
        dt, vol, p_mass, inv_dx, dx
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    P2GV2Inline, P2GV2InlineImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()   // x
        .Arg<ffi::Buffer<ffi::F32>>()   // v
        .Arg<ffi::Buffer<ffi::F32>>()   // C
        .Arg<ffi::Buffer<ffi::F32>>()   // stress
        .Ret<ffi::Buffer<ffi::F32>>()   // grid_mv
        .Ret<ffi::Buffer<ffi::F32>>()   // grid_m
        .Attr<int32_t>("N")
        .Attr<int32_t>("G")
        .Attr<float>("dt")
        .Attr<float>("vol")
        .Attr<float>("p_mass")
        .Attr<float>("inv_dx")
        .Attr<float>("dx")
);
