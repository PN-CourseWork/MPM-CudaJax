// Inline P2G scatter kernel with warp-shuffle atomic coalescing (cuda_v3_inline).
//
// Same structure as p2g_inline.cu (cuda_v1_inline): one thread per particle,
// register-resident state, inline B-spline weights, 27-stencil scatter loop.
//
// Difference: before each atomicAdd into the grid, threads in the same warp
// that happen to target the SAME grid node detect each other with
// __match_any_sync, sum their contributions with __shfl_xor_sync, and elect
// a single leader to do one atomic on behalf of the group. When particles
// are spatially sorted (Morton / Z-order) BEFORE this kernel runs, many
// warp lanes hit the same stencil cell and the number of global atomics
// drops dramatically.
//
// Helper warp_reduce_masked is cribbed from p2g_scatter_warp.cu.
//
// Inputs (all float32):
//   x:      (N, 3)        particle positions          (assumed sorted)
//   v:      (N, 3)        particle velocities         (in same order as x)
//   C:      (N, 9)        APIC affine matrix          (in same order as x)
//   stress: (N, 9)        Kirchhoff stress            (in same order as x)
//
// Outputs:
//   grid_mv: (G^3, 3)
//   grid_m:  (G^3,)
//
// Scalar attributes: N, G, dt, vol, p_mass, inv_dx, dx

#include "xla/ffi/api/ffi.h"

#define BLOCK_SIZE 256
#define FULL_MASK 0xFFFFFFFFu

namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------
// Warp-level masked reduction (copied from p2g_scatter_warp.cu).
// Reduce `val` across all lanes in `mask` using butterfly shuffle. Returns
// the sum in ALL lanes of the group (not just the leader).
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_masked(float val, unsigned mask) {
    for (int delta = 16; delta >= 1; delta >>= 1) {
        float other = __shfl_xor_sync(mask, val, delta);
        if (mask & (1u << ((threadIdx.x & 31) ^ delta)))
            val += other;
    }
    return val;
}

// ---------------------------------------------------------------------------
// Zero a float buffer (grid-stride loop).
// ---------------------------------------------------------------------------

__global__ void zero_kernel(float* __restrict__ buf, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x)
        buf[i] = 0.0f;
}

__global__ void p2g_v3_inline_kernel(
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

    // Out-of-range threads have to participate in the warp sync intrinsics
    // (otherwise __match_any_sync deadlocks). We mark them as "inactive" by
    // giving them a grid index of -1 so they never match an in-range lane,
    // and we skip the atomicAdd at the bottom.
    bool active = (pid < N);
    int lane = threadIdx.x & 31;

    float px[3] = {0,0,0};
    float pv[3] = {0,0,0};
    float pC[9];
    float pS[9];
    #pragma unroll
    for (int i = 0; i < 9; i++) { pC[i] = 0.0f; pS[i] = 0.0f; }

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

    float fpx[3], fx[3];
    int base[3];
    for (int d = 0; d < 3; d++) {
        fpx[d] = px[d] * inv_dx;
        base[d] = (int)floorf(fpx[d] - 0.5f);
        fx[d] = fpx[d] - (float)base[d];
    }

    float w[3][3], dw[3][3];
    for (int d = 0; d < 3; d++) {
        w[d][0] = 0.5f * (1.5f - fx[d]) * (1.5f - fx[d]);
        w[d][1] = 0.75f - (fx[d] - 1.0f) * (fx[d] - 1.0f);
        w[d][2] = 0.5f * (fx[d] - 0.5f) * (fx[d] - 0.5f);
        dw[d][0] = fx[d] - 1.5f;
        dw[d][1] = -2.0f * (fx[d] - 1.0f);
        dw[d][2] = fx[d] - 0.5f;
    }

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

        int gi = max(0, min(base[0] + di, G - 1));
        int gj = max(0, min(base[1] + dj, G - 1));
        int gk = max(0, min(base[2] + dk, G - 1));
        int grid_idx = gi * G * G + gj * G + gk;

        // Inactive lanes get a sentinel that won't match any real index.
        int match_key = active ? grid_idx : -1;

        float mv0 = 0.0f, mv1 = 0.0f, mv2 = 0.0f, m_contrib = 0.0f;
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
            m_contrib = weight * p_mass;
        }

        // Find all lanes (in this 32-lane warp) targeting the same grid node.
        // Inactive lanes use match_key = -1, so they cluster together and
        // their (zeroed) contributions don't pollute real groups.
        unsigned peers = __match_any_sync(FULL_MASK, match_key);

        mv0 = warp_reduce_masked(mv0, peers);
        mv1 = warp_reduce_masked(mv1, peers);
        mv2 = warp_reduce_masked(mv2, peers);
        m_contrib = warp_reduce_masked(m_contrib, peers);

        int leader = __ffs(peers) - 1;
        if (active && lane == leader) {
            atomicAdd(&grid_mv[grid_idx * 3 + 0], mv0);
            atomicAdd(&grid_mv[grid_idx * 3 + 1], mv1);
            atomicAdd(&grid_mv[grid_idx * 3 + 2], mv2);
            atomicAdd(&grid_m[grid_idx],          m_contrib);
        }
    }
}

// ---------------------------------------------------------------------------
// XLA FFI handler
// ---------------------------------------------------------------------------

ffi::Error P2GV3InlineImpl(
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
    p2g_v3_inline_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
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
    P2GV3Inline, P2GV3InlineImpl,
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
