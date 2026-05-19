// Inline P2G scatter kernel (cuda_v1_inline).
//
// One thread per particle. Each thread:
//   1. Loads x, v, C, stress into registers (stress is precomputed by JAX).
//   2. Computes B-spline weights for its particle position.
//   3. Loops over 27 stencil nodes:
//        - computes per-stencil weight, dweight, dpos, grid index
//        - computes momentum contribution mv = -dt*vol*stress@dweight
//                                              + p_mass*weight*(v + C@dpos)
//        - atomicAdds mv and mass into grid buffers
//
// Compared to cuda_fused:
//   - No SVD, no plasticity, no stress formula in this kernel — JAX does that
//     upstream (via jacobi_svd_3x3 + corotated_elasticity_jacobi).
//   - Result: ~80 lines of CUDA instead of ~400; works for ANY constitutive
//     model (JAX just has to emit stress: (N, 3, 3)).
//   - Mathematically identical to solver._single_particle_p2g (uses the same
//     stress @ dweight formula, not the affine = -dt*vol*4*inv_dx^2*stress
//     MLS-MPM trick used in cuda_fused). Equivalence vs JAX is f32-tight.
//
// Inputs (all float32):
//   x:      (N, 3)        particle positions
//   v:      (N, 3)        particle velocities
//   C:      (N, 9)        APIC affine matrix (row-major)
//   stress: (N, 9)        Kirchhoff stress, precomputed by JAX (row-major)
//
// Outputs:
//   grid_mv: (G^3, 3)
//   grid_m:  (G^3,)
//
// Scalar attributes: N, G, dt, vol, p_mass, inv_dx, dx

#include "xla/ffi/api/ffi.h"

#define BLOCK_SIZE 256

namespace ffi = xla::ffi;

// Zero a float buffer (grid-stride loop).
__global__ void zero_kernel(float* __restrict__ buf, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x)
        buf[i] = 0.0f;
}

__global__ void p2g_inline_kernel(
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
    if (pid >= N) return;

    // Load this particle's state into registers — read ONCE, reused 27x.
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

    // Base node + fractional offset for quadratic B-spline (matches
    // solver._single_particle_weights).
    float fpx[3], fx[3];
    int base[3];
    for (int d = 0; d < 3; d++) {
        fpx[d] = px[d] * inv_dx;
        base[d] = (int)floorf(fpx[d] - 0.5f);
        fx[d] = fpx[d] - (float)base[d];
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

    // Scatter to 27 stencil nodes — register-resident loop, no (N, 27, *)
    // intermediate ever exists in HBM.
    for (int di = 0; di < 3; di++)
    for (int dj = 0; dj < 3; dj++)
    for (int dk = 0; dk < 3; dk++) {
        float weight = w[0][di] * w[1][dj] * w[2][dk];

        // dweight = inv_dx * gradient of weight along each axis
        float dweight[3];
        dweight[0] = inv_dx * dw[0][di] * w[1][dj]  * w[2][dk];
        dweight[1] = inv_dx * w[0][di]  * dw[1][dj] * w[2][dk];
        dweight[2] = inv_dx * w[0][di]  * w[1][dj]  * dw[2][dk];

        // dpos = (offset - fx) * dx
        float dpos[3];
        dpos[0] = ((float)di - fx[0]) * dx;
        dpos[1] = ((float)dj - fx[1]) * dx;
        dpos[2] = ((float)dk - fx[2]) * dx;

        // Flat grid index — clip to valid range (matches jnp.clip in solver.py).
        int gi = max(0, min(base[0] + di, G - 1));
        int gj = max(0, min(base[1] + dj, G - 1));
        int gk = max(0, min(base[2] + dk, G - 1));
        int grid_idx = gi * G * G + gj * G + gk;

        // mv = -dt*vol*stress @ dweight + p_mass*weight*(v + C @ dpos)
        // Matches solver._single_particle_p2g exactly.
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

        atomicAdd(&grid_mv[grid_idx * 3 + 0], mv[0]);
        atomicAdd(&grid_mv[grid_idx * 3 + 1], mv[1]);
        atomicAdd(&grid_mv[grid_idx * 3 + 2], mv[2]);
        atomicAdd(&grid_m[grid_idx],          weight * p_mass);
    }
}

// ---------------------------------------------------------------------------
// XLA FFI handler
// ---------------------------------------------------------------------------

ffi::Error P2GInlineImpl(
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
    p2g_inline_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
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
    P2GInline, P2GInlineImpl,
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
