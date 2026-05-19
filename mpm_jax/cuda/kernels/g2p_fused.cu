// Fused G2P kernel.
//
// One thread per particle does the full G2P gather + state update in
// registers:
//   1. Compute B-spline weights from position (no global tensor)
//   2. Gather 27 grid velocities directly
//   3. Accumulate momentum, APIC C-matrix, and velocity gradient
//   4. Update x, v, C, F and write back
//
// Pairs with p2g_fused.cu so the cuda_fused path doesn't materialise the
// (N, 27, *) intermediate tensors that the JAX G2P stage produces via
// compute_weights_and_indices. At N=10M each (N, 27, 3) tensor is
// 3.24 GB; this kernel keeps everything in registers.
//
// Inputs (all float32):
//   x:       (N, 3)    post-BC particle positions
//   F:       (N, 9)    deformation gradient (row-major)
//   grid_v:  (G^3, 3)  post-update grid velocities
//
// Outputs (all float32):
//   new_x:   (N, 3)    clipped to [clip_bound, 1 - clip_bound]
//   new_v:   (N, 3)
//   new_C:   (N, 9)    APIC affine matrix, row-major
//   new_F:   (N, 9)    F_p + dt * grad_v @ F_p, clipped to [-2, 2]
//
// Scalar attributes: N, G, dt, inv_dx, dx, clip_bound

#include "xla/ffi/api/ffi.h"
#include <math.h>

#define BLOCK_SIZE 256

namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------
// Minimal 3x3 matmul (registers, row-major)
// ---------------------------------------------------------------------------

struct Mat3 {
    float m[9];
};

__device__ __forceinline__ Mat3 mat3_mul(const Mat3& A, const Mat3& B) {
    Mat3 C;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            float s = 0.0f;
            for (int k = 0; k < 3; k++)
                s += A.m[i*3+k] * B.m[k*3+j];
            C.m[i*3+j] = s;
        }
    return C;
}

// ---------------------------------------------------------------------------
// Fused G2P kernel
// ---------------------------------------------------------------------------

__global__ void g2p_fused_kernel(
    const float* __restrict__ x,        // (N, 3)
    const float* __restrict__ F,        // (N, 9) row-major
    const float* __restrict__ grid_v,   // (G^3, 3)
    float*       __restrict__ new_x,    // (N, 3)
    float*       __restrict__ new_v,    // (N, 3)
    float*       __restrict__ new_C,    // (N, 9)
    float*       __restrict__ new_F,    // (N, 9)
    int N, int G,
    float dt, float inv_dx, float dx, float clip_bound
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= N) return;

    // Load particle position
    float px[3];
    for (int d = 0; d < 3; d++) px[d] = x[pid * 3 + d];

    // Load deformation gradient
    Mat3 pF;
    for (int i = 0; i < 9; i++) pF.m[i] = F[pid * 9 + i];

    // B-spline weights and derivatives along each axis
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

    // Accumulators (registers)
    float v_acc[3] = {0.0f, 0.0f, 0.0f};
    float C_acc[9] = {0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f};
    float grad_v[9] = {0.0f, 0.0f, 0.0f,
                       0.0f, 0.0f, 0.0f,
                       0.0f, 0.0f, 0.0f};

    // Loop over 27 stencil nodes
    for (int di = 0; di < 3; di++)
    for (int dj = 0; dj < 3; dj++)
    for (int dk = 0; dk < 3; dk++) {
        float weight = w[0][di] * w[1][dj] * w[2][dk];
        float dwt[3];
        dwt[0] = inv_dx * dw[0][di] *  w[1][dj] *  w[2][dk];
        dwt[1] = inv_dx *  w[0][di] * dw[1][dj] *  w[2][dk];
        dwt[2] = inv_dx *  w[0][di] *  w[1][dj] * dw[2][dk];

        float dpos[3];
        dpos[0] = ((float)di - fx[0]) * dx;
        dpos[1] = ((float)dj - fx[1]) * dx;
        dpos[2] = ((float)dk - fx[2]) * dx;

        int gi = base[0] + di;
        int gj = base[1] + dj;
        int gk = base[2] + dk;
        gi = max(0, min(gi, G - 1));
        gj = max(0, min(gj, G - 1));
        gk = max(0, min(gk, G - 1));
        int grid_idx = gi * G * G + gj * G + gk;

        // Gather grid velocity (3 loads)
        float gv[3];
        gv[0] = grid_v[grid_idx * 3 + 0];
        gv[1] = grid_v[grid_idx * 3 + 1];
        gv[2] = grid_v[grid_idx * 3 + 2];

        // new_v += weight * gv
        for (int d = 0; d < 3; d++) v_acc[d] += weight * gv[d];

        // new_C += weight * outer(gv, dpos)  -> stored row-major
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                C_acc[i*3+j] += weight * gv[i] * dpos[j];

        // grad_v += outer(gv, dwt)
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                grad_v[i*3+j] += gv[i] * dwt[j];
    }

    // Scale C by APIC constant 4 * inv_dx^2
    float apic_scale = 4.0f * inv_dx * inv_dx;
    for (int i = 0; i < 9; i++) C_acc[i] *= apic_scale;

    // x_new = clip(x + v * dt)
    float x_out[3];
    for (int d = 0; d < 3; d++) {
        float xn = px[d] + v_acc[d] * dt;
        x_out[d] = fmaxf(clip_bound, fminf(xn, 1.0f - clip_bound));
    }

    // F_new = clip(F + dt * grad_v @ F, -2, 2)
    Mat3 gv_mat;
    for (int i = 0; i < 9; i++) gv_mat.m[i] = grad_v[i];
    Mat3 dgFp = mat3_mul(gv_mat, pF);
    float F_out[9];
    for (int i = 0; i < 9; i++) {
        float fn = pF.m[i] + dt * dgFp.m[i];
        F_out[i] = fmaxf(-2.0f, fminf(fn, 2.0f));
    }

    // Write outputs
    for (int d = 0; d < 3; d++) {
        new_x[pid * 3 + d] = x_out[d];
        new_v[pid * 3 + d] = v_acc[d];
    }
    for (int i = 0; i < 9; i++) {
        new_C[pid * 9 + i] = C_acc[i];
        new_F[pid * 9 + i] = F_out[i];
    }
}

// ---------------------------------------------------------------------------
// XLA FFI handler
// ---------------------------------------------------------------------------

ffi::Error G2PFusedImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> x,
    ffi::Buffer<ffi::F32> F,
    ffi::Buffer<ffi::F32> grid_v,
    ffi::ResultBuffer<ffi::F32> new_x,
    ffi::ResultBuffer<ffi::F32> new_v,
    ffi::ResultBuffer<ffi::F32> new_C,
    ffi::ResultBuffer<ffi::F32> new_F,
    int32_t N, int32_t G,
    float dt, float inv_dx, float dx, float clip_bound
) {
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    g2p_fused_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        x.typed_data(),
        F.typed_data(),
        grid_v.typed_data(),
        new_x->typed_data(),
        new_v->typed_data(),
        new_C->typed_data(),
        new_F->typed_data(),
        N, G,
        dt, inv_dx, dx, clip_bound
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error(ffi::ErrorCode::kInternal, cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    G2PFused, G2PFusedImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()   // x
        .Arg<ffi::Buffer<ffi::F32>>()   // F
        .Arg<ffi::Buffer<ffi::F32>>()   // grid_v
        .Ret<ffi::Buffer<ffi::F32>>()   // new_x
        .Ret<ffi::Buffer<ffi::F32>>()   // new_v
        .Ret<ffi::Buffer<ffi::F32>>()   // new_C
        .Ret<ffi::Buffer<ffi::F32>>()   // new_F
        .Attr<int32_t>("N")
        .Attr<int32_t>("G")
        .Attr<float>("dt")
        .Attr<float>("inv_dx")
        .Attr<float>("dx")
        .Attr<float>("clip_bound")
);
