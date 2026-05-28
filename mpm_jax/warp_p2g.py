"""Warp P2G kernels callable from JAX JIT via Warp's experimental FFI."""

import jax
import jax.numpy as jnp
import warp as wp
from warp.jax_experimental import jax_kernel


@wp.func
def _quad_weight(fx: float, offset: int):
    if offset == 0:
        return 0.5 * (1.5 - fx) * (1.5 - fx)
    if offset == 1:
        d = fx - 1.0
        return 0.75 - d * d
    d = fx - 0.5
    return 0.5 * d * d


@wp.func
def _quad_dweight(fx: float, offset: int):
    if offset == 0:
        return fx - 1.5
    if offset == 1:
        return -2.0 * (fx - 1.0)
    return fx - 0.5


@wp.kernel
def _p2g_inline_kernel(
    x: wp.array2d[float],
    v: wp.array2d[float],
    C: wp.array2d[float],
    stress: wp.array2d[float],
    G: int,
    dt: float,
    vol: float,
    p_mass: float,
    inv_dx: float,
    dx: float,
    grid_mv: wp.array[float],
    grid_m: wp.array[float],
):
    p = wp.tid()

    px0 = x[p, 0] * inv_dx
    px1 = x[p, 1] * inv_dx
    px2 = x[p, 2] * inv_dx

    b0 = int(wp.floor(px0 - 0.5))
    b1 = int(wp.floor(px1 - 0.5))
    b2 = int(wp.floor(px2 - 0.5))

    fx0 = px0 - float(b0)
    fx1 = px1 - float(b1)
    fx2 = px2 - float(b2)

    C00 = C[p, 0]
    C01 = C[p, 1]
    C02 = C[p, 2]
    C10 = C[p, 3]
    C11 = C[p, 4]
    C12 = C[p, 5]
    C20 = C[p, 6]
    C21 = C[p, 7]
    C22 = C[p, 8]

    s00 = stress[p, 0]
    s01 = stress[p, 1]
    s02 = stress[p, 2]
    s10 = stress[p, 3]
    s11 = stress[p, 4]
    s12 = stress[p, 5]
    s20 = stress[p, 6]
    s21 = stress[p, 7]
    s22 = stress[p, 8]

    vp0 = v[p, 0]
    vp1 = v[p, 1]
    vp2 = v[p, 2]

    for ox in range(3):
        wx = _quad_weight(fx0, ox)
        dwx = _quad_dweight(fx0, ox)
        for oy in range(3):
            wy = _quad_weight(fx1, oy)
            dwy = _quad_dweight(fx1, oy)
            for oz in range(3):
                wz = _quad_weight(fx2, oz)
                dwz = _quad_dweight(fx2, oz)

                weight = wx * wy * wz
                dw0 = inv_dx * dwx * wy * wz
                dw1 = inv_dx * wx * dwy * wz
                dw2 = inv_dx * wx * wy * dwz

                dpos0 = (float(ox) - fx0) * dx
                dpos1 = (float(oy) - fx1) * dx
                dpos2 = (float(oz) - fx2) * dx

                affine0 = vp0 + C00 * dpos0 + C01 * dpos1 + C02 * dpos2
                affine1 = vp1 + C10 * dpos0 + C11 * dpos1 + C12 * dpos2
                affine2 = vp2 + C20 * dpos0 + C21 * dpos1 + C22 * dpos2

                stress_dw0 = s00 * dw0 + s01 * dw1 + s02 * dw2
                stress_dw1 = s10 * dw0 + s11 * dw1 + s12 * dw2
                stress_dw2 = s20 * dw0 + s21 * dw1 + s22 * dw2

                mv0 = -dt * vol * stress_dw0 + p_mass * weight * affine0
                mv1 = -dt * vol * stress_dw1 + p_mass * weight * affine1
                mv2 = -dt * vol * stress_dw2 + p_mass * weight * affine2
                mass = p_mass * weight

                gi0 = b0 + ox
                gi1 = b1 + oy
                gi2 = b2 + oz
                idx = gi0 * G * G + gi1 * G + gi2
                idx = wp.clamp(idx, 0, G * G * G - 1)

                wp.atomic_add(grid_mv, idx * 3 + 0, mv0)
                wp.atomic_add(grid_mv, idx * 3 + 1, mv1)
                wp.atomic_add(grid_mv, idx * 3 + 2, mv2)
                wp.atomic_add(grid_m, idx, mass)


_jax_p2g_inline_kernel = jax_kernel(
    _p2g_inline_kernel,
    num_outputs=2,
    in_out_argnames=["grid_mv", "grid_m"],
)


def warp_p2g_inline(x, v, C, stress, num_grids, dt, vol, p_mass, inv_dx, dx):
    """Run inline P2G through a Warp kernel embedded in JAX."""
    n = x.shape[0]
    g3 = num_grids ** 3
    C_flat = C.reshape(n, 9)
    stress_flat = stress.reshape(n, 9)

    grid_mv0 = jnp.zeros((g3 * 3,), dtype=jnp.float32)
    grid_m0 = jnp.zeros((g3,), dtype=jnp.float32)

    grid_mv_flat, grid_m = _jax_p2g_inline_kernel(
        x, v, C_flat, stress_flat,
        int(num_grids),
        float(dt),
        float(vol),
        float(p_mass),
        float(inv_dx),
        float(dx),
        grid_mv0,
        grid_m0,
        launch_dims=n,
    )
    return grid_mv_flat.reshape((g3, 3)), grid_m


def build_jit_frame_warp_inline(params, elasticity_fn, plasticity_fn,
                                pre_particle_fn, post_grid_fn, steps_per_frame,
                                use_cuda_g2p=True):
    """Build a fully JIT'd frame using Warp P2G via JAX FFI."""
    from mpm_jax.solver import (
        MPMState,
        compute_weights_and_indices,
        g2p,
        grid_update as grid_update_fn,
    )

    if use_cuda_g2p:
        from mpm_jax.cuda.p2g_cuda import cuda_g2p_fused, is_available

        if not is_available("g2p_fused"):
            raise RuntimeError(
                "cuda g2p kernel not registered (missing .so?). "
                "Pass use_cuda_g2p=False to fall back to the JAX G2P."
            )
    else:
        cuda_g2p_fused = None

    @jax.jit
    def jit_frame(state):
        def scan_body(state, _):
            with jax.named_scope("pre_particle"):
                x, v = pre_particle_fn(state.x, state.v, 0.0)
            with jax.named_scope("elasticity"):
                stress = elasticity_fn(state.F)
            with jax.named_scope("warp_p2g_inline"):
                grid_mv, grid_m = warp_p2g_inline(
                    x, v, state.C, stress,
                    params.num_grids, params.dt, params.vol, params.p_mass,
                    params.inv_dx, params.dx,
                )
            with jax.named_scope("grid_update"):
                grid_mv = grid_update_fn(
                    grid_mv, grid_m, params.gravity, params.dt, params.damping)
                grid_v = post_grid_fn(grid_mv, grid_m, 0.0)

            with jax.named_scope("g2p"):
                if use_cuda_g2p:
                    new_x, new_v, new_C, new_F = cuda_g2p_fused(
                        x, state.F, grid_v,
                        params.num_grids, params.dt,
                        params.inv_dx, params.dx, params.clip_bound,
                    )
                else:
                    weight, dweight, dpos, index = compute_weights_and_indices(
                        x, params.inv_dx, params.dx, params.num_grids)
                    new_x, new_v, new_C, new_F = g2p(
                        grid_v, weight, dweight, dpos, index,
                        state.F, x, params.dt, params.inv_dx, params.clip_bound)

            with jax.named_scope("plasticity"):
                new_F = plasticity_fn(new_F)
            return MPMState(x=new_x, v=new_v, C=new_C, F=new_F), None

        for _ in range(steps_per_frame):
            state, _ = scan_body(state, None)
        return state

    return jit_frame
