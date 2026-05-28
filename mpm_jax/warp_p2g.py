"""Warp P2G kernels callable from JAX JIT via Warp's experimental FFI."""

import jax
import jax.numpy as jnp
import warp as wp
from warp.jax_experimental import jax_callable, jax_kernel


TILE_SIZE = 64
SUPER_CELL_WIDTH = 2
SUPER_TILE_DIM = SUPER_CELL_WIDTH + 2
SUPER_TILE_NODES = SUPER_TILE_DIM * SUPER_TILE_DIM * SUPER_TILE_DIM


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


@wp.kernel
def _p2g_inline_tile_kernel(
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
    tile_id, lane = wp.tid()

    x_tile = wp.tile_load(
        x, shape=(TILE_SIZE, 3), offset=(tile_id * TILE_SIZE, 0), storage="shared")
    v_tile = wp.tile_load(
        v, shape=(TILE_SIZE, 3), offset=(tile_id * TILE_SIZE, 0), storage="shared")
    C_tile = wp.tile_load(
        C, shape=(TILE_SIZE, 9), offset=(tile_id * TILE_SIZE, 0), storage="shared")
    stress_tile = wp.tile_load(
        stress, shape=(TILE_SIZE, 9), offset=(tile_id * TILE_SIZE, 0), storage="shared")

    px0 = wp.tile_extract(x_tile, lane, 0) * inv_dx
    px1 = wp.tile_extract(x_tile, lane, 1) * inv_dx
    px2 = wp.tile_extract(x_tile, lane, 2) * inv_dx

    b0 = int(wp.floor(px0 - 0.5))
    b1 = int(wp.floor(px1 - 0.5))
    b2 = int(wp.floor(px2 - 0.5))

    fx0 = px0 - float(b0)
    fx1 = px1 - float(b1)
    fx2 = px2 - float(b2)

    C00 = wp.tile_extract(C_tile, lane, 0)
    C01 = wp.tile_extract(C_tile, lane, 1)
    C02 = wp.tile_extract(C_tile, lane, 2)
    C10 = wp.tile_extract(C_tile, lane, 3)
    C11 = wp.tile_extract(C_tile, lane, 4)
    C12 = wp.tile_extract(C_tile, lane, 5)
    C20 = wp.tile_extract(C_tile, lane, 6)
    C21 = wp.tile_extract(C_tile, lane, 7)
    C22 = wp.tile_extract(C_tile, lane, 8)

    s00 = wp.tile_extract(stress_tile, lane, 0)
    s01 = wp.tile_extract(stress_tile, lane, 1)
    s02 = wp.tile_extract(stress_tile, lane, 2)
    s10 = wp.tile_extract(stress_tile, lane, 3)
    s11 = wp.tile_extract(stress_tile, lane, 4)
    s12 = wp.tile_extract(stress_tile, lane, 5)
    s20 = wp.tile_extract(stress_tile, lane, 6)
    s21 = wp.tile_extract(stress_tile, lane, 7)
    s22 = wp.tile_extract(stress_tile, lane, 8)

    vp0 = wp.tile_extract(v_tile, lane, 0)
    vp1 = wp.tile_extract(v_tile, lane, 1)
    vp2 = wp.tile_extract(v_tile, lane, 2)

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


def _p2g_inline_tile_callable(
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
    wp.launch_tiled(
        _p2g_inline_tile_kernel,
        dim=[x.shape[0] // TILE_SIZE],
        inputs=[x, v, C, stress, G, dt, vol, p_mass, inv_dx, dx, grid_mv, grid_m],
        block_dim=TILE_SIZE,
    )


_jax_p2g_inline_tile = jax_callable(
    _p2g_inline_tile_callable,
    num_outputs=2,
    in_out_argnames=["grid_mv", "grid_m"],
)


@wp.kernel
def _p2g_supercell_tile_kernel(
    x: wp.array2d[float],
    v: wp.array2d[float],
    C: wp.array2d[float],
    stress: wp.array2d[float],
    cell_start: wp.array[int],
    G: int,
    dt: float,
    vol: float,
    p_mass: float,
    inv_dx: float,
    dx: float,
    grid_mv: wp.array[float],
    grid_m: wp.array[float],
):
    super_id, lane = wp.tid()

    Gs = G / SUPER_CELL_WIDTH
    si = super_id // (Gs * Gs)
    sj = (super_id // Gs) % Gs
    sk = super_id % Gs

    tile_i = si * SUPER_CELL_WIDTH - 1
    tile_j = sj * SUPER_CELL_WIDTH - 1
    tile_k = sk * SUPER_CELL_WIDTH - 1

    p_start = cell_start[super_id]
    p_end = cell_start[super_id + 1]

    chunk_start = p_start
    while chunk_start < p_end:
        tile_mv0 = wp.tile_zeros(shape=SUPER_TILE_NODES, dtype=float, storage="shared")
        tile_mv1 = wp.tile_zeros(shape=SUPER_TILE_NODES, dtype=float, storage="shared")
        tile_mv2 = wp.tile_zeros(shape=SUPER_TILE_NODES, dtype=float, storage="shared")
        tile_m = wp.tile_zeros(shape=SUPER_TILE_NODES, dtype=float, storage="shared")

        p = chunk_start + lane
        active = p < p_end

        b0 = 0
        b1 = 0
        b2 = 0
        fx0 = 0.0
        fx1 = 0.0
        fx2 = 0.0
        C00 = 0.0
        C01 = 0.0
        C02 = 0.0
        C10 = 0.0
        C11 = 0.0
        C12 = 0.0
        C20 = 0.0
        C21 = 0.0
        C22 = 0.0
        s00 = 0.0
        s01 = 0.0
        s02 = 0.0
        s10 = 0.0
        s11 = 0.0
        s12 = 0.0
        s20 = 0.0
        s21 = 0.0
        s22 = 0.0
        vp0 = 0.0
        vp1 = 0.0
        vp2 = 0.0

        if active:
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

                    gi = wp.clamp(b0 + ox, 0, G - 1)
                    gj = wp.clamp(b1 + oy, 0, G - 1)
                    gk = wp.clamp(b2 + oz, 0, G - 1)

                    ti = gi - tile_i
                    tj = gj - tile_j
                    tk = gk - tile_k
                    tile_idx = ti * SUPER_TILE_DIM * SUPER_TILE_DIM + tj * SUPER_TILE_DIM + tk
                    in_tile = (
                        active
                        and ti >= 0 and ti < SUPER_TILE_DIM
                        and tj >= 0 and tj < SUPER_TILE_DIM
                        and tk >= 0 and tk < SUPER_TILE_DIM
                    )

                    wp.tile_scatter_add(tile_mv0, tile_idx, mv0, in_tile)
                    wp.tile_scatter_add(tile_mv1, tile_idx, mv1, in_tile)
                    wp.tile_scatter_add(tile_mv2, tile_idx, mv2, in_tile)
                    wp.tile_scatter_add(tile_m, tile_idx, mass, in_tile)

                    if active and not in_tile:
                        idx = gi * G * G + gj * G + gk
                        wp.atomic_add(grid_mv, idx * 3 + 0, mv0)
                        wp.atomic_add(grid_mv, idx * 3 + 1, mv1)
                        wp.atomic_add(grid_mv, idx * 3 + 2, mv2)
                        wp.atomic_add(grid_m, idx, mass)

        if lane < SUPER_TILE_NODES:
            smv0 = wp.tile_extract(tile_mv0, lane)
            smv1 = wp.tile_extract(tile_mv1, lane)
            smv2 = wp.tile_extract(tile_mv2, lane)
            sm = wp.tile_extract(tile_m, lane)

            ti = lane // (SUPER_TILE_DIM * SUPER_TILE_DIM)
            tj = (lane // SUPER_TILE_DIM) % SUPER_TILE_DIM
            tk = lane % SUPER_TILE_DIM

            gi = tile_i + ti
            gj = tile_j + tj
            gk = tile_k + tk

            if gi >= 0 and gi < G and gj >= 0 and gj < G and gk >= 0 and gk < G:
                idx = gi * G * G + gj * G + gk
                wp.atomic_add(grid_mv, idx * 3 + 0, smv0)
                wp.atomic_add(grid_mv, idx * 3 + 1, smv1)
                wp.atomic_add(grid_mv, idx * 3 + 2, smv2)
                wp.atomic_add(grid_m, idx, sm)

        chunk_start += TILE_SIZE


def _p2g_supercell_tile_callable(
    x: wp.array2d[float],
    v: wp.array2d[float],
    C: wp.array2d[float],
    stress: wp.array2d[float],
    cell_start: wp.array[int],
    G: int,
    dt: float,
    vol: float,
    p_mass: float,
    inv_dx: float,
    dx: float,
    grid_mv: wp.array[float],
    grid_m: wp.array[float],
):
    Gs = G // SUPER_CELL_WIDTH
    wp.launch_tiled(
        _p2g_supercell_tile_kernel,
        dim=[Gs * Gs * Gs],
        inputs=[
            x, v, C, stress, cell_start,
            G, dt, vol, p_mass, inv_dx, dx,
            grid_mv, grid_m,
        ],
        block_dim=TILE_SIZE,
    )


_jax_p2g_supercell_tile = jax_callable(
    _p2g_supercell_tile_callable,
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


def warp_p2g_inline_tile(x, v, C, stress, num_grids, dt, vol, p_mass, inv_dx, dx):
    """Run inline P2G through a tiled Warp kernel embedded in JAX."""
    n = x.shape[0]
    if n % TILE_SIZE != 0:
        raise RuntimeError(
            f"warp_v2_tile currently requires n_particles divisible by {TILE_SIZE}; "
            f"got {n}."
        )
    g3 = num_grids ** 3
    C_flat = C.reshape(n, 9)
    stress_flat = stress.reshape(n, 9)

    grid_mv0 = jnp.zeros((g3 * 3,), dtype=jnp.float32)
    grid_m0 = jnp.zeros((g3,), dtype=jnp.float32)

    grid_mv_flat, grid_m = _jax_p2g_inline_tile(
        x, v, C_flat, stress_flat,
        int(num_grids),
        float(dt),
        float(vol),
        float(p_mass),
        float(inv_dx),
        float(dx),
        grid_mv0,
        grid_m0,
    )
    return grid_mv_flat.reshape((g3, 3)), grid_m


def warp_p2g_supercell_tile(x_sorted, v_sorted, C_sorted, stress_sorted, cell_start,
                            num_grids, dt, vol, p_mass, inv_dx, dx):
    """Run super-cell-owned tiled Warp P2G embedded in JAX."""
    g3 = num_grids ** 3
    n = x_sorted.shape[0]
    C_flat = C_sorted.reshape(n, 9)
    stress_flat = stress_sorted.reshape(n, 9)

    grid_mv0 = jnp.zeros((g3 * 3,), dtype=jnp.float32)
    grid_m0 = jnp.zeros((g3,), dtype=jnp.float32)

    grid_mv_flat, grid_m = _jax_p2g_supercell_tile(
        x_sorted, v_sorted, C_flat, stress_flat, cell_start,
        int(num_grids),
        float(dt),
        float(vol),
        float(p_mass),
        float(inv_dx),
        float(dx),
        grid_mv0,
        grid_m0,
    )
    return grid_mv_flat.reshape((g3, 3)), grid_m


def _home_super_cell_id(x, inv_dx, G, sc=SUPER_CELL_WIDTH):
    px = x * inv_dx
    base = jnp.floor(px - 0.5).astype(jnp.int32)
    home = base + 1
    home = jnp.clip(home, 0, G - 1)
    Gs = G // sc
    si = home[:, 0] // sc
    sj = home[:, 1] // sc
    sk = home[:, 2] // sc
    return (si * (Gs * Gs) + sj * Gs + sk).astype(jnp.int32)


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


def build_jit_frame_warp_tile(params, elasticity_fn, plasticity_fn,
                              pre_particle_fn, post_grid_fn, steps_per_frame,
                              use_cuda_g2p=True):
    """Build a fully JIT'd frame using tiled Warp P2G via JAX FFI."""
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
            with jax.named_scope("warp_p2g_tile"):
                grid_mv, grid_m = warp_p2g_inline_tile(
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


def build_jit_frame_warp_supercell_tile(params, elasticity_fn, plasticity_fn,
                                        pre_particle_fn, post_grid_fn,
                                        steps_per_frame,
                                        use_cuda_g2p=True):
    """Build a frame using a super-cell-owned Warp tile P2G kernel."""
    from mpm_jax.solver import (
        MPMState,
        compute_weights_and_indices,
        g2p,
        grid_update as grid_update_fn,
    )

    if params.num_grids % SUPER_CELL_WIDTH != 0:
        raise RuntimeError(
            f"warp_v3_supercell_tile requires num_grids ({params.num_grids}) "
            f"divisible by super-cell width ({SUPER_CELL_WIDTH})."
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

    G = params.num_grids
    Gs = G // SUPER_CELL_WIDTH
    Gs3 = Gs ** 3
    super_boundaries = jnp.arange(Gs3 + 1, dtype=jnp.int32)

    @jax.jit
    def jit_frame(state):
        def scan_body(state, _):
            with jax.named_scope("pre_particle"):
                x, v = pre_particle_fn(state.x, state.v, 0.0)
            with jax.named_scope("elasticity"):
                stress = elasticity_fn(state.F)

            with jax.named_scope("warp_supercell_sort"):
                super_id = _home_super_cell_id(x, params.inv_dx, G, SUPER_CELL_WIDTH)
                order = jnp.argsort(super_id)

                x_s = x[order]
                v_s = v[order]
                C_s = state.C[order]
                stress_s = stress[order]
                F_s = state.F[order]

                super_id_sorted = super_id[order]
                cell_start = jnp.searchsorted(
                    super_id_sorted, super_boundaries
                ).astype(jnp.int32)

            with jax.named_scope("warp_p2g_supercell_tile"):
                grid_mv, grid_m = warp_p2g_supercell_tile(
                    x_s, v_s, C_s, stress_s, cell_start,
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
                        x_s, F_s, grid_v,
                        params.num_grids, params.dt,
                        params.inv_dx, params.dx, params.clip_bound,
                    )
                else:
                    weight, dweight, dpos, index = compute_weights_and_indices(
                        x_s, params.inv_dx, params.dx, params.num_grids)
                    new_x, new_v, new_C, new_F = g2p(
                        grid_v, weight, dweight, dpos, index,
                        F_s, x_s, params.dt, params.inv_dx, params.clip_bound)

            with jax.named_scope("plasticity"):
                new_F = plasticity_fn(new_F)
            return MPMState(x=new_x, v=new_v, C=new_C, F=new_F), None

        for _ in range(steps_per_frame):
            state, _ = scan_body(state, None)
        return state

    return jit_frame
