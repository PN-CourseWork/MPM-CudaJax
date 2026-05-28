"""Pure-Warp graph-captured MPM prototype.

This is intentionally narrow: CorotatedElasticityJacobi-style jelly with
identity plasticity and a sticky floor boundary. It exists to explore the
Warp-native tiled/graph path without JAX driving the timestep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp
import warp.utils

TILE_SIZE = 64
INDEXED_TILE_SIZE = 128
SUPER_CELL_WIDTH = 2
SUPER_TILE_DIM = SUPER_CELL_WIDTH + 2
SUPER_TILE_NODES = SUPER_TILE_DIM * SUPER_TILE_DIM * SUPER_TILE_DIM


@dataclass
class WarpBonusResult:
    elapsed_s: float
    ms_per_step: float
    steps_per_sec: float


@dataclass
class WarpGraphTimingResult(WarpBonusResult):
    phase_total_ms: dict[str, float]
    phase_ms_per_frame: dict[str, float]
    phase_ms_per_step: dict[str, float]


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


@wp.func
def _home_super_cell_id_vec(x: wp.vec3, inv_dx: float, G: int):
    return _home_super_cell_id_vec_width(x, inv_dx, G, SUPER_CELL_WIDTH)


@wp.func
def _home_super_cell_id_vec_width(x: wp.vec3, inv_dx: float, G: int, width: int):
    px = x * inv_dx
    b0 = int(wp.floor(px[0] - 0.5)) + 1
    b1 = int(wp.floor(px[1] - 0.5)) + 1
    b2 = int(wp.floor(px[2] - 0.5)) + 1
    b0 = wp.clamp(b0, 0, G - 1)
    b1 = wp.clamp(b1, 0, G - 1)
    b2 = wp.clamp(b2, 0, G - 1)
    Gs = G // width
    return (
        (b0 // width) * Gs * Gs
        + (b1 // width) * Gs
        + (b2 // width)
    )


@wp.func
def _corotated_stress(F: wp.mat33, mu: float, la: float):
    U = wp.mat33()
    sigma = wp.vec3()
    V = wp.mat33()
    U, sigma, V = wp.svd3(F)
    R = U * wp.transpose(V)
    J = sigma[0] * sigma[1] * sigma[2]
    I = wp.identity(n=3, dtype=float)
    return (F - R) * (2.0 * mu) * wp.transpose(F) + I * (la * J * (J - 1.0))


@wp.kernel
def _zero_float_kernel(a: wp.array[float]):
    i = wp.tid()
    a[i] = 0.0


@wp.kernel
def _zero_vec3_kernel(a: wp.array[wp.vec3]):
    i = wp.tid()
    a[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _zero_int_kernel(a: wp.array[int]):
    i = wp.tid()
    a[i] = 0


@wp.kernel
def _count_supercells_kernel(
    x: wp.array[wp.vec3],
    counts: wp.array[int],
    G: int,
    inv_dx: float,
    width: int,
):
    p = wp.tid()
    sid = _home_super_cell_id_vec_width(x[p], inv_dx, G, width)
    wp.atomic_add(counts, sid, 1)


@wp.kernel
def _prefix_to_cell_start_kernel(prefix_inclusive: wp.array[int], cell_start: wp.array[int]):
    i = wp.tid()
    if i == 0:
        cell_start[0] = 0
    cell_start[i + 1] = prefix_inclusive[i]


@wp.kernel
def _scatter_supercell_order_kernel(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    C: wp.array[wp.mat33],
    F: wp.array[wp.mat33],
    cell_start: wp.array[int],
    cursor: wp.array[int],
    x_s: wp.array[wp.vec3],
    v_s: wp.array[wp.vec3],
    C_s: wp.array[wp.mat33],
    F_s: wp.array[wp.mat33],
    G: int,
    inv_dx: float,
    width: int,
):
    p = wp.tid()
    sid = _home_super_cell_id_vec_width(x[p], inv_dx, G, width)
    local = wp.atomic_add(cursor, sid, 1)
    dst = cell_start[sid] + local
    x_s[dst] = x[p]
    v_s[dst] = v[p]
    C_s[dst] = C[p]
    F_s[dst] = F[p]


@wp.kernel
def _scatter_supercell_ids_kernel(
    x: wp.array[wp.vec3],
    cell_start: wp.array[int],
    cursor: wp.array[int],
    ids: wp.array[int],
    G: int,
    inv_dx: float,
    width: int,
):
    p = wp.tid()
    sid = _home_super_cell_id_vec_width(x[p], inv_dx, G, width)
    local = wp.atomic_add(cursor, sid, 1)
    ids[cell_start[sid] + local] = p


@wp.kernel
def _compute_stress_kernel(
    F: wp.array[wp.mat33],
    stress: wp.array[wp.mat33],
    mu: float,
    la: float,
):
    p = wp.tid()
    stress[p] = _corotated_stress(F[p], mu, la)


@wp.kernel
def _p2g_supercell_stress_tile_kernel(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    C: wp.array[wp.mat33],
    F: wp.array[wp.mat33],
    cell_start: wp.array[int],
    G: int,
    dt: float,
    vol: float,
    p_mass: float,
    inv_dx: float,
    dx: float,
    mu: float,
    la: float,
    grid_mv: wp.array[wp.vec3],
    grid_m: wp.array[float],
):
    super_id, lane = wp.tid()
    Gs = G // SUPER_CELL_WIDTH
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
        xp = wp.vec3(0.0, 0.0, 0.0)
        vp = wp.vec3(0.0, 0.0, 0.0)
        Cp = wp.mat33()
        stress = wp.mat33()
        if active:
            xp = x[p]
            vp = v[p]
            Cp = C[p]
            stress = _corotated_stress(F[p], mu, la)

        px = xp * inv_dx
        b0 = int(wp.floor(px[0] - 0.5))
        b1 = int(wp.floor(px[1] - 0.5))
        b2 = int(wp.floor(px[2] - 0.5))
        fx0 = px[0] - float(b0)
        fx1 = px[1] - float(b1)
        fx2 = px[2] - float(b2)

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
                    dweight = wp.vec3(
                        inv_dx * dwx * wy * wz,
                        inv_dx * wx * dwy * wz,
                        inv_dx * wx * wy * dwz,
                    )
                    dpos = wp.vec3(
                        (float(ox) - fx0) * dx,
                        (float(oy) - fx1) * dx,
                        (float(oz) - fx2) * dx,
                    )
                    mv = -dt * vol * (stress * dweight) + p_mass * weight * (vp + Cp * dpos)
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
                    wp.tile_scatter_add(tile_mv0, tile_idx, mv[0], in_tile)
                    wp.tile_scatter_add(tile_mv1, tile_idx, mv[1], in_tile)
                    wp.tile_scatter_add(tile_mv2, tile_idx, mv[2], in_tile)
                    wp.tile_scatter_add(tile_m, tile_idx, mass, in_tile)

                    if active and not in_tile:
                        idx = gi * G * G + gj * G + gk
                        wp.atomic_add(grid_mv, idx, mv)
                        wp.atomic_add(grid_m, idx, mass)

        if lane < SUPER_TILE_NODES:
            smv = wp.vec3(
                wp.tile_extract(tile_mv0, lane),
                wp.tile_extract(tile_mv1, lane),
                wp.tile_extract(tile_mv2, lane),
            )
            sm = wp.tile_extract(tile_m, lane)
            ti = lane // (SUPER_TILE_DIM * SUPER_TILE_DIM)
            tj = (lane // SUPER_TILE_DIM) % SUPER_TILE_DIM
            tk = lane % SUPER_TILE_DIM
            gi = tile_i + ti
            gj = tile_j + tj
            gk = tile_k + tk
            if gi >= 0 and gi < G and gj >= 0 and gj < G and gk >= 0 and gk < G:
                idx = gi * G * G + gj * G + gk
                wp.atomic_add(grid_mv, idx, smv)
                wp.atomic_add(grid_m, idx, sm)

        chunk_start += TILE_SIZE


@wp.kernel
def _p2g_supercell_stress_tile_indexed_kernel(
    ids: wp.array[int],
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    C: wp.array[wp.mat33],
    stress_in: wp.array[wp.mat33],
    cell_start: wp.array[int],
    G: int,
    dt: float,
    vol: float,
    p_mass: float,
    inv_dx: float,
    dx: float,
    grid_mv: wp.array[wp.vec3],
    grid_m: wp.array[float],
):
    super_id, lane = wp.tid()
    Gs = G // SUPER_CELL_WIDTH
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
        xp = wp.vec3(0.0, 0.0, 0.0)
        vp = wp.vec3(0.0, 0.0, 0.0)
        Cp = wp.mat33()
        stress = wp.mat33()
        if active:
            src = ids[p]
            xp = x[src]
            vp = v[src]
            Cp = C[src]
            stress = stress_in[src]

        px = xp * inv_dx
        b0 = int(wp.floor(px[0] - 0.5))
        b1 = int(wp.floor(px[1] - 0.5))
        b2 = int(wp.floor(px[2] - 0.5))
        fx0 = px[0] - float(b0)
        fx1 = px[1] - float(b1)
        fx2 = px[2] - float(b2)

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
                    dweight = wp.vec3(
                        inv_dx * dwx * wy * wz,
                        inv_dx * wx * dwy * wz,
                        inv_dx * wx * wy * dwz,
                    )
                    dpos = wp.vec3(
                        (float(ox) - fx0) * dx,
                        (float(oy) - fx1) * dx,
                        (float(oz) - fx2) * dx,
                    )
                    mv = -dt * vol * (stress * dweight) + p_mass * weight * (vp + Cp * dpos)
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
                    wp.tile_scatter_add(tile_mv0, tile_idx, mv[0], in_tile)
                    wp.tile_scatter_add(tile_mv1, tile_idx, mv[1], in_tile)
                    wp.tile_scatter_add(tile_mv2, tile_idx, mv[2], in_tile)
                    wp.tile_scatter_add(tile_m, tile_idx, mass, in_tile)

                    if active and not in_tile:
                        idx = gi * G * G + gj * G + gk
                        wp.atomic_add(grid_mv, idx, mv)
                        wp.atomic_add(grid_m, idx, mass)

        if lane < SUPER_TILE_NODES:
            smv = wp.vec3(
                wp.tile_extract(tile_mv0, lane),
                wp.tile_extract(tile_mv1, lane),
                wp.tile_extract(tile_mv2, lane),
            )
            sm = wp.tile_extract(tile_m, lane)
            ti = lane // (SUPER_TILE_DIM * SUPER_TILE_DIM)
            tj = (lane // SUPER_TILE_DIM) % SUPER_TILE_DIM
            tk = lane % SUPER_TILE_DIM
            gi = tile_i + ti
            gj = tile_j + tj
            gk = tile_k + tk
            if gi >= 0 and gi < G and gj >= 0 and gj < G and gk >= 0 and gk < G:
                idx = gi * G * G + gj * G + gk
                wp.atomic_add(grid_mv, idx, smv)
                wp.atomic_add(grid_m, idx, sm)

        chunk_start += INDEXED_TILE_SIZE


@wp.kernel
def _p2g_supercell_stress_tile_indexed_inline_kernel(
    ids: wp.array[int],
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    C: wp.array[wp.mat33],
    F: wp.array[wp.mat33],
    cell_start: wp.array[int],
    G: int,
    dt: float,
    vol: float,
    p_mass: float,
    inv_dx: float,
    dx: float,
    mu: float,
    la: float,
    grid_mv: wp.array[wp.vec3],
    grid_m: wp.array[float],
):
    super_id, lane = wp.tid()
    Gs = G // SUPER_CELL_WIDTH
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
        xp = wp.vec3(0.0, 0.0, 0.0)
        vp = wp.vec3(0.0, 0.0, 0.0)
        Cp = wp.mat33()
        stress = wp.mat33()
        if active:
            src = ids[p]
            xp = x[src]
            vp = v[src]
            Cp = C[src]
            stress = _corotated_stress(F[src], mu, la)

        px = xp * inv_dx
        b0 = int(wp.floor(px[0] - 0.5))
        b1 = int(wp.floor(px[1] - 0.5))
        b2 = int(wp.floor(px[2] - 0.5))
        fx0 = px[0] - float(b0)
        fx1 = px[1] - float(b1)
        fx2 = px[2] - float(b2)

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
                    dweight = wp.vec3(
                        inv_dx * dwx * wy * wz,
                        inv_dx * wx * dwy * wz,
                        inv_dx * wx * wy * dwz,
                    )
                    dpos = wp.vec3(
                        (float(ox) - fx0) * dx,
                        (float(oy) - fx1) * dx,
                        (float(oz) - fx2) * dx,
                    )
                    mv = -dt * vol * (stress * dweight) + p_mass * weight * (vp + Cp * dpos)
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
                    wp.tile_scatter_add(tile_mv0, tile_idx, mv[0], in_tile)
                    wp.tile_scatter_add(tile_mv1, tile_idx, mv[1], in_tile)
                    wp.tile_scatter_add(tile_mv2, tile_idx, mv[2], in_tile)
                    wp.tile_scatter_add(tile_m, tile_idx, mass, in_tile)

                    if active and not in_tile:
                        idx = gi * G * G + gj * G + gk
                        wp.atomic_add(grid_mv, idx, mv)
                        wp.atomic_add(grid_m, idx, mass)

        if lane < SUPER_TILE_NODES:
            smv = wp.vec3(
                wp.tile_extract(tile_mv0, lane),
                wp.tile_extract(tile_mv1, lane),
                wp.tile_extract(tile_mv2, lane),
            )
            sm = wp.tile_extract(tile_m, lane)
            ti = lane // (SUPER_TILE_DIM * SUPER_TILE_DIM)
            tj = (lane // SUPER_TILE_DIM) % SUPER_TILE_DIM
            tk = lane % SUPER_TILE_DIM
            gi = tile_i + ti
            gj = tile_j + tj
            gk = tile_k + tk
            if gi >= 0 and gi < G and gj >= 0 and gj < G and gk >= 0 and gk < G:
                idx = gi * G * G + gj * G + gk
                wp.atomic_add(grid_mv, idx, smv)
                wp.atomic_add(grid_m, idx, sm)

        chunk_start += INDEXED_TILE_SIZE


@wp.kernel
def _grid_update_kernel(
    grid_mv: wp.array[wp.vec3],
    grid_m: wp.array[float],
    G: int,
    dt: float,
    damping: float,
    gravity: wp.vec3,
    floor_bound: float,
):
    i = wp.tid()
    m = grid_m[i]
    gv = wp.vec3(0.0, 0.0, 0.0)
    if m > 1.0e-15:
        gv = grid_mv[i] / m
    gv = damping * (gv + dt * gravity)

    z = i % G
    if float(z) / float(G) < floor_bound:
        gv = wp.vec3(0.0, 0.0, 0.0)
    grid_mv[i] = gv


@wp.kernel
def _g2p_kernel(
    x: wp.array[wp.vec3],
    F: wp.array[wp.mat33],
    grid_v: wp.array[wp.vec3],
    x_out: wp.array[wp.vec3],
    v_out: wp.array[wp.vec3],
    C_out: wp.array[wp.mat33],
    F_out: wp.array[wp.mat33],
    G: int,
    dt: float,
    inv_dx: float,
    dx: float,
    clip_bound: float,
):
    p = wp.tid()
    xp = x[p]
    Fp = F[p]
    px = xp * inv_dx
    b0 = int(wp.floor(px[0] - 0.5))
    b1 = int(wp.floor(px[1] - 0.5))
    b2 = int(wp.floor(px[2] - 0.5))
    fx0 = px[0] - float(b0)
    fx1 = px[1] - float(b1)
    fx2 = px[2] - float(b2)
    new_v = wp.vec3(0.0, 0.0, 0.0)
    new_C = wp.mat33()
    grad_v = wp.mat33()

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
                dweight = wp.vec3(
                    inv_dx * dwx * wy * wz,
                    inv_dx * wx * dwy * wz,
                    inv_dx * wx * wy * dwz,
                )
                dpos = wp.vec3(
                    (float(ox) - fx0) * dx,
                    (float(oy) - fx1) * dx,
                    (float(oz) - fx2) * dx,
                )
                gi = wp.clamp(b0 + ox, 0, G - 1)
                gj = wp.clamp(b1 + oy, 0, G - 1)
                gk = wp.clamp(b2 + oz, 0, G - 1)
                idx = gi * G * G + gj * G + gk
                gv = grid_v[idx]
                new_v += weight * gv
                new_C += wp.outer(gv, dpos) * weight
                grad_v += wp.outer(gv, dweight)

    new_C = new_C * (4.0 * inv_dx * inv_dx)
    xn = xp + new_v * dt
    xn = wp.vec3(
        wp.clamp(xn[0], clip_bound, 1.0 - clip_bound),
        wp.clamp(xn[1], clip_bound, 1.0 - clip_bound),
        wp.clamp(xn[2], clip_bound, 1.0 - clip_bound),
    )
    Fn = Fp + dt * (grad_v * Fp)
    for i in range(3):
        for j in range(3):
            Fn[i, j] = wp.clamp(Fn[i, j], -2.0, 2.0)
    x_out[p] = xn
    v_out[p] = new_v
    C_out[p] = new_C
    F_out[p] = Fn


@wp.kernel
def _g2p_indexed_kernel(
    ids: wp.array[int],
    x: wp.array[wp.vec3],
    F: wp.array[wp.mat33],
    grid_v: wp.array[wp.vec3],
    x_out: wp.array[wp.vec3],
    v_out: wp.array[wp.vec3],
    C_out: wp.array[wp.mat33],
    F_out: wp.array[wp.mat33],
    G: int,
    dt: float,
    inv_dx: float,
    dx: float,
    clip_bound: float,
):
    p = wp.tid()
    src = ids[p]
    xp = x[src]
    Fp = F[src]
    px = xp * inv_dx
    b0 = int(wp.floor(px[0] - 0.5))
    b1 = int(wp.floor(px[1] - 0.5))
    b2 = int(wp.floor(px[2] - 0.5))
    fx0 = px[0] - float(b0)
    fx1 = px[1] - float(b1)
    fx2 = px[2] - float(b2)
    new_v = wp.vec3(0.0, 0.0, 0.0)
    new_C = wp.mat33()
    grad_v = wp.mat33()

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
                dweight = wp.vec3(
                    inv_dx * dwx * wy * wz,
                    inv_dx * wx * dwy * wz,
                    inv_dx * wx * wy * dwz,
                )
                dpos = wp.vec3(
                    (float(ox) - fx0) * dx,
                    (float(oy) - fx1) * dx,
                    (float(oz) - fx2) * dx,
                )
                gi = wp.clamp(b0 + ox, 0, G - 1)
                gj = wp.clamp(b1 + oy, 0, G - 1)
                gk = wp.clamp(b2 + oz, 0, G - 1)
                idx = gi * G * G + gj * G + gk
                gv = grid_v[idx]
                new_v += weight * gv
                new_C += wp.outer(gv, dpos) * weight
                grad_v += wp.outer(gv, dweight)

    new_C = new_C * (4.0 * inv_dx * inv_dx)
    xn = xp + new_v * dt
    xn = wp.vec3(
        wp.clamp(xn[0], clip_bound, 1.0 - clip_bound),
        wp.clamp(xn[1], clip_bound, 1.0 - clip_bound),
        wp.clamp(xn[2], clip_bound, 1.0 - clip_bound),
    )
    Fn = Fp + dt * (grad_v * Fp)
    for i in range(3):
        for j in range(3):
            Fn[i, j] = wp.clamp(Fn[i, j], -2.0, 2.0)
    x_out[p] = xn
    v_out[p] = new_v
    C_out[p] = new_C
    F_out[p] = Fn


class WarpBonusSimulator:
    def __init__(
        self,
        particles_np,
        cfg,
        *,
        indexed_sort: bool = False,
        precompute_stress: bool = True,
    ):
        wp.init()
        self.device = wp.get_device("cuda:0")
        self.n = int(cfg.sim.n_particles)
        self.G = int(cfg.sim.num_grids)
        self.G3 = self.G ** 3
        self.super_cell_width = SUPER_CELL_WIDTH
        self.tile_size = INDEXED_TILE_SIZE if indexed_sort else TILE_SIZE
        self.Gs = self.G // self.super_cell_width
        self.Gs3 = self.Gs ** 3
        self.steps_per_frame = int(cfg.sim.steps_per_frame)
        self.dt = float(cfg.sim.dt)
        self.dx = 1.0 / self.G
        self.inv_dx = float(self.G)
        self.clip_bound = float(cfg.sim.clip_bound) * self.dx
        self.damping = float(cfg.sim.damping)
        self.gravity = wp.vec3(*[float(x) for x in cfg.sim.gravity])
        self.floor_bound = 0.02
        vol = float(np.prod(np.array(cfg.sim.size, dtype=np.float32))) / self.n
        self.vol = vol
        self.p_mass = float(cfg.sim.rho) * vol
        E = float(cfg.material.elasticity.get("E", 2e6))
        nu = float(cfg.material.elasticity.get("nu", 0.4))
        self.mu = E / (2.0 * (1.0 + nu))
        self.la = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        v0 = np.broadcast_to(np.array(cfg.sim.initial_velocity, dtype=np.float32), (self.n, 3)).copy()
        C0 = np.zeros((self.n, 3, 3), dtype=np.float32)
        F0 = np.broadcast_to(np.eye(3, dtype=np.float32), (self.n, 3, 3)).copy()

        self.x = wp.array(particles_np.astype(np.float32), dtype=wp.vec3, device=self.device)
        self.v = wp.array(v0, dtype=wp.vec3, device=self.device)
        self.C = wp.array(C0, dtype=wp.mat33, device=self.device)
        self.F = wp.array(F0, dtype=wp.mat33, device=self.device)
        self.x2 = wp.empty_like(self.x)
        self.v2 = wp.empty_like(self.v)
        self.C2 = wp.empty_like(self.C)
        self.F2 = wp.empty_like(self.F)
        self.xs = None
        self.vs = None
        self.Cs = None
        self.Fs = None
        self.stress = None
        self.precompute_stress = indexed_sort and precompute_stress
        if self.precompute_stress:
            self.stress = wp.empty_like(self.F)
        elif not indexed_sort:
            self.xs = wp.empty_like(self.x)
            self.vs = wp.empty_like(self.v)
            self.Cs = wp.empty_like(self.C)
            self.Fs = wp.empty_like(self.F)
        self.grid_mv = wp.zeros(self.G3, dtype=wp.vec3, device=self.device)
        self.grid_m = wp.zeros(self.G3, dtype=float, device=self.device)
        self.counts = wp.zeros(self.Gs3, dtype=int, device=self.device)
        self.prefix = wp.zeros(self.Gs3, dtype=int, device=self.device)
        self.cell_start = wp.zeros(self.Gs3 + 1, dtype=int, device=self.device)
        self.cursor = wp.zeros(self.Gs3, dtype=int, device=self.device)
        self.graphs = []
        self.graph_outputs = []
        self.graph_timing_events = []
        self.graph_index = 0
        self.compiled = False
        self.indexed_sort = indexed_sort
        self._capture_timing_events = None
        if indexed_sort:
            self.p2g_kernel = (
                _p2g_supercell_stress_tile_indexed_kernel
                if self.precompute_stress
                else _p2g_supercell_stress_tile_indexed_inline_kernel
            )
        else:
            self.p2g_kernel = _p2g_supercell_stress_tile_kernel
        self.ids = wp.empty(self.n, dtype=int, device=self.device)

        wp.load_module(device=self.device)

    def _state_tuple(self):
        return self.x, self.v, self.C, self.F, self.x2, self.v2, self.C2, self.F2

    def _set_state_tuple(self, state):
        self.x, self.v, self.C, self.F, self.x2, self.v2, self.C2, self.F2 = state

    def _timing_begin(self, label: str):
        if self._capture_timing_events is None:
            return None
        start = wp.Event(enable_timing=True)
        end = wp.Event(enable_timing=True)
        self._capture_timing_events.append((label, start, end))
        wp.record_event(start)
        return end

    def _timing_end(self, end):
        if end is not None:
            wp.record_event(end)

    def _substep(self):
        evt = self._timing_begin("bin")
        wp.launch(_zero_int_kernel, dim=self.Gs3, inputs=[self.counts], device=self.device)
        wp.launch(
            _count_supercells_kernel,
            dim=self.n,
            inputs=[self.x, self.counts, self.G, self.inv_dx, self.super_cell_width],
            device=self.device,
        )
        warp.utils.array_scan(self.counts, self.prefix, inclusive=True)
        wp.launch(_prefix_to_cell_start_kernel, dim=self.Gs3, inputs=[self.prefix, self.cell_start], device=self.device)
        wp.launch(_zero_int_kernel, dim=self.Gs3, inputs=[self.cursor], device=self.device)
        if self.indexed_sort:
            wp.launch(
                _scatter_supercell_ids_kernel,
                dim=self.n,
                inputs=[
                    self.x, self.cell_start, self.cursor, self.ids,
                    self.G, self.inv_dx, self.super_cell_width,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                _scatter_supercell_order_kernel,
                dim=self.n,
                inputs=[
                    self.x, self.v, self.C, self.F,
                    self.cell_start, self.cursor,
                    self.xs, self.vs, self.Cs, self.Fs,
                    self.G, self.inv_dx, self.super_cell_width,
                ],
                device=self.device,
            )
        self._timing_end(evt)

        evt = self._timing_begin("zero_grid")
        wp.launch(_zero_float_kernel, dim=self.G3, inputs=[self.grid_m], device=self.device)
        wp.launch(_zero_vec3_kernel, dim=self.G3, inputs=[self.grid_mv], device=self.device)
        self._timing_end(evt)

        if self.indexed_sort:
            if self.precompute_stress:
                evt = self._timing_begin("stress")
                wp.launch(
                    _compute_stress_kernel,
                    dim=self.n,
                    inputs=[self.F, self.stress, self.mu, self.la],
                    device=self.device,
                )
                self._timing_end(evt)
                p2g_inputs = [
                    self.ids, self.x, self.v, self.C, self.stress, self.cell_start,
                    self.G, self.dt, self.vol, self.p_mass, self.inv_dx, self.dx,
                    self.grid_mv, self.grid_m,
                ]
            else:
                p2g_inputs = [
                    self.ids, self.x, self.v, self.C, self.F, self.cell_start,
                    self.G, self.dt, self.vol, self.p_mass, self.inv_dx, self.dx,
                    self.mu, self.la, self.grid_mv, self.grid_m,
                ]
            evt = self._timing_begin("p2g")
            wp.launch_tiled(
                self.p2g_kernel,
                dim=[self.Gs3],
                inputs=p2g_inputs,
                block_dim=self.tile_size,
                device=self.device,
            )
            self._timing_end(evt)
        else:
            evt = self._timing_begin("p2g")
            wp.launch_tiled(
                self.p2g_kernel,
                dim=[self.Gs3],
                inputs=[
                    self.xs, self.vs, self.Cs, self.Fs, self.cell_start,
                    self.G, self.dt, self.vol, self.p_mass, self.inv_dx, self.dx,
                    self.mu, self.la, self.grid_mv, self.grid_m,
                ],
                block_dim=self.tile_size,
                device=self.device,
            )
            self._timing_end(evt)
        evt = self._timing_begin("grid_update")
        wp.launch(
            _grid_update_kernel,
            dim=self.G3,
            inputs=[
                self.grid_mv, self.grid_m, self.G, self.dt, self.damping,
                self.gravity, self.floor_bound,
            ],
            device=self.device,
        )
        self._timing_end(evt)
        if self.indexed_sort:
            evt = self._timing_begin("g2p")
            wp.launch(
                _g2p_indexed_kernel,
                dim=self.n,
                inputs=[
                    self.ids, self.x, self.F, self.grid_mv,
                    self.x2, self.v2, self.C2, self.F2,
                    self.G, self.dt, self.inv_dx, self.dx, self.clip_bound,
                ],
                device=self.device,
            )
            self._timing_end(evt)
        else:
            evt = self._timing_begin("g2p")
            wp.launch(
                _g2p_kernel,
                dim=self.n,
                inputs=[
                    self.xs, self.Fs, self.grid_mv,
                    self.x2, self.v2, self.C2, self.F2,
                    self.G, self.dt, self.inv_dx, self.dx, self.clip_bound,
                ],
                device=self.device,
            )
            self._timing_end(evt)
        self.x, self.x2 = self.x2, self.x
        self.v, self.v2 = self.v2, self.v
        self.C, self.C2 = self.C2, self.C
        self.F, self.F2 = self.F2, self.F

    def _capture_one_frame(self, timing: bool = False):
        timing_events = [] if timing else None
        self._capture_timing_events = timing_events
        with wp.ScopedCapture(device=self.device, force_module_load=True) as cap:
            for _ in range(self.steps_per_frame):
                self._substep()
        self._capture_timing_events = None
        return cap.graph, self._state_tuple(), timing_events or []

    def capture_frame(self, timing: bool = False):
        if not self.compiled:
            self._substep()
            wp.synchronize_device(self.device)
            self.compiled = True

        self.graphs = []
        self.graph_outputs = []
        self.graph_timing_events = []
        self.graph_index = 0

        start = self._state_tuple()
        graph, end, timing_events = self._capture_one_frame(timing=timing)
        self.graphs.append(graph)
        self.graph_outputs.append(end)
        self.graph_timing_events.append(timing_events)

        if end[0] is not start[0]:
            graph, end, timing_events = self._capture_one_frame(timing=timing)
            self.graphs.append(graph)
            self.graph_outputs.append(end)
            self.graph_timing_events.append(timing_events)

    def warmup(self):
        self.capture_frame()
        self._launch_frame()
        wp.synchronize_device(self.device)

    def launch_frame(self):
        if not self.graphs:
            self.capture_frame()
        self._launch_frame()

    def _launch_frame(self):
        launched_index = self.graph_index
        wp.capture_launch(self.graphs[self.graph_index])
        self._set_state_tuple(self.graph_outputs[self.graph_index])
        if len(self.graphs) == 2:
            self.graph_index = 1 - self.graph_index
        return launched_index

    def run_frames(self, num_frames: int):
        import time
        if not self.graphs:
            self.capture_frame()
        wp.synchronize_device(self.device)
        t0 = time.perf_counter()
        for _ in range(num_frames):
            self._launch_frame()
        wp.synchronize_device(self.device)
        elapsed = time.perf_counter() - t0
        total_steps = num_frames * self.steps_per_frame
        return WarpBonusResult(
            elapsed_s=elapsed,
            ms_per_step=elapsed * 1000.0 / total_steps,
            steps_per_sec=total_steps / elapsed,
        )

    def _read_graph_timing_ms(self, graph_index: int):
        phase_total_ms = {}
        for label, start, end in self.graph_timing_events[graph_index]:
            phase_total_ms[label] = phase_total_ms.get(label, 0.0) + wp.get_event_elapsed_time(start, end)
        return phase_total_ms

    def run_frames_with_graph_timing(self, num_frames: int):
        if not self.graphs or not self.graph_timing_events or not self.graph_timing_events[0]:
            self.capture_frame(timing=True)
        wp.synchronize_device(self.device)
        phase_total_ms = {}
        import time
        t0 = time.perf_counter()
        for _ in range(num_frames):
            graph_index = self._launch_frame()
            wp.synchronize_device(self.device)
            frame_phase_ms = self._read_graph_timing_ms(graph_index)
            for label, elapsed_ms in frame_phase_ms.items():
                phase_total_ms[label] = phase_total_ms.get(label, 0.0) + elapsed_ms
        elapsed = time.perf_counter() - t0
        total_steps = num_frames * self.steps_per_frame
        return WarpGraphTimingResult(
            elapsed_s=elapsed,
            ms_per_step=elapsed * 1000.0 / total_steps,
            steps_per_sec=total_steps / elapsed,
            phase_total_ms=phase_total_ms,
            phase_ms_per_frame={
                label: elapsed_ms / num_frames
                for label, elapsed_ms in phase_total_ms.items()
            },
            phase_ms_per_step={
                label: elapsed_ms / total_steps
                for label, elapsed_ms in phase_total_ms.items()
            },
        )
