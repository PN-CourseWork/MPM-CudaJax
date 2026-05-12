from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np

class MPMState(NamedTuple):
    x: jax.Array      # (N, 3) positions
    v: jax.Array      # (N, 3) velocities
    C: jax.Array      # (N, 3, 3) APIC affine matrix
    F: jax.Array      # (N, 3, 3) deformation gradient


class StepIntermediates(NamedTuple):
    """Minimal state carried from the P2G stage into the G2P stage.

    Only the post-BC positions and the F that should feed G2P are kept.
    B-spline weight / dweight / dpos / index tensors are NOT cached here -
    they would be (N, 27, *) and at large N (a few million particles)
    materialising them across the JIT boundary blows the GPU memory
    budget. G2P recomputes them from x_post_bc - the math is cheap
    (~50 flops/particle, no SVD) and the savings are ~1100 bytes/particle.
    """
    x_post_bc: jax.Array     # (N, 3) positions after pre-particle BCs
    F_pre_plast: jax.Array   # (N, 3, 3) F that G2P should use as its F_p input

class MPMParams(NamedTuple):
    num_grids: int
    dt: float
    gravity: jax.Array
    dx: float
    inv_dx: float
    clip_bound: float
    damping: float
    vol: float
    p_mass: float
    n_particles: int

# Precomputed 27 offsets for 3x3x3 neighborhood
OFFSET_27 = jnp.array(
    [[i, j, k] for i in range(3) for j in range(3) for k in range(3)],
    dtype=jnp.float32,
)  # (27, 3)

def make_params(
    n_particles: int,
    num_grids: int = 25,
    dt: float = 3e-4,
    gravity: list = [0.0, 0.0, -9.8],
    rho: float = 1000.0,
    clip_bound: float = 0.5,
    damping: float = 1.0,
    center: list = [0.5, 0.5, 0.5],
    size: list = [1.0, 1.0, 1.0],
) -> MPMParams:
    dx = 1.0 / num_grids
    vol = float(np.prod(size)) / n_particles
    return MPMParams(
        num_grids=num_grids,
        dt=dt,
        gravity=jnp.array(gravity, dtype=jnp.float32),
        dx=dx,
        inv_dx=float(num_grids),
        clip_bound=clip_bound * dx,
        damping=damping,
        vol=vol,
        p_mass=rho * vol,
        n_particles=n_particles,
    )


# ---------------------------------------------------------------------------
# Single-particle functions (one thread in CUDA)
# ---------------------------------------------------------------------------

def _single_particle_weights(x_p, inv_dx, dx, num_grids):
    """Compute B-spline weights, gradients, dpos, and indices for one particle.

    Args:
        x_p: (3,) particle position

    Returns:
        weight:  (27,)   scalar weight per stencil node
        dweight: (27, 3) weight gradient per stencil node
        dpos:    (27, 3) offset from particle to each stencil node
        index:   (27,)   flat grid index per stencil node
    """
    px = x_p * inv_dx
    base = jnp.floor(px - 0.5).astype(int)  # (3,)
    fx = px - base.astype(jnp.float32)       # (3,)

    # Quadratic B-spline weights per dimension: (3, 3) -> offset x dim
    w = jnp.stack([
        0.5 * (1.5 - fx) ** 2,
        0.75 - (fx - 1.0) ** 2,
        0.5 * (fx - 0.5) ** 2,
    ])  # (3, 3): [offset_idx, spatial_dim]

    dw = jnp.stack([
        fx - 1.5,
        -2.0 * (fx - 1.0),
        fx - 0.5,
    ])  # (3, 3)

    # 3D tensor product over 27 nodes
    offsets = OFFSET_27.astype(int)  # (27, 3)
    weight = w[offsets[:, 0], 0] * w[offsets[:, 1], 1] * w[offsets[:, 2], 2]  # (27,)

    dweight = inv_dx * jnp.stack([
        dw[offsets[:, 0], 0] *  w[offsets[:, 1], 1] *  w[offsets[:, 2], 2],
         w[offsets[:, 0], 0] * dw[offsets[:, 1], 1] *  w[offsets[:, 2], 2],
         w[offsets[:, 0], 0] *  w[offsets[:, 1], 1] * dw[offsets[:, 2], 2],
    ], axis=-1)  # (27, 3)

    dpos = (OFFSET_27 - fx[None, :]) * dx  # (27, 3)

    # Flat grid indices
    idx_3d = base[None, :] + offsets  # (27, 3)
    index = idx_3d[:, 0] * num_grids * num_grids + idx_3d[:, 1] * num_grids + idx_3d[:, 2]
    index = jnp.clip(index, 0, num_grids ** 3 - 1)  # (27,)

    return weight, dweight, dpos, index


def _single_particle_p2g(v_p, C_p, stress_p, weight, dweight, dpos, dt, vol, p_mass):
    """Compute P2G contributions for one particle (one CUDA thread).

    Args:
        v_p:      (3,)   particle velocity
        C_p:      (3, 3) APIC affine matrix
        stress_p: (3, 3) Kirchhoff stress
        weight:   (27,)  B-spline weights
        dweight:  (27, 3) weight gradients
        dpos:     (27, 3) particle-to-node offsets
        dt, vol, p_mass: scalars

    Returns:
        mv: (27, 3) momentum contribution per stencil node
        m:  (27,)   mass contribution per stencil node
    """
    # Affine momentum: stress term + APIC term
    mv = (
        -dt * vol * (stress_p @ dweight.T).T              # (27, 3)
        + p_mass * weight[:, None] * (v_p[None, :] + (C_p @ dpos.T).T)  # (27, 3)
    )
    m = weight * p_mass  # (27,)
    return mv, m


def _single_particle_g2p(grid_mv, weight, dweight, dpos, index, F_p, x_p, dt, inv_dx, clip_bound):
    """Compute G2P gather for one particle (one CUDA thread).

    Args:
        grid_mv: (G^3, 3) grid velocities (read-only)
        weight:  (27,)  B-spline weights
        dweight: (27, 3) weight gradients
        dpos:    (27, 3) particle-to-node offsets
        index:   (27,)  flat grid indices
        F_p:     (3, 3) deformation gradient
        x_p:     (3,)   particle position
        dt, inv_dx, clip_bound: scalars

    Returns:
        new_x: (3,)   updated position
        new_v: (3,)   updated velocity
        new_C: (3, 3) updated APIC matrix
        new_F: (3, 3) updated deformation gradient
    """
    gv = grid_mv[index]  # (27, 3) — gather from grid
    new_v = (weight[:, None] * gv).sum(axis=0)  # (3,)
    new_C = 4.0 * inv_dx * inv_dx * (weight[:, None, None] * jnp.einsum('ij,ik->ijk', gv, dpos)).sum(axis=0)  # (3, 3)
    grad_v = jnp.einsum('ij,ik->ijk', gv, dweight).sum(axis=0)  # (3, 3)

    new_x = jnp.clip(x_p + new_v * dt, clip_bound, 1.0 - clip_bound)  # (3,)
    new_F = jnp.clip(F_p + dt * grad_v @ F_p, -2.0, 2.0)  # (3, 3)

    return new_x, new_v, new_C, new_F


# ---------------------------------------------------------------------------
# Batched versions via vmap
# ---------------------------------------------------------------------------

# vmap over particles (axis 0), keep grid params as scalars
compute_weights_and_indices = jax.vmap(
    _single_particle_weights,
    in_axes=(0, None, None, None),
)  # (N, 3) -> (N, 27), (N, 27, 3), (N, 27, 3), (N, 27)


def p2g_compute(v, C, stress, weight, dweight, dpos, dt, vol, p_mass):
    """Per-particle P2G computation via vmap (embarrassingly parallel).

    Returns:
        mv: (N, 27, 3) momentum contributions per particle per stencil node
        m:  (N, 27)    mass contributions per particle per stencil node
    """
    return jax.vmap(
        _single_particle_p2g,
        in_axes=(0, 0, 0, 0, 0, 0, None, None, None),
    )(v, C, stress, weight, dweight, dpos, dt, vol, p_mass)


def p2g_scatter(mv, m, index, num_grids):
    """Scatter particle contributions onto the grid (the reduction).

    This is the only non-embarrassingly-parallel operation in the timestep.
    XLA lowers this to atomicAdd on GPU — the primary target for CUDA
    optimisation (shared memory staging, spatial sorting, warp reductions).

    Returns:
        grid_mv: (G^3, 3) grid momentum
        grid_m:  (G^3,)   grid mass
    """
    G = num_grids
    grid_mv = jnp.zeros((G ** 3, 3)).at[index.ravel()].add(mv.reshape(-1, 3))
    grid_m = jnp.zeros((G ** 3,)).at[index.ravel()].add(m.ravel())
    return grid_mv, grid_m


def p2g(v, C, stress, weight, dweight, dpos, index, dt, vol, p_mass, num_grids):
    """Full P2G: compute + scatter. Drop-in compatible with existing interface."""
    mv, m = p2g_compute(v, C, stress, weight, dweight, dpos, dt, vol, p_mass)
    return p2g_scatter(mv, m, index, num_grids)


@jax.jit
def grid_update(grid_mv, grid_m, gravity, dt, damping):
    """Normalize momentum by mass, apply gravity and damping.

    Embarrassingly parallel over grid nodes.
    """
    valid = grid_m > 1e-15
    grid_mv = jnp.where(valid[:, None], grid_mv / grid_m[:, None], grid_mv)
    grid_mv = damping * (grid_mv + dt * gravity)
    return grid_mv


def g2p(grid_mv, weight, dweight, dpos, index, F, x, dt, inv_dx, clip_bound):
    """G2P gather via vmap (embarrassingly parallel over particles)."""
    return jax.vmap(
        _single_particle_g2p,
        in_axes=(None, 0, 0, 0, 0, 0, 0, None, None, None),
    )(grid_mv, weight, dweight, dpos, index, F, x, dt, inv_dx, clip_bound)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def step(params, state, stress, pre_particle_fn, post_grid_fn, time, p2g_fn=None):
    """One full P2G2P step.

    Pure function — safe to JIT when closures are captured.
    Optionally accepts a custom p2g_fn to swap the scatter implementation.
    """
    p2g_fn = p2g_fn or p2g

    # Pre-particle BCs
    x, v = pre_particle_fn(state.x, state.v, time)

    # Weights (vmap over particles)
    weight, dweight, dpos, index = compute_weights_and_indices(
        x, params.inv_dx, params.dx, params.num_grids)

    # P2G: compute + scatter
    grid_mv, grid_m = p2g_fn(v, state.C, stress, weight, dweight, dpos, index,
                              params.dt, params.vol, params.p_mass, params.num_grids)

    # Grid update
    grid_mv = grid_update(grid_mv, grid_m, params.gravity, params.dt, params.damping)

    # Post-grid BCs
    grid_mv = post_grid_fn(grid_mv, grid_m, time)

    # G2P (vmap over particles)
    new_x, new_v, new_C, new_F = g2p(grid_mv, weight, dweight, dpos, index,
                                       state.F, x, params.dt, params.inv_dx,
                                       params.clip_bound)

    return MPMState(x=new_x, v=new_v, C=new_C, F=new_F)


def build_jit_step(params, elasticity_fn, plasticity_fn,
                   pre_particle_fn, post_grid_fn, p2g_fn=None):
    """Build a JIT-compiled single-step function.

    Captures all closures at trace time so the entire timestep compiles
    to one XLA program.
    """
    _p2g_fn = p2g_fn or p2g

    @jax.jit
    def jit_step(state):
        stress = elasticity_fn(state.F)
        state = step(params, state, stress, pre_particle_fn, post_grid_fn,
                     0.0, p2g_fn=_p2g_fn)
        return state._replace(F=plasticity_fn(state.F))

    return jit_step


def build_jit_frame(params, elasticity_fn, plasticity_fn,
                    pre_particle_fn, post_grid_fn, steps_per_frame, p2g_fn=None):
    """Build a JIT-compiled frame function using lax.scan.

    One XLA program, zero Python loop overhead.
    """
    _p2g_fn = p2g_fn or p2g

    @jax.jit
    def jit_frame(state):
        def scan_body(state, _):
            stress = elasticity_fn(state.F)
            state = step(params, state, stress, pre_particle_fn, post_grid_fn,
                         0.0, p2g_fn=_p2g_fn)
            state = state._replace(F=plasticity_fn(state.F))
            return state, None
        state, _ = jax.lax.scan(scan_body, state, None, length=steps_per_frame)
        return state

    return jit_frame


def simulate_frame(params, state, elasticity_fn, plasticity_fn,
                   pre_particle_fn, post_grid_fn, steps_per_frame, time, p2g_fn=None):
    """Run multiple substeps for one frame (unjitted, for per-stage profiling)."""
    for _ in range(steps_per_frame):
        stress = elasticity_fn(state.F)
        state = step(params, state, stress, pre_particle_fn, post_grid_fn, time, p2g_fn=p2g_fn)
        state = state._replace(F=plasticity_fn(state.F))
        time += params.dt
    return state, time


# ---------------------------------------------------------------------------
# Per-stage JIT path: one JIT per stage, host-side loop drives them.
# ---------------------------------------------------------------------------
#
# Trade-off vs build_jit_frame: extra Python dispatch + 2 sync points per
# substep, but enables per-stage timing and clean CUDA interop without
# baking the boundary into the traced graph.

def build_jit_stages(params, elasticity_fn, plasticity_fn,
                     pre_particle_fn, post_grid_fn, p2g_fn=None):
    """Build three individually-JIT'd stage functions.

    Returns:
        (jit_p2g_stage, jit_grid_stage, jit_g2p_stage)

    Signatures (time fixed at 0.0 internally to match build_jit_frame —
    boundary conditions don't see substep time in the JIT'd path):
        jit_p2g_stage(state)             -> (grid_mv, grid_m, intermediates)
        jit_grid_stage(grid_mv, grid_m)  -> grid_v
        jit_g2p_stage(state, grid_v, intermediates) -> MPMState
    """
    _p2g_fn = p2g_fn or p2g

    @jax.jit
    def jit_p2g_stage(state):
        x, v = pre_particle_fn(state.x, state.v, 0.0)
        stress = elasticity_fn(state.F)
        weight, dweight, dpos, index = compute_weights_and_indices(
            x, params.inv_dx, params.dx, params.num_grids)
        grid_mv, grid_m = _p2g_fn(
            v, state.C, stress, weight, dweight, dpos, index,
            params.dt, params.vol, params.p_mass, params.num_grids)
        inter = StepIntermediates(x_post_bc=x, F_pre_plast=state.F)
        return grid_mv, grid_m, inter

    @jax.jit
    def jit_grid_stage(grid_mv, grid_m):
        grid_mv_normalized = grid_update(
            grid_mv, grid_m, params.gravity, params.dt, params.damping)
        grid_v = post_grid_fn(grid_mv_normalized, grid_m, 0.0)
        return grid_v

    @jax.jit
    def jit_g2p_stage(state, grid_v, inter):
        # Recompute weights/indices instead of carrying them across the JIT
        # boundary - see StepIntermediates docstring.
        weight, dweight, dpos, index = compute_weights_and_indices(
            inter.x_post_bc, params.inv_dx, params.dx, params.num_grids)
        new_x, new_v, new_C, new_F = g2p(
            grid_v, weight, dweight, dpos, index,
            inter.F_pre_plast, inter.x_post_bc,
            params.dt, params.inv_dx, params.clip_bound)
        new_F = plasticity_fn(new_F)
        return MPMState(x=new_x, v=new_v, C=new_C, F=new_F)

    return jit_p2g_stage, jit_grid_stage, jit_g2p_stage
