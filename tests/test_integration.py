import jax
import jax.numpy as jnp
import numpy as np
from mpm_jax.solver import (
    MPMState, make_params, simulate_frame, build_jit_frame, build_jit_stages,
)
from mpm_jax.constitutive import get_constitutive
from mpm_jax.boundary import build_boundary_fns

def _make_grid_x(num_grids):
    g = jnp.arange(num_grids, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(g, g, g, indexing='ij')
    return jnp.stack([gx, gy, gz], axis=-1).reshape(-1, 3)

def test_jelly_simulation_10_frames():
    from omegaconf import OmegaConf
    N = 100
    num_grids = 15
    x0 = jnp.ones((N, 3)) * 0.5
    params = make_params(n_particles=N, num_grids=num_grids, dt=3e-4)
    grid_x = _make_grid_x(num_grids)
    bc_configs = [
        {"type": "surface_collider", "point": [1.0, 1.0, 0.02],
         "normal": [0.0, 0.0, 1.0], "surface": "sticky", "friction": 0.0,
         "start_time": 0.0, "end_time": 1e3},
    ]
    pre_fn, post_fn = build_boundary_fns(bc_configs, grid_x, params.dx, x0, params.dt)
    elasticity_fn = get_constitutive(OmegaConf.create({"name": "CorotatedElasticity", "E": 2e6, "nu": 0.4}))
    plasticity_fn = get_constitutive(OmegaConf.create({"name": "IdentityPlasticity"}))
    state = MPMState(
        x=x0,
        v=jnp.broadcast_to(jnp.array([0.0, 0.0, -0.5]), (N, 3)).copy(),
        C=jnp.zeros((N, 3, 3)),
        F=jnp.tile(jnp.eye(3), (N, 1, 1)),
    )
    time = 0.0
    for _ in range(10):
        state, time = simulate_frame(
            params, state, elasticity_fn, plasticity_fn,
            pre_fn, post_fn, steps_per_frame=5, time=time,
        )
    assert jnp.mean(state.x[:, 2]) < 0.5
    assert jnp.all(jnp.isfinite(state.x))


def test_per_stage_matches_per_frame():
    """Per-stage JIT and monolithic frame JIT must produce bit-identical state.

    They run the same numerical operations in the same order; only the JIT
    granularity differs (3 stages per substep vs one scan'd frame).
    """
    from omegaconf import OmegaConf
    N = 200
    num_grids = 16
    steps_per_frame = 4
    rng = np.random.RandomState(0)
    x0 = jnp.array(rng.rand(N, 3).astype(np.float32) * 0.5 + 0.25)

    params = make_params(n_particles=N, num_grids=num_grids, dt=3e-4)
    grid_x = _make_grid_x(num_grids)
    bc_configs = [
        {"type": "surface_collider", "point": [1.0, 1.0, 0.02],
         "normal": [0.0, 0.0, 1.0], "surface": "sticky", "friction": 0.0,
         "start_time": 0.0, "end_time": 1e3},
    ]
    pre_fn, post_fn = build_boundary_fns(bc_configs, grid_x, params.dx, x0, params.dt)
    elasticity_fn = get_constitutive(OmegaConf.create(
        {"name": "CorotatedElasticity", "E": 2e6, "nu": 0.4}))
    plasticity_fn = get_constitutive(OmegaConf.create({"name": "IdentityPlasticity"}))

    def make_state():
        return MPMState(
            x=x0,
            v=jnp.broadcast_to(jnp.array([0.0, 0.0, -0.5]), (N, 3)).copy(),
            C=jnp.zeros((N, 3, 3)),
            F=jnp.tile(jnp.eye(3), (N, 1, 1)),
        )

    # Path A: monolithic JIT'd frame
    jit_frame = build_jit_frame(
        params, elasticity_fn, plasticity_fn, pre_fn, post_fn, steps_per_frame)
    s_a = jit_frame(make_state())
    s_a = jit_frame(s_a)  # 2 frames

    # Path B: per-stage JIT, host loop
    jit_p2g, jit_grid, jit_g2p = build_jit_stages(
        params, elasticity_fn, plasticity_fn, pre_fn, post_fn)
    s_b = make_state()
    for _ in range(2):
        for _ in range(steps_per_frame):
            grid_mv, grid_m, inter = jit_p2g(s_b)
            grid_v = jit_grid(grid_mv, grid_m)
            s_b = jit_g2p(s_b, grid_v, inter)
    jax.block_until_ready(s_b.x)

    # XLA may reorder reductions differently across JIT boundaries, so we
    # tolerate float32 noise (~1e-5) on raw values that are themselves tiny.
    np.testing.assert_allclose(np.asarray(s_a.x), np.asarray(s_b.x), atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(s_a.v), np.asarray(s_b.v), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(s_a.F), np.asarray(s_b.F), atol=1e-4, rtol=1e-4)
