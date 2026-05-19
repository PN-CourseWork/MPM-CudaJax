"""jax_v1_5 (lax.scan over 27 stencils) must match the jax baseline.

Both paths are pure JAX with no host-side intermediate state, both use
``CorotatedElasticityJacobi`` for stress (the cuSOLVER-free Jacobi SVD),
and both ultimately lower their scatter to ``at[].add``. The only
algorithmic difference is *when* the 27 contributions get summed:

  jax       — vmap produces (N, 27, 3) momentum, then a single scatter
              with (27*N,) flat indices.
  jax_v1_5  — lax.scan over the 27 offsets, each iteration computes
              (N, 3) contributions and runs its own scatter.

Per-substep float32 atomicAdd drift is bounded but nonzero (atomicAdd is
not order-deterministic on GPU). After 10 substeps with N=2000 we expect
single-precision agreement at rtol=1e-3, atol=1e-4.
"""

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from mpm_jax.solver import MPMState, build_jit_stages, make_params
from mpm_jax.constitutive import get_constitutive
from mpm_jax.boundary import build_boundary_fns
from mpm_jax.p2g_scan import build_jit_stages_scan


def _build_setup(n: int = 2000, num_grids: int = 16):
    """Deterministic mini-simulation setup, mirrors test_cuda_equivalence."""
    rng = np.random.RandomState(0)
    x0 = jnp.array(rng.rand(n, 3).astype(np.float32) * 0.4 + 0.3)
    params = make_params(n_particles=n, num_grids=num_grids, dt=3e-4)

    g = jnp.arange(num_grids, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(g, g, g, indexing="ij")
    grid_x = jnp.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    bcs = [
        {
            "type": "surface_collider",
            "point": [1.0, 1.0, 0.02],
            "normal": [0.0, 0.0, 1.0],
            "surface": "sticky",
            "friction": 0.0,
            "start_time": 0.0,
            "end_time": 1e3,
        }
    ]
    pre_fn, post_fn = build_boundary_fns(bcs, grid_x, params.dx, x0, params.dt)

    # jelly_jacobi material — the pure-JAX Jacobi SVD path.
    e_cfg = OmegaConf.create({"name": "CorotatedElasticityJacobi", "E": 2e6, "nu": 0.4})
    p_cfg = OmegaConf.create({"name": "IdentityPlasticity"})
    elasticity_fn = get_constitutive(e_cfg)
    plasticity_fn = get_constitutive(p_cfg)

    initial_state = MPMState(
        x=x0,
        v=jnp.broadcast_to(jnp.array([0.0, 0.0, -0.5]), (n, 3)).copy(),
        C=jnp.zeros((n, 3, 3)),
        F=jnp.tile(jnp.eye(3), (n, 1, 1)),
    )
    return params, pre_fn, post_fn, initial_state, elasticity_fn, plasticity_fn


def _run_stages(stages, state, n_substeps: int):
    jit_p2g, jit_grid, jit_g2p = stages
    for _ in range(n_substeps):
        grid_mv, grid_m, inter = jit_p2g(state)
        grid_v = jit_grid(grid_mv, grid_m)
        state = jit_g2p(state, grid_v, inter)
    jax.block_until_ready(state.x)
    return state


def test_jax_v1_5_matches_jax_baseline():
    params, pre, post, state0, ef, pf = _build_setup(n=2000, num_grids=16)

    s_jax = _run_stages(
        build_jit_stages(params, ef, pf, pre, post),
        state0,
        n_substeps=10,
    )
    s_scan = _run_stages(
        build_jit_stages_scan(params, ef, pf, pre, post),
        state0,
        n_substeps=10,
    )

    rtol, atol = 1e-3, 1e-4
    np.testing.assert_allclose(
        np.asarray(s_jax.x), np.asarray(s_scan.x), rtol=rtol, atol=atol,
        err_msg="positions diverge",
    )
    np.testing.assert_allclose(
        np.asarray(s_jax.v), np.asarray(s_scan.v), rtol=rtol, atol=atol,
        err_msg="velocities diverge",
    )
    np.testing.assert_allclose(
        np.asarray(s_jax.C), np.asarray(s_scan.C), rtol=rtol, atol=atol,
        err_msg="C matrices diverge",
    )
    np.testing.assert_allclose(
        np.asarray(s_jax.F), np.asarray(s_scan.F), rtol=rtol, atol=atol,
        err_msg="deformation gradients diverge",
    )
