"""Pure-Warp graph prototype should match the JAX timestep math."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import warp as wp
from omegaconf import OmegaConf

from mpm_jax.boundary import build_boundary_fns
from mpm_jax.constitutive import get_constitutive
from mpm_jax.solver import MPMState, build_jit_frame, make_params
from mpm_jax.warp_bonus import WarpBonusSimulator


def _has_cuda() -> bool:
    try:
        return jax.default_backend() == "gpu"
    except Exception:
        return False


@pytest.mark.skipif(not _has_cuda(), reason="requires a CUDA-backed JAX/Warp runtime")
@pytest.mark.parametrize("indexed_sort", [False, True])
def test_warp_bonus_single_substep_matches_jax(indexed_sort):
    n = 1024
    num_grids = 16
    rng = np.random.RandomState(0)
    x_np = rng.rand(n, 3).astype(np.float32) * 0.4 + 0.3

    cfg = OmegaConf.create({
        "sim": {
            "n_particles": n,
            "num_grids": num_grids,
            "steps_per_frame": 1,
            "dt": 3e-4,
            "gravity": [0.0, 0.0, -9.8],
            "rho": 1000.0,
            "clip_bound": 0.5,
            "damping": 1.0,
            "center": [0.5, 0.5, 0.5],
            "size": [1.0, 1.0, 1.0],
            "initial_velocity": [0.0, 0.0, -0.5],
        },
        "material": {
            "elasticity": {"name": "CorotatedElasticityJacobi", "E": 2e6, "nu": 0.4},
            "plasticity": {"name": "IdentityPlasticity"},
        },
    })

    params = make_params(
        n_particles=n,
        num_grids=num_grids,
        dt=cfg.sim.dt,
        gravity=list(cfg.sim.gravity),
        rho=cfg.sim.rho,
        clip_bound=cfg.sim.clip_bound,
        damping=cfg.sim.damping,
        center=list(cfg.sim.center),
        size=list(cfg.sim.size),
    )

    x0 = jnp.array(x_np)
    g = jnp.arange(num_grids, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(g, g, g, indexing="ij")
    grid_x = jnp.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    pre_fn, post_fn = build_boundary_fns([], grid_x, params.dx, x0, params.dt, params.p_mass)

    elasticity_fn = get_constitutive(
        OmegaConf.create({"name": "CorotatedElasticityJacobi", "E": 2e6, "nu": 0.4})
    )
    plasticity_fn = get_constitutive(OmegaConf.create({"name": "IdentityPlasticity"}))
    state0 = MPMState(
        x=x0,
        v=jnp.broadcast_to(jnp.array([0.0, 0.0, -0.5], dtype=jnp.float32), (n, 3)).copy(),
        C=jnp.zeros((n, 3, 3)),
        F=jnp.tile(jnp.eye(3), (n, 1, 1)),
    )

    jit_frame = build_jit_frame(params, elasticity_fn, plasticity_fn, pre_fn, post_fn, 1)
    jax_state = jit_frame(state0)
    jax.block_until_ready(jax_state.x)

    warp_sim = WarpBonusSimulator(x_np, cfg, indexed_sort=indexed_sort)
    warp_sim._substep()
    wp.synchronize_device(warp_sim.device)

    x_jax = np.asarray(jax_state.x)
    v_jax = np.asarray(jax_state.v)
    F_jax = np.asarray(jax_state.F)
    x_warp = warp_sim.x.numpy()
    v_warp = warp_sim.v.numpy()
    F_warp = warp_sim.F.numpy()

    order_jax = np.lexsort((x_jax[:, 2], x_jax[:, 1], x_jax[:, 0]))
    order_warp = np.lexsort((x_warp[:, 2], x_warp[:, 1], x_warp[:, 0]))

    np.testing.assert_allclose(x_jax[order_jax], x_warp[order_warp], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(v_jax[order_jax], v_warp[order_warp], atol=2e-5, rtol=1e-5)
    np.testing.assert_allclose(F_jax[order_jax], F_warp[order_warp], atol=1e-6, rtol=1e-6)
