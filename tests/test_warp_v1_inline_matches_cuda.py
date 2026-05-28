"""warp_v1_inline should match cuda_v1_inline for one short frame."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from mpm_jax.boundary import build_boundary_fns
from mpm_jax.constitutive import get_constitutive
from mpm_jax.solver import MPMState, make_params


def _has_cuda() -> bool:
    try:
        return jax.default_backend() == "gpu"
    except Exception:
        return False


def _kernel_available(kind: str) -> bool:
    if not _has_cuda():
        return False
    from mpm_jax.cuda.p2g_cuda import is_available
    return is_available(kind)


@pytest.mark.skipif(
    not (_kernel_available("inline") and _kernel_available("g2p_fused")),
    reason="cuda_v1_inline / g2p_fused .so not built or no GPU",
)
def test_warp_v1_inline_matches_cuda_v1_inline():
    from mpm_jax.cuda.p2g_cuda import build_jit_frame_inline
    from mpm_jax.warp_p2g import build_jit_frame_warp_inline

    n = 200
    num_grids = 16
    rng = np.random.RandomState(0)
    x0 = jnp.array(rng.rand(n, 3).astype(np.float32) * 0.4 + 0.3)
    params = make_params(n_particles=n, num_grids=num_grids, dt=3e-4)

    g = jnp.arange(num_grids, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(g, g, g, indexing="ij")
    grid_x = jnp.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    pre_fn, post_fn = build_boundary_fns([], grid_x, params.dx, x0, params.dt)

    e_cfg = OmegaConf.create({"name": "CorotatedElasticityJacobi", "E": 2e6, "nu": 0.4})
    p_cfg = OmegaConf.create({"name": "IdentityPlasticity"})
    elasticity_fn = get_constitutive(e_cfg)
    plasticity_fn = get_constitutive(p_cfg)

    state0 = MPMState(
        x=x0,
        v=jnp.broadcast_to(jnp.array([0.0, 0.0, -0.5]), (n, 3)).copy(),
        C=jnp.zeros((n, 3, 3)),
        F=jnp.tile(jnp.eye(3), (n, 1, 1)),
    )

    jit_cuda = build_jit_frame_inline(
        params, elasticity_fn, plasticity_fn, pre_fn, post_fn, 1)
    jit_warp = build_jit_frame_warp_inline(
        params, elasticity_fn, plasticity_fn, pre_fn, post_fn, 1)

    s_cuda = jit_cuda(state0)
    s_warp = jit_warp(state0)
    jax.block_until_ready(s_cuda.x)
    jax.block_until_ready(s_warp.x)

    np.testing.assert_allclose(np.asarray(s_cuda.x), np.asarray(s_warp.x),
                               atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(s_cuda.v), np.asarray(s_warp.v),
                               atol=1e-5, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(s_cuda.F), np.asarray(s_warp.F),
                               atol=1e-6, rtol=1e-6)
