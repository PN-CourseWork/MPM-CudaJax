"""cuda_v2_inline must match cuda_v1_inline up to atomic-order f32 drift.

Both kernels are bit-identical in their per-particle math; only the scatter
strategy differs (v2_inline adds a warp-shuffle reduction before each
atomicAdd). The same drift sources as the rest of the CUDA equivalence
suite apply (non-deterministic atomicAdd ordering), so we use the same
tolerances as the v1-vs-JAX comparison.

Skipped when the .so isn't built or there's no GPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from mpm_jax.solver import MPMState, make_params
from mpm_jax.constitutive import get_constitutive
from mpm_jax.boundary import build_boundary_fns


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
    not (_kernel_available("inline") and _kernel_available("v2_inline")
         and _kernel_available("g2p_fused")),
    reason="cuda_v1_inline / cuda_v2_inline / g2p_fused .so not built or no GPU",
)
def test_cuda_v2_inline_matches_v1_inline():
    """Run a short sim under both inline kernels and compare final state."""
    from mpm_jax.cuda.p2g_cuda import (
        build_jit_frame_inline,
        build_jit_frame_v2_inline,
    )

    n = 2000
    num_grids = 16
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

    # Use the Jacobi-SVD elasticity model (same as material=jelly_jacobi)
    # so stress is computed on the JAX side without a cuSOLVER dependence.
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

    steps_per_frame = 5
    num_frames = 3

    jit_v1 = build_jit_frame_inline(
        params, elasticity_fn, plasticity_fn, pre_fn, post_fn, steps_per_frame)
    jit_v2 = build_jit_frame_v2_inline(
        params, elasticity_fn, plasticity_fn, pre_fn, post_fn, steps_per_frame)

    s1 = state0
    s2 = state0
    for _ in range(num_frames):
        s1 = jit_v1(s1)
        s2 = jit_v2(s2)
    jax.block_until_ready(s1.x)
    jax.block_until_ready(s2.x)

    # Same scatter-only tolerance band as the rest of the CUDA-equivalence
    # suite: positions ~1e-4, velocities ~5e-3, F ~1e-4.
    np.testing.assert_allclose(np.asarray(s1.x), np.asarray(s2.x),
                               atol=1e-4, rtol=1e-3)
    np.testing.assert_allclose(np.asarray(s1.v), np.asarray(s2.v),
                               atol=5e-3, rtol=1e-3)
    np.testing.assert_allclose(np.asarray(s1.F), np.asarray(s2.F),
                               atol=1e-4, rtol=1e-3)
