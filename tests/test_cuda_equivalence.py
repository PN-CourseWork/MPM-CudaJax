"""Each CUDA P2G variant must produce results numerically equivalent to the
JAX baseline.

Strategy: build the per-stage stage functions for both JAX and the CUDA
variant, run an identical short simulation from the same seeded initial
state, then assert the final particle state matches.

Tolerances reflect the source of expected drift:

  v1, v3, v4 — only the scatter is different from JAX. ``atomicAdd`` is
      associative-ish but not order-deterministic on floats, so per-substep
      drift is tiny and bounded; we use atol=1e-4 on positions / 5e-3 on
      velocities after 20 substeps.
  v2 — the entire P2G stage (SVD + stress + APIC + scatter) is replaced by
      a single fused CUDA kernel that does its own Jacobi SVD instead of
      cuSOLVER. SVD outputs differ at f32 noise, which feeds back into
      stress and accumulates through the timestep. We use atol=1e-3 on
      positions / 1e-2 on velocities.

All tests are skipped on systems without a CUDA GPU or when the relevant
.so is missing (CPU-only installs).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from mpm_jax.solver import MPMState, make_params, build_jit_stages
from mpm_jax.constitutive import get_constitutive
from mpm_jax.boundary import build_boundary_fns


def _has_cuda() -> bool:
    try:
        return jax.default_backend() == "gpu"
    except Exception:
        return False


def _kernel_available(kind: str) -> bool:
    """True if the prebuilt .so for the requested kernel can be loaded.

    ``is_available`` is idempotent (caches the registration), so calling it
    at test-collection time is cheap.
    """
    if not _has_cuda():
        return False
    from mpm_jax.cuda.p2g_cuda import is_available
    return is_available(kind)


def _build_setup(n: int = 200, num_grids: int = 16):
    """Common deterministic simulation setup shared by every test.

    Returns a tuple (params, pre_fn, post_fn, initial_state, elasticity_cfg,
    plasticity_cfg, elasticity_fn, plasticity_fn).
    """
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
    e_cfg = OmegaConf.create({"name": "CorotatedElasticity", "E": 2e6, "nu": 0.4})
    p_cfg = OmegaConf.create({"name": "IdentityPlasticity"})
    elasticity_fn = get_constitutive(e_cfg)
    plasticity_fn = get_constitutive(p_cfg)

    initial_state = MPMState(
        x=x0,
        v=jnp.broadcast_to(jnp.array([0.0, 0.0, -0.5]), (n, 3)).copy(),
        C=jnp.zeros((n, 3, 3)),
        F=jnp.tile(jnp.eye(3), (n, 1, 1)),
    )
    return (params, pre_fn, post_fn, initial_state, e_cfg, p_cfg,
            elasticity_fn, plasticity_fn)


def _run_stages(stages, state, n_substeps: int):
    """Drive ``n_substeps`` of the simulation through a triple of stage fns."""
    jit_p2g, jit_grid, jit_g2p = stages
    for _ in range(n_substeps):
        grid_mv, grid_m, inter = jit_p2g(state)
        grid_v = jit_grid(grid_mv, grid_m)
        state = jit_g2p(state, grid_v, inter)
    jax.block_until_ready(state.x)
    return state


def _assert_states_close(s_jax, s_cuda, *, atol_x, atol_v, atol_F):
    np.testing.assert_allclose(
        np.asarray(s_jax.x), np.asarray(s_cuda.x), atol=atol_x, rtol=1e-3,
        err_msg="positions diverge",
    )
    np.testing.assert_allclose(
        np.asarray(s_jax.v), np.asarray(s_cuda.v), atol=atol_v, rtol=1e-3,
        err_msg="velocities diverge",
    )
    np.testing.assert_allclose(
        np.asarray(s_jax.F), np.asarray(s_cuda.F), atol=atol_F, rtol=1e-3,
        err_msg="deformation gradients diverge",
    )


# Common scatter-only tolerance for v1/v3/v4: only the atomicAdd order
# differs from XLA's scatter.
_SCATTER_ATOL_X = 1e-4
_SCATTER_ATOL_V = 5e-3
_SCATTER_ATOL_F = 1e-4
_N_SUBSTEPS = 20


@pytest.mark.skipif(not _kernel_available("scatter"),
                    reason="cuda_v1 (naive scatter) .so not built or no GPU")
def test_cuda_v1_matches_jax():
    from mpm_jax.cuda.p2g_cuda import make_cuda_p2g

    params, pre, post, state0, _, _, ef, pf = _build_setup()
    s_jax = _run_stages(build_jit_stages(params, ef, pf, pre, post), state0, _N_SUBSTEPS)

    cuda_p2g = make_cuda_p2g(num_grids=16, kernel="scatter")
    assert cuda_p2g is not None, "cuda_v1 kernel was reported available but factory returned None"
    s_cuda = _run_stages(
        build_jit_stages(params, ef, pf, pre, post, p2g_fn=cuda_p2g),
        state0, _N_SUBSTEPS,
    )

    _assert_states_close(s_jax, s_cuda,
                         atol_x=_SCATTER_ATOL_X,
                         atol_v=_SCATTER_ATOL_V,
                         atol_F=_SCATTER_ATOL_F)


@pytest.mark.skipif(not _kernel_available("warp"),
                    reason="cuda_v3 (warp-reduced scatter) .so not built or no GPU")
def test_cuda_v3_matches_jax():
    from mpm_jax.cuda.p2g_cuda import make_cuda_p2g

    params, pre, post, state0, _, _, ef, pf = _build_setup()
    s_jax = _run_stages(build_jit_stages(params, ef, pf, pre, post), state0, _N_SUBSTEPS)

    cuda_p2g = make_cuda_p2g(num_grids=16, kernel="warp")
    assert cuda_p2g is not None
    s_cuda = _run_stages(
        build_jit_stages(params, ef, pf, pre, post, p2g_fn=cuda_p2g),
        state0, _N_SUBSTEPS,
    )

    _assert_states_close(s_jax, s_cuda,
                         atol_x=_SCATTER_ATOL_X,
                         atol_v=_SCATTER_ATOL_V,
                         atol_F=_SCATTER_ATOL_F)


@pytest.mark.skipif(not _kernel_available("smem"),
                    reason="cuda_v4 (smem-staged scatter) .so not built or no GPU")
def test_cuda_v4_matches_jax():
    from mpm_jax.cuda.p2g_cuda import make_cuda_p2g

    params, pre, post, state0, _, _, ef, pf = _build_setup()
    s_jax = _run_stages(build_jit_stages(params, ef, pf, pre, post), state0, _N_SUBSTEPS)

    cuda_p2g = make_cuda_p2g(num_grids=16, kernel="smem")
    assert cuda_p2g is not None
    s_cuda = _run_stages(
        build_jit_stages(params, ef, pf, pre, post, p2g_fn=cuda_p2g),
        state0, _N_SUBSTEPS,
    )

    _assert_states_close(s_jax, s_cuda,
                         atol_x=_SCATTER_ATOL_X,
                         atol_v=_SCATTER_ATOL_V,
                         atol_F=_SCATTER_ATOL_F)


@pytest.mark.skipif(not _kernel_available("fused"),
                    reason="cuda_v2 (fused) .so not built or no GPU")
def test_cuda_v2_matches_jax():
    """Looser tolerances: the fused kernel does its own Jacobi SVD instead of
    cuSOLVER, so f32-noise drift in singular values feeds back through the
    stress and accumulates across substeps."""
    from mpm_jax.cuda.p2g_cuda import make_fused_stages

    params, pre, post, state0, e_cfg, p_cfg, ef, pf = _build_setup()
    s_jax = _run_stages(build_jit_stages(params, ef, pf, pre, post), state0, _N_SUBSTEPS)

    v2_stages = make_fused_stages(params, e_cfg, p_cfg, pre, post)
    s_v2 = _run_stages(v2_stages, state0, _N_SUBSTEPS)

    _assert_states_close(s_jax, s_v2,
                         atol_x=1e-3,
                         atol_v=1e-2,
                         atol_F=1e-3)
