"""CUDA P2G kernels, integrated via JAX FFI.

The .so files are built at install time by scikit-build-core + CMake (see
CMakeLists.txt) and shipped inside ``mpm_jax/cuda/_lib/``. Run
``pixi install -e gpu`` to (re)build; with ``editable.rebuild=true`` in
pyproject.toml, edits to the .cu sources also trigger a rebuild on import.

Override the CUDA architecture at build time with ``MPM_CUDA_ARCH=sm_86``
(default: ``native``).
"""

import ctypes
import logging
from pathlib import Path
from threading import Lock

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

_PACKAGE_DIR = Path(__file__).resolve().parent
_LIB_DIR = _PACKAGE_DIR / "_lib"
_REGISTERED: dict[str, bool] = {}
_REGISTER_LOCK = Lock()


def _shared_library_path(so_name: str) -> Path:
    return _LIB_DIR / so_name


def _register(name: str, so_name: str, symbol: str) -> bool:
    """Load .so from the package's _lib/ dir and register the FFI target."""
    with _REGISTER_LOCK:
        if name in _REGISTERED:
            return _REGISTERED[name]

        so_path = _shared_library_path(so_name)
        if not so_path.exists():
            logger.warning(
                "CUDA kernel %s not built. Run `pixi install -e gpu` "
                "in an environment where nvcc is on PATH.",
                so_name,
            )
            _REGISTERED[name] = False
            return False

        try:
            lib = ctypes.cdll.LoadLibrary(str(so_path))
            jax.ffi.register_ffi_target(
                name,
                jax.ffi.pycapsule(getattr(lib, symbol)),
                platform="CUDA",
                api_version=1,
            )
            _REGISTERED[name] = True
            logger.info("Registered CUDA kernel '%s' from %s", name, so_path)
            return True
        except Exception as e:
            logger.error("Failed to register CUDA kernel '%s': %s", name, e)
            _REGISTERED[name] = False
            return False


def _register_scatter():
    return _register("p2g_scatter_cuda", "libp2g_scatter.so", "P2GScatter")


def _register_warp():
    return _register("p2g_scatter_warp_cuda", "libp2g_scatter_warp.so", "P2GScatterWarp")


def _register_smem():
    return _register("p2g_scatter_smem_cuda", "libp2g_scatter_smem.so", "P2GScatterSmem")


def _register_fused():
    return _register("p2g_fused_cuda", "libp2g_fused.so", "P2GFused")


def _register_g2p_fused():
    return _register("g2p_fused_cuda", "libg2p_fused.so", "G2PFused")


def _register_inline():
    return _register("p2g_inline_cuda", "libp2g_inline.so", "P2GInline")


def cuda_p2g_scatter(mv, m, index, num_grids):
    """CUDA P2G scatter via JAX FFI.

    Drop-in replacement for solver.p2g_scatter().
    """
    G3 = num_grids ** 3
    index = index.astype(jnp.int32)

    grid_mv, grid_m = jax.ffi.ffi_call(
        "p2g_scatter_cuda",
        (
            jax.ShapeDtypeStruct((G3, 3), jnp.float32),
            jax.ShapeDtypeStruct((G3,), jnp.float32),
        ),
        vmap_method="broadcast_all",
    )(mv, m, index)

    return grid_mv, grid_m


def cuda_p2g_scatter_warp(mv, m, index, num_grids):
    """CUDA P2G scatter with warp-level reduction via JAX FFI."""
    G3 = num_grids ** 3
    index = index.astype(jnp.int32)

    grid_mv, grid_m = jax.ffi.ffi_call(
        "p2g_scatter_warp_cuda",
        (
            jax.ShapeDtypeStruct((G3, 3), jnp.float32),
            jax.ShapeDtypeStruct((G3,), jnp.float32),
        ),
        vmap_method="broadcast_all",
    )(mv, m, index)

    return grid_mv, grid_m


def cuda_p2g_scatter_smem(mv, m, index, num_grids):
    """CUDA P2G scatter with shared-memory tile staging via JAX FFI."""
    G = num_grids
    G3 = G ** 3

    index_i32 = index.astype(jnp.int32)

    # Sort particles by their home cell (center stencil node = offset 13)
    cell_id = index_i32[:, 13]
    order = jnp.argsort(cell_id)

    mv_sorted = mv[order]
    m_sorted = m[order]
    index_sorted = index_i32[order]

    cell_id_sorted = cell_id[order]
    cell_boundaries = jnp.arange(G3 + 1, dtype=jnp.int32)
    cell_start = jnp.searchsorted(cell_id_sorted, cell_boundaries).astype(jnp.int32)

    grid_mv, grid_m = jax.ffi.ffi_call(
        "p2g_scatter_smem_cuda",
        (
            jax.ShapeDtypeStruct((G3, 3), jnp.float32),
            jax.ShapeDtypeStruct((G3,), jnp.float32),
        ),
        vmap_method="broadcast_all",
    )(mv_sorted, m_sorted, index_sorted, cell_start)

    return grid_mv, grid_m


def cuda_p2g_fused(x, v, C, F, num_grids, dt, vol, p_mass, inv_dx, dx,
                   mu_0, lambda_0, theta_c=0.025, theta_s=0.0075, hardening=0.0):
    """Fused CUDA P2G via JAX FFI.

    Replaces the entire P2G pipeline (stress + weights + compute + scatter)
    with a single CUDA kernel. Also returns plasticity-corrected F.
    """
    N = x.shape[0]
    G = num_grids
    G3 = G ** 3

    C_flat = C.reshape(N, 9)
    F_flat = F.reshape(N, 9)

    grid_mv, grid_m, F_out_flat = jax.ffi.ffi_call(
        "p2g_fused_cuda",
        (
            jax.ShapeDtypeStruct((G3, 3), jnp.float32),
            jax.ShapeDtypeStruct((G3,), jnp.float32),
            jax.ShapeDtypeStruct((N, 9), jnp.float32),
        ),
        vmap_method="broadcast_all",
    )(
        x, v, C_flat, F_flat,
        N=np.int32(N),
        G=np.int32(G),
        dt=np.float32(dt),
        vol=np.float32(vol),
        p_mass=np.float32(p_mass),
        inv_dx=np.float32(inv_dx),
        dx=np.float32(dx),
        mu_0=np.float32(mu_0),
        lambda_0=np.float32(lambda_0),
        theta_c=np.float32(theta_c),
        theta_s=np.float32(theta_s),
        hardening_coeff=np.float32(hardening),
    )

    return grid_mv, grid_m, F_out_flat.reshape(N, 3, 3)


def is_available(kernel='scatter'):
    """Check if a prebuilt CUDA kernel can be loaded and registered."""
    if kernel == 'scatter':
        return _register_scatter()
    elif kernel == 'warp':
        return _register_warp()
    elif kernel == 'smem':
        return _register_smem()
    elif kernel == 'fused':
        return _register_fused()
    elif kernel == 'g2p_fused':
        return _register_g2p_fused()
    elif kernel == 'inline':
        return _register_inline()
    return False


def cuda_p2g_inline(x, v, C, stress, num_grids, dt, vol, p_mass, inv_dx, dx):
    """Inline-scatter CUDA P2G via JAX FFI (cuda_v1_inline).

    Takes per-particle state including precomputed stress (from JAX-side
    Jacobi SVD). One CUDA kernel launch, one thread per particle, with a
    register-resident 27-stencil loop. No (N, 27, *) tensor materialised.

    Drop-in replacement for solver.p2g_compute + solver.p2g_scatter when
    stress has already been computed by an upstream elasticity model.
    """
    N = x.shape[0]
    G = num_grids
    G3 = G ** 3
    C_flat = C.reshape(N, 9)
    stress_flat = stress.reshape(N, 9)

    grid_mv, grid_m = jax.ffi.ffi_call(
        "p2g_inline_cuda",
        (
            jax.ShapeDtypeStruct((G3, 3), jnp.float32),
            jax.ShapeDtypeStruct((G3,), jnp.float32),
        ),
        vmap_method="broadcast_all",
    )(
        x, v, C_flat, stress_flat,
        N=np.int32(N),
        G=np.int32(G),
        dt=np.float32(dt),
        vol=np.float32(vol),
        p_mass=np.float32(p_mass),
        inv_dx=np.float32(inv_dx),
        dx=np.float32(dx),
    )

    return grid_mv, grid_m


def build_jit_frame_inline(params, elasticity_fn, plasticity_fn,
                           pre_particle_fn, post_grid_fn, steps_per_frame):
    """Per-frame JIT'd function using the cuda_v1_inline P2G kernel.

    Mirrors ``solver.build_jit_frame`` but the substep body routes P2G
    through one CUDA kernel call that does weights + per-stencil momentum
    + atomic scatter inline per particle. The ``(N, 27, *)`` momentum
    tensor never materialises in HBM.

    Result is one ``@jax.jit`` + ``lax.scan`` over ``steps_per_frame`` —
    a single XLA program per frame, just like the pure-JAX ``build_jit_frame``.
    Stress and G2P stay in JAX; only the P2G scatter is replaced.
    """
    if not is_available('inline'):
        raise RuntimeError(
            "cuda_v1_inline P2G kernel not registered (missing .so?). "
            "Run `pixi install -e gpu` to build.")

    from mpm_jax.solver import (
        MPMState,
        compute_weights_and_indices,
        g2p,
        grid_update as grid_update_fn,
    )

    @jax.jit
    def jit_frame(state):
        def scan_body(state, _):
            x, v = pre_particle_fn(state.x, state.v, 0.0)
            stress = elasticity_fn(state.F)
            grid_mv, grid_m = cuda_p2g_inline(
                x, v, state.C, stress,
                params.num_grids, params.dt, params.vol, params.p_mass,
                params.inv_dx, params.dx,
            )
            grid_mv = grid_update_fn(
                grid_mv, grid_m, params.gravity, params.dt, params.damping)
            grid_mv = post_grid_fn(grid_mv, grid_m, 0.0)
            # G2P needs weights — compute them here (after P2G), so the
            # only (N, 27, *) tensors that exist are the ones G2P actually
            # consumes (gather + grad_v). No materialisation between
            # P2G compute and P2G scatter.
            weight, dweight, dpos, index = compute_weights_and_indices(
                x, params.inv_dx, params.dx, params.num_grids)
            new_x, new_v, new_C, new_F = g2p(
                grid_mv, weight, dweight, dpos, index,
                state.F, x, params.dt, params.inv_dx, params.clip_bound)
            new_F = plasticity_fn(new_F)
            return MPMState(x=new_x, v=new_v, C=new_C, F=new_F), None

        state, _ = jax.lax.scan(scan_body, state, None, length=steps_per_frame)
        return state

    return jit_frame


def cuda_g2p_fused(x, F, grid_v, num_grids, dt, inv_dx, dx, clip_bound):
    """Fused CUDA G2P via JAX FFI.

    Each thread computes its own B-spline weights in registers, gathers
    27 grid velocities, and produces (new_x, new_v, new_C, new_F) — no
    (N, 27, *) tensors materialised in HBM.
    """
    N = x.shape[0]
    G = num_grids
    F_flat = F.reshape(N, 9)

    new_x, new_v, new_C_flat, new_F_flat = jax.ffi.ffi_call(
        "g2p_fused_cuda",
        (
            jax.ShapeDtypeStruct((N, 3), jnp.float32),
            jax.ShapeDtypeStruct((N, 3), jnp.float32),
            jax.ShapeDtypeStruct((N, 9), jnp.float32),
            jax.ShapeDtypeStruct((N, 9), jnp.float32),
        ),
        vmap_method="broadcast_all",
    )(
        x, F_flat, grid_v,
        N=np.int32(N),
        G=np.int32(G),
        dt=np.float32(dt),
        inv_dx=np.float32(inv_dx),
        dx=np.float32(dx),
        clip_bound=np.float32(clip_bound),
    )

    return new_x, new_v, new_C_flat.reshape(N, 3, 3), new_F_flat.reshape(N, 3, 3)


def make_cuda_p2g(num_grids, kernel='scatter'):
    """Create a CUDA-accelerated p2g function matching the solver interface.

    Returns None if the prebuilt kernel is not available (.so missing or
    failed to register).
    """
    if kernel == 'scatter':
        if not is_available('scatter'):
            return None

        from mpm_jax.solver import p2g_compute

        def cuda_p2g_v1(v, C, stress, weight, dweight, dpos, index, dt, vol, p_mass, num_grids):
            mv, m = p2g_compute(v, C, stress, weight, dweight, dpos, dt, vol, p_mass)
            return cuda_p2g_scatter(mv, m, index, num_grids)

        return cuda_p2g_v1

    elif kernel == 'warp':
        if not is_available('warp'):
            return None

        from mpm_jax.solver import p2g_compute

        def cuda_p2g_v3(v, C, stress, weight, dweight, dpos, index, dt, vol, p_mass, num_grids):
            mv, m = p2g_compute(v, C, stress, weight, dweight, dpos, dt, vol, p_mass)
            return cuda_p2g_scatter_warp(mv, m, index, num_grids)

        return cuda_p2g_v3

    elif kernel == 'smem':
        if not is_available('smem'):
            return None

        from mpm_jax.solver import p2g_compute

        def cuda_p2g_v4(v, C, stress, weight, dweight, dpos, index, dt, vol, p_mass, num_grids):
            mv, m = p2g_compute(v, C, stress, weight, dweight, dpos, dt, vol, p_mass)
            return cuda_p2g_scatter_smem(mv, m, index, num_grids)

        return cuda_p2g_v4

    elif kernel == 'fused':
        if not is_available('fused'):
            return None
        logger.info("Fused CUDA P2G registered - use make_fused_stages()")
        return None  # handled specially in the driver

    return None


def make_fused_stages(params, elasticity_cfg, plasticity_cfg, pre_particle_fn, post_grid_fn):
    """Build per-stage JIT'd functions for the cuda_fused kernel.

    The fused kernel does SVD + plasticity + corotated stress + APIC + scatter
    in one launch. Differences vs the standard per-stage path:
      * No separate stress / weights / p2g_compute / p2g_scatter calls.
      * Plasticity is applied at the START of the step (kernel returns the
        corrected F). The G2P stage uses that corrected F and skips the
        separate plasticity_fn call.
      * Constitutive model is hard-coded to Corotated elasticity with snow-
        style singular-value clamping (Stomakhin 2013). Identity plasticity
        is realised by setting theta_c = theta_s = 1e9 (no clamp).

    Returns (jit_p2g_fused_stage, jit_grid_stage, jit_g2p_no_plast_stage)
    or raises if the kernel isn't available or the material config is
    unsupported.
    """
    if not is_available('fused'):
        raise RuntimeError(
            "cuda_fused P2G kernel is not registered (missing .so?). "
            "Run `pixi install -e gpu` to build.")
    if not is_available('g2p_fused'):
        raise RuntimeError(
            "cuda_fused G2P kernel is not registered (missing .so?). "
            "Run `pixi install -e gpu` to build.")

    if elasticity_cfg.name != "CorotatedElasticity":
        raise NotImplementedError(
            f"cuda_fused kernel only supports CorotatedElasticity, "
            f"got {elasticity_cfg.name}.")

    # Map plasticity config to (theta_c, theta_s, hardening_coeff). The kernel
    # implements snow-style clamping; with theta_c=theta_s=1e9 there's no clamp,
    # i.e. effectively identity plasticity.
    plast_name = plasticity_cfg.name
    if plast_name == "IdentityPlasticity":
        theta_c, theta_s, hardening = 1e9, 1e9, 0.0
    elif plast_name == "SnowPlasticity":
        theta_c = float(plasticity_cfg.get("theta_c", 0.025))
        theta_s = float(plasticity_cfg.get("theta_s", 0.0075))
        hardening = float(plasticity_cfg.get("hardening", 10.0))
    else:
        raise NotImplementedError(
            f"cuda_fused kernel only supports IdentityPlasticity or "
            f"SnowPlasticity, got {plast_name}.")

    E = float(elasticity_cfg.E)
    nu = float(elasticity_cfg.nu)
    mu_0 = E / (2.0 * (1.0 + nu))
    lambda_0 = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    # Late imports to avoid circular dep at module load.
    from mpm_jax.solver import (
        MPMState, StepIntermediates,
        grid_update as grid_update_fn,
    )

    @jax.jit
    def jit_p2g_fused_stage(state):
        x, v = pre_particle_fn(state.x, state.v, 0.0)
        grid_mv, grid_m, F_corrected = cuda_p2g_fused(
            x, v, state.C, state.F,
            params.num_grids, params.dt, params.vol, params.p_mass,
            params.inv_dx, params.dx,
            mu_0, lambda_0, theta_c, theta_s, hardening,
        )
        # Slim intermediates - G2P recomputes weights from x_post_bc.
        inter = StepIntermediates(x_post_bc=x, F_pre_plast=F_corrected)
        return grid_mv, grid_m, inter

    @jax.jit
    def jit_grid_stage(grid_mv, grid_m):
        grid_mv_normalized = grid_update_fn(
            grid_mv, grid_m, params.gravity, params.dt, params.damping)
        grid_v = post_grid_fn(grid_mv_normalized, grid_m, 0.0)
        return grid_v

    @jax.jit
    def jit_g2p_no_plast_stage(state, grid_v, inter):
        # Fully fused: each thread does its own B-spline math + gather
        # in registers. No (N, 27, *) tensors anywhere on this stage.
        new_x, new_v, new_C, new_F = cuda_g2p_fused(
            inter.x_post_bc, inter.F_pre_plast, grid_v,
            params.num_grids, params.dt, params.inv_dx, params.dx,
            params.clip_bound,
        )
        return MPMState(x=new_x, v=new_v, C=new_C, F=new_F)

    return jit_p2g_fused_stage, jit_grid_stage, jit_g2p_no_plast_stage
