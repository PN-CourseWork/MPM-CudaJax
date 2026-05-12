"""CUDA P2G kernels, integrated via JAX FFI.

The .so files are built at install time by scikit-build-core + CMake (see
CMakeLists.txt) and shipped inside ``mpm_jax/cuda/_lib/``. Run
``uv sync --extra jax-cuda`` to (re)build; with ``editable.rebuild=true`` in
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
                "CUDA kernel %s not built. Run `uv sync --extra jax-cuda` "
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
    )(x, v, C_flat, F_flat)

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
    return False


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
        logger.info("Fused CUDA P2G registered - use cuda_p2g_fused() directly")
        return None  # handled specially in the driver

    return None
