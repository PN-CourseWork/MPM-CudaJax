"""CUDA P2G kernels, integrated via JAX FFI.

Kernels are compiled automatically from .cu source on first use (requires
nvcc on PATH). The compiled .so is cached in build/ffi by default, or in
MPM_FFI_CACHE when that environment variable is set.
"""

import os
import subprocess
import ctypes
import logging
import shutil
from pathlib import Path
from threading import Lock

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parents[1]
_KERNEL_DIR = _PACKAGE_DIR / "kernels"
_FFI_CACHE_ENV = "MPM_FFI_CACHE"
_REGISTERED = {}
_REGISTER_LOCK = Lock()


def _ffi_cache_dir() -> Path:
    cache_root = os.environ.get(_FFI_CACHE_ENV)
    if cache_root:
        path = Path(cache_root).expanduser()
    else:
        path = _PROJECT_ROOT / "build" / "ffi"
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _kernel_source_path(cu_name) -> Path:
    return _KERNEL_DIR / cu_name


def _shared_library_path(so_name) -> Path:
    return _ffi_cache_dir() / so_name


def _compile_kernel(cu_name, so_name):
    """Compile a .cu file to .so using nvcc. Returns True on success."""
    cu_path = _kernel_source_path(cu_name)
    so_path = _shared_library_path(so_name)

    # Skip if .so exists and is newer than .cu
    if so_path.exists():
        if so_path.stat().st_mtime > cu_path.stat().st_mtime:
            return True

    if not cu_path.exists():
        logger.error("CUDA source not found: %s", cu_path)
        return False

    # Prefer an nvcc matching the driver's CUDA version to avoid
    # "driver version insufficient" errors. Fall back to PATH nvcc.
    nvcc = os.environ.get("NVCC", shutil.which("nvcc"))
    if not nvcc:
        logger.warning("nvcc not found on PATH — cannot compile CUDA kernels. "
                       "Set NVCC env var to point to a compatible nvcc.")
        return False

    # Get FFI include dir
    try:
        ffi_inc = jax.ffi.include_dir()
    except Exception:
        logger.warning("Cannot determine JAX FFI include dir")
        return False

    # Find the GCC libstdc++ and set RPATH so the .so can find it at runtime
    gcc_lib_dir = None
    try:
        gcc_lib = subprocess.run(
            ["gcc", "-print-file-name=libstdc++.so"],
            capture_output=True, text=True
        )
        if gcc_lib.returncode == 0 and "/" in gcc_lib.stdout.strip():
            gcc_lib_dir = os.path.dirname(os.path.realpath(gcc_lib.stdout.strip()))
            logger.info("Using GCC libstdc++ from %s", gcc_lib_dir)
    except FileNotFoundError:
        pass

    cmd = [
        nvcc,
        "-arch=sm_90",
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "-Xcompiler", "-fPIC",
        "-shared",
        f"-I{ffi_inc}",
        "-diag-suppress=940,2473",
        "-Xcompiler", "-Wno-return-type",
    ]
    if gcc_lib_dir:
        cmd.extend(["-Xlinker", "-rpath", "-Xlinker", gcc_lib_dir])
    cmd.extend(["-o", str(so_path), str(cu_path)])

    logger.info("Compiling %s -> %s", cu_name, so_name)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("nvcc failed:\n%s\n%s", result.stdout, result.stderr)
        return False

    logger.info("Compiled %s successfully", so_name)
    return True


def _register(name, cu_name, so_name, symbol):
    """Compile if needed, load .so, and register FFI target."""
    with _REGISTER_LOCK:
        if name in _REGISTERED:
            return _REGISTERED[name]

        # Auto-compile
        if not _compile_kernel(cu_name, so_name):
            _REGISTERED[name] = False
            return False

        so_path = _shared_library_path(so_name)
        try:
            lib = ctypes.cdll.LoadLibrary(str(so_path))
            jax.ffi.register_ffi_target(
                name,
                jax.ffi.pycapsule(getattr(lib, symbol)),
                platform="CUDA",
                api_version=1,
            )
            _REGISTERED[name] = True
            logger.info("Registered CUDA kernel '%s'", name)
            return True
        except Exception as e:
            logger.error("Failed to register CUDA kernel '%s': %s", name, e)
            _REGISTERED[name] = False
            return False


def _register_scatter():
    return _register("p2g_scatter_cuda", "p2g_scatter.cu", "libp2g_scatter.so", "P2GScatter")


def _register_warp():
    return _register("p2g_scatter_warp_cuda", "p2g_scatter_warp.cu", "libp2g_scatter_warp.so", "P2GScatterWarp")


def _register_smem():
    return _register("p2g_scatter_smem_cuda", "p2g_scatter_smem.cu", "libp2g_scatter_smem.so", "P2GScatterSmem")


def _register_fused():
    return _register("p2g_fused_cuda", "p2g_fused.cu", "libp2g_fused.so", "P2GFused")


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
    """CUDA P2G scatter with warp-level reduction via JAX FFI.

    Same interface as cuda_p2g_scatter but uses __match_any_sync +
    __shfl_down_sync to reduce atomics within each warp.
    """
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
    """CUDA P2G scatter with shared memory staging via JAX FFI.

    Sorts particles by cell, then uses per-cell shared memory tiles
    (4x4x4) to accumulate contributions before flushing to global memory.
    This reduces global atomics from 4*27*N to ~4*64*num_occupied_cells.
    """
    G = num_grids
    G3 = G ** 3

    index_i32 = index.astype(jnp.int32)

    # Sort particles by their home cell (center stencil node = offset 13)
    cell_id = index_i32[:, 13]  # (N,) — particle's cell
    order = jnp.argsort(cell_id)

    mv_sorted = mv[order]        # (N, 27, 3)
    m_sorted = m[order]           # (N, 27)
    index_sorted = index_i32[order]  # (N, 27)

    # Build cell_start: CSR-style, cell_start[c] = first particle in cell c
    # Using searchsorted on the sorted cell IDs
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
    """Check if a CUDA kernel can be compiled and registered."""
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

    Compiles the kernel automatically if needed.
    Returns None if nvcc is not available.
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
        logger.info("Fused CUDA P2G registered — use cuda_p2g_fused() directly")
        return None  # handled specially in the driver

    return None
