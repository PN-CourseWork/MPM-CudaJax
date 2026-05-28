"""CUDA P2G kernels, integrated via JAX FFI.

The .so files are built at install time by scikit-build-core + CMake (see
CMakeLists.txt) and shipped inside ``mpm_jax/cuda/_lib/``. In editable Pixi
environments, ``editable.rebuild=true`` lets scikit-build-core incrementally
rebuild changed CUDA sources when the packaged artifact is resolved.

Override the CUDA architecture at build time with ``MPM_CUDA_ARCH=sm_86``
(default: ``native``).
"""

import ctypes
import importlib.resources as resources
import importlib.util
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


def _shared_library_candidates(so_name: str) -> list[Path]:
    """Return plausible locations for a packaged CUDA shared library.

    Prefer Python's package/artifact lookup so scikit-build-core's editable
    rebuild hook can run when CUDA sources changed. The source-tree ``_lib``
    path is only a fallback for older installs or direct in-tree builds.
    """
    # Ask Python's import machinery for the installed artifact. In editable
    # scikit-build-core installs, this triggers the native editable rebuild
    # hook for known wheel files when needed and gives us the built .so path.
    candidates = []
    module_name = Path(so_name).stem
    try:
        spec = importlib.util.find_spec(f"mpm_jax.cuda._lib.{module_name}")
    except Exception:
        spec = None
    if spec is not None and spec.origin:
        candidates.append(Path(spec.origin))

    try:
        resource_path = resources.files("mpm_jax.cuda._lib").joinpath(so_name)
        candidates.append(Path(str(resource_path)))
    except (ModuleNotFoundError, FileNotFoundError):
        pass

    candidates.append(_shared_library_path(so_name))

    return candidates


def _find_shared_library(so_name: str) -> Path | None:
    for candidate in _shared_library_candidates(so_name):
        if candidate.exists():
            return candidate
    return None


def _register(name: str, so_name: str, symbol: str) -> bool:
    """Load .so from the package's _lib/ dir and register the FFI target."""
    with _REGISTER_LOCK:
        if name in _REGISTERED:
            return _REGISTERED[name]

        so_path = _find_shared_library(so_name)
        if so_path is None:
            logger.warning(
                "CUDA kernel %s not found in package resources. Run "
                "`pixi install -e gpu` in an environment where nvcc is on PATH.",
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


def _register_fused():
    return _register("p2g_fused_cuda", "libp2g_fused.so", "P2GFused")


def _register_g2p_fused():
    return _register("g2p_fused_cuda", "libg2p_fused.so", "G2PFused")


def _register_inline():
    return _register("p2g_inline_cuda", "libp2g_inline.so", "P2GInline")


def _register_v2_inline():
    return _register("p2g_v2_inline_cuda", "libp2g_v2_inline.so", "P2GV2Inline")


def _register_v3_inline():
    return _register("p2g_v3_inline_cuda", "libp2g_v3_inline.so", "P2GV3Inline")


def _register_v4_inline():
    return _register("p2g_v4_inline_cuda", "libp2g_v4_inline.so", "P2GV4Inline")


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


def is_available(kernel='inline'):
    """Check if a prebuilt CUDA kernel can be loaded and registered."""
    if kernel == 'fused':
        return _register_fused()
    elif kernel == 'g2p_fused':
        return _register_g2p_fused()
    elif kernel == 'inline':
        return _register_inline()
    elif kernel == 'v2_inline':
        return _register_v2_inline()
    elif kernel == 'v3_inline':
        return _register_v3_inline()
    elif kernel == 'v4_inline':
        return _register_v4_inline()
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
                           pre_particle_fn, post_grid_fn, steps_per_frame,
                           use_cuda_g2p=True):
    """Per-frame JIT'd function using the cuda_v1_inline P2G kernel.

    Mirrors ``solver.build_jit_frame`` but routes P2G through one CUDA
    kernel call (inline weights + 27-stencil atomic scatter per particle,
    no ``(N, 27, *)`` momentum tensor in HBM). When ``use_cuda_g2p=True``
    (the default), the G2P gather also uses a CUDA kernel
    (``g2p_fused.cu``) so the ``(N, 27, *)`` weight/dweight/dpos/index
    tensors don't materialise on the G2P side either.

    Result is one ``@jax.jit`` + ``lax.scan`` over ``steps_per_frame`` —
    a single XLA program per frame. Stress and plasticity stay in JAX
    (model-agnostic); only the two scatter/gather kernels are CUDA.
    """
    if not is_available('inline'):
        raise RuntimeError(
            "cuda_v1_inline P2G kernel not registered (missing .so?). "
            "Run `pixi install -e gpu` to build.")

    if use_cuda_g2p and not is_available('g2p_fused'):
        raise RuntimeError(
            "cuda g2p kernel not registered (missing .so?). "
            "Run `pixi install -e gpu` to build, or pass "
            "use_cuda_g2p=False to fall back to the JAX G2P.")

    from mpm_jax.solver import (
        MPMState,
        compute_weights_and_indices,
        g2p,
        grid_update as grid_update_fn,
    )

    @jax.jit
    def jit_frame(state):
        def scan_body(state, _):
            with jax.named_scope("pre_particle"):
                x, v = pre_particle_fn(state.x, state.v, 0.0)
            with jax.named_scope("elasticity"):
                stress = elasticity_fn(state.F)
            with jax.named_scope("p2g_inline"):
                grid_mv, grid_m = cuda_p2g_inline(
                    x, v, state.C, stress,
                    params.num_grids, params.dt, params.vol, params.p_mass,
                    params.inv_dx, params.dx,
                )
            with jax.named_scope("grid_update"):
                grid_mv = grid_update_fn(
                    grid_mv, grid_m, params.gravity, params.dt, params.damping)
                grid_v = post_grid_fn(grid_mv, grid_m, 0.0)

            with jax.named_scope("g2p"):
                if use_cuda_g2p:
                    # CUDA G2P: gather + grad_v + state update, register-resident
                    # 27-loop. No (N, 27, *) tensors anywhere in this substep.
                    new_x, new_v, new_C, new_F = cuda_g2p_fused(
                        x, state.F, grid_v,
                        params.num_grids, params.dt,
                        params.inv_dx, params.dx, params.clip_bound,
                    )
                else:
                    # JAX G2P (materialises (N, 27, *) weights for the gather).
                    weight, dweight, dpos, index = compute_weights_and_indices(
                        x, params.inv_dx, params.dx, params.num_grids)
                    new_x, new_v, new_C, new_F = g2p(
                        grid_v, weight, dweight, dpos, index,
                        state.F, x, params.dt, params.inv_dx, params.clip_bound)

            with jax.named_scope("plasticity"):
                new_F = plasticity_fn(new_F)
            return MPMState(x=new_x, v=new_v, C=new_C, F=new_F), None

        for _ in range(steps_per_frame):
            state, _ = scan_body(state, None)
        return state

    return jit_frame


def cuda_p2g_v2_inline(x, v, C, stress, num_grids, dt, vol, p_mass, inv_dx, dx):
    """Inline-scatter CUDA P2G with warp-shuffle reduction (cuda_v2_inline).

    Same FFI signature as ``cuda_p2g_inline`` — only the C++ symbol is
    different. The kernel inserts a ``__match_any_sync`` + ``__shfl_xor_sync``
    warp reduction in front of every atomicAdd inside the 27-stencil scatter
    loop, so warp-resident contributions to the same grid_idx collapse to a
    single atomic.
    """
    N = x.shape[0]
    G = num_grids
    G3 = G ** 3
    C_flat = C.reshape(N, 9)
    stress_flat = stress.reshape(N, 9)

    grid_mv, grid_m = jax.ffi.ffi_call(
        "p2g_v2_inline_cuda",
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


def cuda_p2g_v3_inline(x, v, C, stress, num_grids, dt, vol, p_mass, inv_dx, dx):
    """Inline-scatter CUDA P2G with warp-shuffle atomic coalescing (cuda_v3_inline).

    Identical kernel-side reduction as ``cuda_p2g_v2_inline``. Designed to
    be called on Morton-sorted particles (see
    :func:`mpm_jax.morton.morton_argsort`) so adjacent warp lanes share
    stencil targets — the sort is what makes the warp reduction productive.
    """
    N = x.shape[0]
    G = num_grids
    G3 = G ** 3
    C_flat = C.reshape(N, 9)
    stress_flat = stress.reshape(N, 9)

    grid_mv, grid_m = jax.ffi.ffi_call(
        "p2g_v3_inline_cuda",
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


# Super-cell width used by cuda_v4_inline. Must match the SC #define in
# mpm_jax/cuda/kernels/p2g_v4_inline.cu. With SC=k the kernel launches
# (G/SC)^3 blocks instead of G^3 (k^3 fewer) and each block aggregates
# particles from SC^3 cells into a (SC+2)^3 smem tile. SC=2 is the sweet
# spot at G=64 — the tile stays at 4^3=64 nodes (no extra flush cost) but
# the block count drops 8x. Empirically SC=4 makes the tile too big
# (216 nodes) and the per-block flush dominates.
V4_SUPER_CELL_WIDTH = 2


def cuda_p2g_v4_inline(x_sorted, v_sorted, C_sorted, stress_sorted, cell_start,
                       num_grids, dt, vol, p_mass, inv_dx, dx):
    """Cell-major inline P2G via JAX FFI (cuda_v4_inline).

    The Python wrapper assumes the inputs are already sorted by home
    *super*-cell and that ``cell_start`` is the CSR boundary array of length
    (G/SC)^3 + 1, where SC is :data:`V4_SUPER_CELL_WIDTH`.

    The kernel uses one CUDA block per super-cell and aggregates each
    super-cell's contributions into a 4x4x4 shared-memory tile before
    flushing to HBM. The super-cell coarsening (SC=2) cuts the block count
    by 8x vs the old SC=1 cell-major variant — most of those blocks were
    empty since the jelly cube only occupies ~31K of 262K cells at G=64.
    """
    N = x_sorted.shape[0]
    G = num_grids
    G3 = G ** 3
    C_flat = C_sorted.reshape(N, 9)
    stress_flat = stress_sorted.reshape(N, 9)

    grid_mv, grid_m = jax.ffi.ffi_call(
        "p2g_v4_inline_cuda",
        (
            jax.ShapeDtypeStruct((G3, 3), jnp.float32),
            jax.ShapeDtypeStruct((G3,), jnp.float32),
        ),
        vmap_method="broadcast_all",
    )(
        x_sorted, v_sorted, C_flat, stress_flat, cell_start,
        G=np.int32(G),
        dt=np.float32(dt),
        vol=np.float32(vol),
        p_mass=np.float32(p_mass),
        inv_dx=np.float32(inv_dx),
        dx=np.float32(dx),
    )

    return grid_mv, grid_m


def _home_cell_id(x, inv_dx, G):
    """Home cell = center stencil node for the quadratic B-spline.

    Used by build_jit_frame_v4_inline for the cell-major sort.
    """
    px = x * inv_dx
    base = jnp.floor(px - 0.5).astype(jnp.int32)
    home = base + 1
    home = jnp.clip(home, 0, G - 1)
    flat = home[:, 0] * (G * G) + home[:, 1] * G + home[:, 2]
    return flat.astype(jnp.int32)


def _home_super_cell_id(x, inv_dx, G, sc=V4_SUPER_CELL_WIDTH):
    """Home super-cell id for the v4_inline cell-major sort.

    A super-cell covers ``sc^3`` grid cells. The sort key is the super-cell
    that contains the particle's home cell. Used by build_jit_frame_v4_inline
    to feed the kernel a CSR layout indexed by super-cell.
    """
    px = x * inv_dx
    base = jnp.floor(px - 0.5).astype(jnp.int32)
    home = base + 1
    home = jnp.clip(home, 0, G - 1)
    Gs = G // sc
    si = home[:, 0] // sc
    sj = home[:, 1] // sc
    sk = home[:, 2] // sc
    flat = si * (Gs * Gs) + sj * Gs + sk
    return flat.astype(jnp.int32)


def build_jit_frame_v2_inline(params, elasticity_fn, plasticity_fn,
                              pre_particle_fn, post_grid_fn, steps_per_frame,
                              use_cuda_g2p=True):
    """Per-frame JIT'd function using the cuda_v2_inline P2G kernel.

    Identical structure to ``build_jit_frame_inline``; only the P2G FFI call
    is swapped for the warp-reduction variant. G2P still uses the fused CUDA
    kernel (``cuda_g2p_fused``) when ``use_cuda_g2p=True``.
    """
    if not is_available('v2_inline'):
        raise RuntimeError(
            "cuda_v2_inline P2G kernel not registered (missing .so?). "
            "Run `pixi install -e gpu` to build.")

    if use_cuda_g2p and not is_available('g2p_fused'):
        raise RuntimeError(
            "cuda g2p kernel not registered (missing .so?). "
            "Run `pixi install -e gpu` to build, or pass "
            "use_cuda_g2p=False to fall back to the JAX G2P.")

    from mpm_jax.solver import (
        MPMState,
        compute_weights_and_indices,
        g2p,
        grid_update as grid_update_fn,
    )

    @jax.jit
    def jit_frame(state):
        def scan_body(state, _):
            with jax.named_scope("pre_particle"):
                x, v = pre_particle_fn(state.x, state.v, 0.0)
            with jax.named_scope("elasticity"):
                stress = elasticity_fn(state.F)
            with jax.named_scope("p2g_v2_inline"):
                grid_mv, grid_m = cuda_p2g_v2_inline(
                    x, v, state.C, stress,
                    params.num_grids, params.dt, params.vol, params.p_mass,
                    params.inv_dx, params.dx,
                )
            with jax.named_scope("grid_update"):
                grid_mv = grid_update_fn(
                    grid_mv, grid_m, params.gravity, params.dt, params.damping)
                grid_v = post_grid_fn(grid_mv, grid_m, 0.0)

            with jax.named_scope("g2p"):
                if use_cuda_g2p:
                    new_x, new_v, new_C, new_F = cuda_g2p_fused(
                        x, state.F, grid_v,
                        params.num_grids, params.dt,
                        params.inv_dx, params.dx, params.clip_bound,
                    )
                else:
                    weight, dweight, dpos, index = compute_weights_and_indices(
                        x, params.inv_dx, params.dx, params.num_grids)
                    new_x, new_v, new_C, new_F = g2p(
                        grid_v, weight, dweight, dpos, index,
                        state.F, x, params.dt, params.inv_dx, params.clip_bound)

            with jax.named_scope("plasticity"):
                new_F = plasticity_fn(new_F)
            return MPMState(x=new_x, v=new_v, C=new_C, F=new_F), None

        for _ in range(steps_per_frame):
            state, _ = scan_body(state, None)
        return state

    return jit_frame


def build_jit_frame_v3_inline(params, elasticity_fn, plasticity_fn,
                              pre_particle_fn, post_grid_fn, steps_per_frame,
                              use_cuda_g2p=True):
    """Per-frame JIT'd function using cuda_v3_inline (Morton sort + warp shuffle).

    Each substep sorts particles by Morton (Z-order) code, then runs the
    inline + warp-shuffle P2G kernel + CUDA G2P. State persists in sorted
    order across substeps (re-sorted each substep on the new positions).
    """
    if not is_available('v3_inline'):
        raise RuntimeError(
            "cuda_v3_inline P2G kernel not registered (missing .so?). "
            "Run `pixi install -e gpu` to build.")

    if use_cuda_g2p and not is_available('g2p_fused'):
        raise RuntimeError(
            "cuda g2p kernel not registered (missing .so?). "
            "Run `pixi install -e gpu` to build, or pass "
            "use_cuda_g2p=False to fall back to the JAX G2P.")

    from mpm_jax.morton import morton_argsort
    from mpm_jax.solver import (
        MPMState,
        compute_weights_and_indices,
        g2p,
        grid_update as grid_update_fn,
    )

    @jax.jit
    def jit_frame(state):
        def scan_body(state, _):
            with jax.named_scope("morton_sort"):
                order = morton_argsort(state.x, params.inv_dx, params.num_grids)
                x_sorted = state.x[order]
                v_sorted = state.v[order]
                C_sorted = state.C[order]
                F_sorted = state.F[order]

            with jax.named_scope("pre_particle"):
                x, v = pre_particle_fn(x_sorted, v_sorted, 0.0)
            with jax.named_scope("elasticity"):
                stress = elasticity_fn(F_sorted)
            with jax.named_scope("p2g_v3_inline"):
                grid_mv, grid_m = cuda_p2g_v3_inline(
                    x, v, C_sorted, stress,
                    params.num_grids, params.dt, params.vol, params.p_mass,
                    params.inv_dx, params.dx,
                )
            with jax.named_scope("grid_update"):
                grid_mv = grid_update_fn(
                    grid_mv, grid_m, params.gravity, params.dt, params.damping)
                grid_v = post_grid_fn(grid_mv, grid_m, 0.0)

            with jax.named_scope("g2p"):
                if use_cuda_g2p:
                    new_x, new_v, new_C, new_F = cuda_g2p_fused(
                        x, F_sorted, grid_v,
                        params.num_grids, params.dt,
                        params.inv_dx, params.dx, params.clip_bound,
                    )
                else:
                    weight, dweight, dpos, index = compute_weights_and_indices(
                        x, params.inv_dx, params.dx, params.num_grids)
                    new_x, new_v, new_C, new_F = g2p(
                        grid_v, weight, dweight, dpos, index,
                        F_sorted, x, params.dt, params.inv_dx, params.clip_bound)

            with jax.named_scope("plasticity"):
                new_F = plasticity_fn(new_F)
            return MPMState(x=new_x, v=new_v, C=new_C, F=new_F), None

        for _ in range(steps_per_frame):
            state, _ = scan_body(state, None)
        return state

    return jit_frame


def build_jit_frame_v4_inline(params, elasticity_fn, plasticity_fn,
                              pre_particle_fn, post_grid_fn, steps_per_frame,
                              use_cuda_g2p=True):
    """Per-frame JIT'd function using the cuda_v4_inline P2G kernel.

    Each substep argsorts particles by home cell, builds a CSR cell_start
    array, and runs the cell-major + smem-tile P2G kernel. State persists
    in sorted order across substeps.
    """
    if not is_available('v4_inline'):
        raise RuntimeError(
            "cuda_v4_inline P2G kernel not registered (missing .so?). "
            "Run `pixi install -e gpu` to build.")

    if use_cuda_g2p and not is_available('g2p_fused'):
        raise RuntimeError(
            "cuda g2p kernel not registered (missing .so?). "
            "Run `pixi install -e gpu` to build, or pass "
            "use_cuda_g2p=False to fall back to the JAX G2P.")

    G = params.num_grids
    sc = V4_SUPER_CELL_WIDTH
    if G % sc != 0:
        raise RuntimeError(
            f"cuda_v4_inline requires num_grids ({G}) divisible by "
            f"super-cell width ({sc})."
        )
    Gs = G // sc
    Gs3 = Gs ** 3
    super_boundaries = jnp.arange(Gs3 + 1, dtype=jnp.int32)

    from mpm_jax.solver import (
        MPMState,
        compute_weights_and_indices,
        g2p,
        grid_update as grid_update_fn,
    )

    @jax.jit
    def jit_frame(state):
        def scan_body(state, _):
            with jax.named_scope("pre_particle"):
                x, v = pre_particle_fn(state.x, state.v, 0.0)
            with jax.named_scope("elasticity"):
                stress = elasticity_fn(state.F)

            with jax.named_scope("super_cell_sort"):
                super_id = _home_super_cell_id(x, params.inv_dx, G, sc)
                order = jnp.argsort(super_id)

                x_s = x[order]
                v_s = v[order]
                C_s = state.C[order]
                stress_s = stress[order]
                F_s = state.F[order]

                super_id_sorted = super_id[order]
                cell_start = jnp.searchsorted(
                    super_id_sorted, super_boundaries
                ).astype(jnp.int32)

            with jax.named_scope("p2g_v4_inline"):
                grid_mv, grid_m = cuda_p2g_v4_inline(
                    x_s, v_s, C_s, stress_s, cell_start,
                    params.num_grids, params.dt, params.vol, params.p_mass,
                    params.inv_dx, params.dx,
                )

            with jax.named_scope("grid_update"):
                grid_mv = grid_update_fn(
                    grid_mv, grid_m, params.gravity, params.dt, params.damping)
                grid_v = post_grid_fn(grid_mv, grid_m, 0.0)

            with jax.named_scope("g2p"):
                if use_cuda_g2p:
                    new_x, new_v, new_C, new_F = cuda_g2p_fused(
                        x_s, F_s, grid_v,
                        params.num_grids, params.dt,
                        params.inv_dx, params.dx, params.clip_bound,
                    )
                else:
                    weight, dweight, dpos, index = compute_weights_and_indices(
                        x_s, params.inv_dx, params.dx, params.num_grids)
                    new_x, new_v, new_C, new_F = g2p(
                        grid_v, weight, dweight, dpos, index,
                        F_s, x_s, params.dt, params.inv_dx, params.clip_bound)

            with jax.named_scope("plasticity"):
                new_F = plasticity_fn(new_F)
            return MPMState(x=new_x, v=new_v, C=new_C, F=new_F), None

        for _ in range(steps_per_frame):
            state, _ = scan_body(state, None)
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
