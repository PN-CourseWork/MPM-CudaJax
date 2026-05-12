# MPM-CudaJax

3D MLS-MPM (Moving Least Squares Material Point Method) solver in **JAX** with progressively optimised hand-written **CUDA** P2G scatter kernels. The point of the project is to investigate where JAX/XLA's automatic GPU compilation is sufficient and where custom CUDA kernels are needed.

## Package manager: uv

**Always use `uv` to install, sync, and run.** Never invoke `pip`, `pip install`, `python -m pip`, or a bare `python` from the system interpreter — those will miss the project's locked environment.

Common patterns:

```bash
uv sync --extra jax              # CPU
uv sync --extra jax-cuda         # GPU (CUDA 12)
uv run --extra jax python simulate.py ...
uv run --extra jax-cuda python simulate.py ...
uv run --extra jax --with pytest python -m pytest tests/ -v
uv add <pkg>                     # add a runtime dep (edits pyproject.toml)
uv add --optional jax <pkg>      # add to the jax extra
```

The two extras are mutually exclusive in spirit — pick `jax` for CPU/local work, `jax-cuda` on a GPU box.

### CUDA kernel build (scikit-build-core + CMake)

CUDA kernels in `mpm_jax/cuda/kernels/*.cu` build via `CMakeLists.txt` driven by **scikit-build-core** at `uv sync` time. Output `.so` files land in `mpm_jax/cuda/_lib/` (gitignored) and are loaded by `mpm_jax/cuda/p2g_cuda.py` which registers them with JAX FFI (`jax.ffi.register_ffi_target` / `ffi_call`).

Key knobs:

- `MPM_CUDA_ARCH=sm_86` (or `sm_90`, etc.) at sync time → CMake picks that arch. Default is `native` (CMake auto-detects the local GPU). Set this before `uv sync` on cross-build hosts.
- If `nvcc` is not on PATH, CMake's `check_language(CUDA)` returns early and the wheel installs fine without CUDA kernels — the JAX baseline still works. Useful for CPU-only dev.
- `editable.rebuild = true` in `pyproject.toml` means edits to `.cu` sources trigger a rebuild on the next `import mpm_jax.cuda.p2g_cuda`. Manual rebuild: `make cuda` or `uv sync --extra jax-cuda --reinstall-package mpm-cudajax`.
- `[build-system].requires` pulls in `scikit-build-core>=0.10`, `cmake>=3.24`, and `jax>=0.4.20` (jax is needed at build time so CMake can `import jax.ffi` to find the FFI headers).

## Layout

```
simulate.py            Hydra entry point + wandb logging + nsys/ncu re-launch
Makefile               setup / cuda / test / sweep / clean targets
pyproject.toml         deps + scikit-build-core build config; extras: jax, jax-cuda
CMakeLists.txt         CUDA kernel build (called by scikit-build-core)
ruff.toml              lint config
conf/                  Hydra config groups
  config.yaml          top-level defaults (material/sim/kernel/profile)
  material/            jelly.yaml, sand.yaml          (constitutive model)
  sim/default.yaml     n_particles, num_grids, dt, BCs, ...
  kernel/              jax.yaml, cuda_v1..v4.yaml      (P2G scatter impl)
  profile/             none.yaml, nsys.yaml, ncu.yaml, jax.yaml
  sweep_*.yaml         pre-baked Hydra multirun sweeps
mpm_jax/
  solver.py            single-particle fns + vmap + lax.scan JIT'd frame
  constitutive.py      5 elasticity + 4 plasticity models
  boundary.py          6 boundary condition types
  cuda/
    p2g_cuda.py        loads prebuilt .so + jax.ffi.register_ffi_target
    _lib/              prebuilt .so files (gitignored, populated by CMake)
    kernels/
      p2g_scatter.cu        v1: one thread/particle, global atomicAdd
      p2g_scatter_warp.cu   v3: __match_any_sync warp reduction
      p2g_scatter_smem.cu   v4: shared-memory tile staging (sorted particles)
      p2g_fused.cu          v2: SVD + plasticity + stress + APIC + scatter in one launch
      g2p_fused.cu          v2: gather + grad_v + update in one launch (pairs with p2g_fused)
tests/                 pytest suite (boundary, constitutive, ffi loader, integration, solver, cuda_equivalence)
docs/superpowers/      design specs and implementation plans
```

## Architecture (one timestep)

Three embarrassingly parallel phases per substep:

1. **P2G** — per-particle: stress (SVD) + B-spline weights + APIC momentum → scatter to grid
2. **Grid update** — per-node: normalize momentum, gravity, boundary conditions
3. **G2P** — per-particle: gather grid velocities, update position/velocity/F

Two JIT structures, selected by `timing_mode`:

- **`per_frame`** (`build_jit_frame` in solver.py): full frame including all substeps wrapped in one `@jax.jit` + `lax.scan`. One XLA program per frame.
- **`per_stage`** (`build_jit_stages`): host Python loop with three individual JITs (`jit_p2g_stage`, `jit_grid_stage`, `jit_g2p_stage`). Required for `kernel=cuda_v2` (the fused kernel doesn't fit the per_frame shape). In benchmark mode neither path does any intra-loop `block_until_ready` — one sync at the very end and elapsed/num_frames is the average.

**`kernel=cuda_v2` is the path that actually wins** (10–15× over JAX at N≥200K). It replaces P2G *and* G2P with two fused CUDA kernels (`p2g_fused.cu`, `g2p_fused.cu`); each thread runs the entire per-particle pipeline in registers, so no `(N, 27, *)` tensors ever materialise in HBM. That's the structural advantage — the other CUDA variants (v1/v3/v4) still pay for that materialisation across the FFI boundary, which is why they tie JAX rather than beating it.

`StepIntermediates` (the namedtuple passed from p2g_stage to g2p_stage) is intentionally minimal — only `x_post_bc` and `F_pre_plast`. Carrying weight / dweight / dpos / index would cost ~1148 bytes per particle of inter-stage HBM and OOM the GPU at N=5M. G2P recomputes B-spline weights from `x_post_bc` instead.

## Common commands

```bash
# Default run (renders GIF to ./output)
uv run --extra jax-cuda python simulate.py

# Benchmark mode (timing only, no GIF)
uv run --extra jax-cuda python simulate.py benchmark=true

# Switch P2G kernel (cuda_v2 is the only one that beats JAX — others tie or lose)
uv run --extra jax-cuda python simulate.py kernel=jax                                   # XLA default
uv run --extra jax-cuda python simulate.py kernel=cuda_v2 timing_mode=per_stage         # fully fused (10-15x at N>=200K)
uv run --extra jax-cuda python simulate.py kernel=cuda_v1                               # naive atomicAdd (ties jax)
uv run --extra jax-cuda python simulate.py kernel=cuda_v3                               # warp reduction (ties jax)
uv run --extra jax-cuda python simulate.py kernel=cuda_v4                               # smem staging (slow - argsort overhead)

# Override sim params
uv run --extra jax-cuda python simulate.py sim.n_particles=50000 sim.num_grids=64

# Profilers (auto re-launches under nsys / ncu)
uv run --extra jax-cuda python simulate.py profile=nsys benchmark=true
uv run --extra jax-cuda python simulate.py profile=ncu  sim.num_frames=1 benchmark=true
uv run --extra jax-cuda python simulate.py profile=jax  benchmark=true

# Sweeps (Hydra multirun)
uv run --extra jax-cuda python simulate.py -cn sweep_per_stage   # all kernels x sizes x timing_mode
uv run --extra jax-cuda python simulate.py -cn sweep_cuda_v2     # cuda_v2 across particle counts
uv run --extra jax-cuda python simulate.py -cn sweep_baseline    # JAX-only scaling
uv run --extra jax-cuda python simulate.py -cn sweep_all
uv run --extra jax-cuda python simulate.py -cn sweep_quick
uv run --extra jax-cuda python simulate.py -cn sweep_profile

# Tests
uv run --extra jax --with pytest python -m pytest tests/ -v

# Lint
uv run --with ruff ruff check .
```

`make setup` / `make cuda` / `make test` / `make sweep` / `make clean` wrap the above and load the DTU HPC modules (`nvhpc/26.1 gcc/15.2`).

## DTU HPC notes

On the cluster the toolchain comes from modules:

```bash
module load nvhpc/26.1 gcc/15.2
export LD_LIBRARY_PATH=/appl/gcc/15.2.0-binutils-2.45/lib64:$LD_LIBRARY_PATH
export PATH=/appl/nvhpc/2024_249/Linux_aarch64/24.9/cuda/bin:$PATH   # ensures nvcc on PATH for CMake
MPM_CUDA_ARCH=sm_90 uv sync --extra jax-cuda                          # build for Hopper
```

CMake auto-detects the local GPU arch when `MPM_CUDA_ARCH` is unset. `make setup` wraps the module-load + `uv sync` in one step.

## Conventions

- **Sweeps must use Hydra multirun**, never a bash `for` loop. Either use a pre-baked sweep config (`-cn sweep_*`) or pass the axes inline with `-m / --multirun`, e.g. `uv run --extra jax-cuda python simulate.py -m sim.n_particles=5000,50000,200000 kernel=jax,cuda_v1,cuda_v3 timing_mode=per_frame,per_stage benchmark=true`. For repeated experiments, add a new `conf/sweep_<name>.yaml` rather than encoding the grid in shell. Hydra puts each combination in its own `multirun/<date>/<run>/` subdir, which is what wandb / log parsers expect.
- **Default to short benchmarks.** Steady-state ms/step is locked in after the first frame (the warmup), so `sim.num_frames=5` (= 50 substeps) gives stable timings — don't burn 10× the wall time on `num_frames=30` unless you specifically need tight per-frame std. Bump it only when an individual measurement looks noisy.
- Single-particle functions live in `mpm_jax/solver.py`; vectorise via `jax.vmap`. Don't write batched code by hand — vmap is the contract.
- A new CUDA scatter kernel = a new `.cu` file in `mpm_jax/cuda/kernels/`, a `_register_*` + `cuda_p2g_scatter_*` wrapper in `p2g_cuda.py`, a `kernel=cuda_vX` branch in `simulate.run_jax`, a matching `conf/kernel/cuda_vX.yaml`, **and add the kernel name to the `KERNELS` list in `CMakeLists.txt`**. After editing, `make cuda` (or any `uv sync --reinstall-package mpm-cudajax`) rebuilds.
- For a kernel that replaces an entire stage (like cuda_v2's fused P2G+G2P), follow the `make_fused_stages` pattern in `p2g_cuda.py`: build three JIT'd stage closures (`jit_p2g_*`, `jit_grid_*`, `jit_g2p_*`) and route to them from a `kernel_name == '...'` branch in `simulate.run_jax`. Per-frame mode won't work without a matching `build_jit_frame_*`; just error out if requested.
- **No `block_until_ready` inside the timed region in benchmark mode.** Both timing modes dispatch all frames back-to-back and sync exactly once after the loop; elapsed/num_frames is the average. Per-stage breakdown comes from `profile=nsys|ncu`, not from `simulate.py`'s output.
- Boundary conditions and constitutive models are registry-based (`REGISTRY` dict in `constitutive.py`, `build_boundary_fns` in `boundary.py`); add a function and a config entry.
- `simulate.py` re-launches itself under `nsys` / `ncu` when `profile=nsys|ncu`, gated by the `_MPM_INSIDE_PROFILER` env var. Don't add wandb calls inside `run_jax`'s timed loop.
- Lint with ruff (config in `ruff.toml`); `I` is allowed as a variable name (identity matrix), and `tests/*` skips E402/F401.

## Don't

- Don't run `pip install` — use `uv add` / `uv sync`.
- Don't commit `build/`, `output/`, `outputs/`, `multirun/`, `wandb/`, `*.nsys-rep`, `*.sqlite`, or `uv.lock` (`.gitignore` covers these).
- Don't bypass the JIT'd frame (`build_jit_frame`) for benchmarking — `simulate_frame` exists only for unjitted per-stage profiling.
- Don't hard-code particle counts, grid sizes, or material params in code — they live in `conf/`.
