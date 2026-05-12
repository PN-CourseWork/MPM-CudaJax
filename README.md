# MPM-CudaJax

3D MLS-MPM (Moving Least Squares Material Point Method) solver in **JAX**
with hand-written **CUDA** kernels integrated via JAX FFI. Investigates
where JAX/XLA's automatic GPU compilation is sufficient and where custom
CUDA wins.

**Headline:** the fully fused CUDA path (`kernel=cuda_v2`) is **10–15×
faster than the JAX baseline** at 200K–5M particles on an RTX 3080, and
clears the JAX OOM ceiling — the JAX path runs out at ~5M particles on
10 GB, cuda_v2 keeps going past 10M.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:philipnickel/MPM-CudaJax.git
cd MPM-CudaJax
```

**Local (CPU only):**
```bash
uv sync --extra jax
uv run --extra jax python simulate.py sim.num_frames=5
```

**GPU:**
```bash
uv sync --extra jax-cuda          # builds CUDA kernels via CMake at install time
uv run --extra jax-cuda python simulate.py
```

CUDA kernels are built by [scikit-build-core](https://scikit-build-core.readthedocs.io/)
+ CMake during `uv sync`. Output `.so` files land in `mpm_jax/cuda/_lib/`
and are loaded at runtime via `jax.ffi.register_ffi_target`. The build is
best-effort: when `nvcc` is missing the CMake step returns early, the wheel
installs cleanly, and the JAX baseline still works.

Override the CUDA architecture at build time:
```bash
MPM_CUDA_ARCH=sm_86 uv sync --extra jax-cuda     # Ampere
MPM_CUDA_ARCH=sm_90 uv sync --extra jax-cuda     # Hopper
# default is 'native' (CMake auto-detects the local GPU)
```

**DTU HPC:**
```bash
module load nvhpc/26.1 gcc/15.2
export LD_LIBRARY_PATH=/appl/gcc/15.2.0-binutils-2.45/lib64:$LD_LIBRARY_PATH
export PATH=/appl/nvhpc/2024_249/Linux_aarch64/24.9/cuda/bin:$PATH
MPM_CUDA_ARCH=sm_90 uv sync --extra jax-cuda
```

## Usage

```bash
# Default run (renders GIF to ./output)
uv run --extra jax-cuda python simulate.py

# Benchmark mode (no GIF, no per-frame state capture, wall-clock timing)
uv run --extra jax-cuda python simulate.py benchmark=true

# Pick a kernel
uv run --extra jax-cuda python simulate.py kernel=jax        # XLA baseline
uv run --extra jax-cuda python simulate.py kernel=cuda_v2 timing_mode=per_stage  # fully fused (the winner)
uv run --extra jax-cuda python simulate.py kernel=cuda_v1    # naive atomicAdd scatter
uv run --extra jax-cuda python simulate.py kernel=cuda_v3    # warp-reduced scatter
uv run --extra jax-cuda python simulate.py kernel=cuda_v4    # smem-tile scatter (slow — argsort overhead)

# Override sim params
uv run --extra jax-cuda python simulate.py sim.n_particles=1000000 sim.num_grids=64
```

`kernel=cuda_v2` requires `timing_mode=per_stage`. The fused kernel
replaces the entire P2G + G2P pipeline so it doesn't fit the monolithic
`lax.scan` shape that `timing_mode=per_frame` uses.

## Kernel variants

| `kernel=` | What it does | Speedup vs JAX |
|---|---|---|
| `jax` | Pure JAX/XLA. cuSOLVER SVD, vmap'd compute, `jnp.at[].add()` scatter. | 1.0× (baseline) |
| `cuda_v1` | JAX compute, CUDA naive atomicAdd scatter. | ≈ 1.0× |
| `cuda_v3` | JAX compute, CUDA warp-reduced scatter (`__match_any_sync`). | ≈ 1.0× |
| `cuda_v4` | JAX argsort + CSR build, CUDA smem-tile scatter. | ~0.4× (argsort dominates) |
| **`cuda_v2`** | **Fully fused: SVD + plasticity + corotated stress + APIC + B-spline weights + scatter in one kernel launch, plus a matching fused G2P kernel. No `(N, 27, *)` tensors materialised in HBM.** | **10–15×** at 200K–5M particles |

The takeaway: the scatter itself is not the bottleneck on this GPU — XLA's
scatter is already near-optimal. The real cost is **materialising the
`(N, 27, 3)` momentum / weight / dweight tensors between JAX-compute and
CUDA-scatter every substep**. `cuda_v2` skips that entirely by doing the
whole pipeline in registers, one thread per particle. That's where the
order-of-magnitude win comes from.

Only `cuda_v2` supports CorotatedElasticity with Identity or Snow
plasticity (constitutive model is hard-coded inside the kernel).

## Sweeps

Pre-baked Hydra multirun sweeps:

```bash
uv run --extra jax-cuda python simulate.py -cn sweep_per_stage   # all kernels × particle counts × both timing modes
uv run --extra jax-cuda python simulate.py -cn sweep_cuda_v2     # cuda_v2 across particle counts
uv run --extra jax-cuda python simulate.py -cn sweep_baseline    # JAX-only scaling
uv run --extra jax-cuda python simulate.py -cn sweep_all
uv run --extra jax-cuda python simulate.py -cn sweep_quick
uv run --extra jax-cuda python simulate.py -cn sweep_profile
```

Each combination gets its own `multirun/<date>/<run>/` subdir. Sweeps
must use Hydra multirun (not a bash `for` loop) so wandb runs and log
parsers see the structure they expect.

## Profiling

```bash
uv run --extra jax-cuda python simulate.py profile=nsys benchmark=true
uv run --extra jax-cuda python simulate.py profile=ncu  sim.num_frames=1 benchmark=true
uv run --extra jax-cuda python simulate.py profile=jax  benchmark=true
```

`nsys` / `ncu` auto-relaunch this process under the profiler (gated by
`_MPM_INSIDE_PROFILER` env var). Results are uploaded to wandb as
artifacts. `profile=jax` writes a TensorBoard-readable trace.

`simulate.py` itself does not produce a per-stage timing breakdown in
benchmark mode — only total wall-clock. Use nsys/ncu when you need to
attribute time to P2G / grid_update / G2P.

## Config

Hydra config groups in `conf/`:

| Group | Options | Description |
|---|---|---|
| `material` | `jelly` (default), `sand` | Constitutive model |
| `sim` | `default` | n_particles, num_grids, dt, BCs, ... |
| `kernel` | `jax` (default), `cuda_v1`, `cuda_v2`, `cuda_v3`, `cuda_v4` | P2G implementation |
| `profile` | `none` (default), `nsys`, `ncu`, `jax` | GPU profiler |

Top-level fields: `benchmark`, `timing_mode` (`per_frame` or `per_stage`),
`tag`, `output_dir`. All overridable from CLI:

```bash
uv run --extra jax-cuda python simulate.py sim.n_particles=100000 kernel=cuda_v2 timing_mode=per_stage benchmark=true
```

## Tests

```bash
uv run --extra jax --with pytest python -m pytest tests/ -v
```

32 tests:
- 28 CPU tests (solver, constitutive, boundary, FFI loader, integration)
- 4 GPU equivalence tests (each CUDA variant produces same trajectory as
  JAX baseline within bounded tolerance — automatically skipped on
  CPU-only installs)

## Project Structure

```
MPM-CudaJax/
├── simulate.py              # Hydra entry + wandb + profiler re-launch
├── pyproject.toml           # scikit-build-core build, jax / jax-cuda extras
├── CMakeLists.txt           # CUDA kernel build (called by scikit-build-core)
├── Makefile                 # setup / cuda / test / sweep / clean shortcuts
├── conf/
│   ├── config.yaml
│   ├── material/            # jelly.yaml, sand.yaml
│   ├── sim/default.yaml
│   ├── kernel/              # jax.yaml, cuda_v1..v4.yaml
│   ├── profile/             # none / nsys / ncu / jax
│   └── sweep_*.yaml
├── mpm_jax/
│   ├── solver.py            # vmap single-particle fns + build_jit_frame + build_jit_stages
│   ├── constitutive.py      # 5 elasticity + 4 plasticity models
│   ├── boundary.py
│   └── cuda/
│       ├── p2g_cuda.py      # FFI registration + make_fused_stages
│       ├── _lib/            # built .so files (gitignored)
│       └── kernels/
│           ├── p2g_scatter.cu        # v1: naive atomicAdd scatter
│           ├── p2g_scatter_warp.cu   # v3: __match_any_sync warp reduction
│           ├── p2g_scatter_smem.cu   # v4: smem tile staging
│           ├── p2g_fused.cu          # v2: fused P2G in one kernel launch
│           └── g2p_fused.cu          # v2: fused G2P (paired with p2g_fused)
└── tests/
```

## Architecture

Three embarrassingly parallel phases per timestep:

1. **P2G** — per-particle: stress (SVD) + B-spline weights + APIC momentum → scatter to grid
2. **Grid update** — per-node: normalize momentum, apply gravity + damping + boundary conditions
3. **G2P** — per-particle: gather grid velocities, update position/velocity/F

Each phase is implemented as a `jax.vmap` over a single-particle function.
The pure-JAX path JIT-compiles the entire frame (multiple substeps) as
one XLA program via `jax.lax.scan`.

**`cuda_v2` collapses P2G and G2P each into a single CUDA kernel launch.**
Each thread runs the whole per-particle pipeline in registers — no
intermediate tensors of shape `(N, 27, 3)` ever exist in HBM. That's the
key structural advantage: with the other CUDA variants (v1/v3/v4) only
the scatter is replaced, and XLA still has to materialise the
`(N, 27, 3)` momentum tensor across the FFI boundary to feed it. cuda_v2
also computes its own 3×3 Jacobi SVD in-thread instead of calling
cuSOLVER, because cuSOLVER is host-side and would force the same
materialisation.

## References

- Hu et al., "A Moving Least Squares Material Point Method", ACM TOG 2018
- Stomakhin et al., "A Material Point Method for Snow Simulation", ACM TOG 2013
- Gao et al., "GPU Optimization of Material Point Methods", ACM TOG 2018
- McAdams et al., "Computing the Singular Value Decomposition of 3×3 matrices with minimal branching and elementary floating point operations", 2011
