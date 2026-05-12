# MPM-CudaJax

MLS-MPM (Moving Least Squares Material Point Method) solver in **JAX** with progressively optimised hand-written **CUDA** P2G scatter kernels. Investigates how far JAX/XLA's automatic GPU compilation gets and where custom CUDA is needed.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:philipnickel/MPM-CudaJax.git
cd MPM-CudaJax
```

**Local (CPU):**
```bash
uv run --extra jax python simulate.py sim.num_frames=5
```

**DTU HPC (GPU):**
```bash
module load nvhpc/26.1 gcc/15.2
export LD_LIBRARY_PATH=/appl/gcc/15.2.0-binutils-2.45/lib64:$LD_LIBRARY_PATH
export NVCC=/appl/nvhpc/2024_249/Linux_aarch64/24.9/cuda/bin/nvcc

uv run --extra jax-cuda python simulate.py sim.num_frames=5 benchmark=true
```

CUDA kernels auto-compile on first use when `nvcc` is on PATH. Compiled shared
libraries are cached under `build/ffi` by default; set `MPM_FFI_CACHE` to use a
different cache directory.

## Usage

```bash
# Run simulation (renders GIF to output/)
uv run --extra jax-cuda python simulate.py

# Benchmark mode (timing only, no GIF, logs to wandb)
uv run --extra jax-cuda python simulate.py benchmark=true

# Switch P2G kernel
uv run --extra jax-cuda python simulate.py kernel=jax         # XLA default
uv run --extra jax-cuda python simulate.py kernel=cuda_v1     # naive atomicAdd
uv run --extra jax-cuda python simulate.py kernel=cuda_v3     # warp-reduction scatter

# Override sim params
uv run --extra jax-cuda python simulate.py sim.n_particles=50000 sim.num_grids=64
```

## Sweeps

Pre-configured Hydra multirun sweeps:

```bash
# Baseline scaling: vary particle count (JAX only)
uv run --extra jax-cuda python simulate.py -cn sweep_baseline

# Compare all kernels × particle counts
uv run --extra jax-cuda python simulate.py -cn sweep_all

# Quick sanity check (2 sizes)
uv run --extra jax-cuda python simulate.py -cn sweep_quick
```

## Profiling

Profile any run by setting the `profile` config:

```bash
# Nsight Systems (GPU timeline)
uv run --extra jax-cuda python simulate.py profile=nsys benchmark=true

# Nsight Compute (per-kernel metrics — slow, use few frames)
uv run --extra jax-cuda python simulate.py profile=ncu sim.num_frames=1 benchmark=true

# JAX trace (TensorBoard)
uv run --extra jax-cuda python simulate.py profile=jax benchmark=true

# Sweep profilers × kernels
uv run --extra jax-cuda python simulate.py -cn sweep_profile
```

`nsys` and `ncu` auto-relaunch the process under the profiler. Results are extracted and uploaded to wandb as artifacts.

## Config

Hydra config groups in `conf/`:

| Group | Options | Description |
|-------|---------|-------------|
| `material` | `jelly` (default), `sand` | Constitutive model |
| `sim` | `default` | Simulation params (n_particles, num_grids, dt, ...) |
| `kernel` | `jax` (default), `cuda_v1`, `cuda_v3` | P2G scatter implementation |
| `profile` | `none` (default), `nsys`, `ncu`, `jax` | GPU profiler |

All parameters overridable from CLI, e.g. `sim.n_particles=100000 sim.num_grids=128`.

## Tests

```bash
uv run --extra jax --with pytest python -m pytest tests/ -v
```

## Project Structure

```
MPM-CudaJax/
├── simulate.py              # Hydra entry point + wandb logging
├── Makefile                 # setup, sweep, clean targets
├── conf/
│   ├── config.yaml          # defaults
│   ├── material/            #   jelly.yaml, sand.yaml
│   ├── sim/                 #   default.yaml
│   ├── kernel/              #   jax.yaml, cuda_v1.yaml, cuda_v3.yaml
│   ├── profile/             #   none.yaml, nsys.yaml, ncu.yaml, jax.yaml
│   └── sweep_*.yaml         #   pre-configured multirun sweeps
├── mpm_jax/
│   ├── solver.py            #   vmap single-particle functions + lax.scan JIT
│   ├── constitutive.py      #   5 elasticity + 4 plasticity models
│   ├── boundary.py          #   6 boundary condition types
│   └── cuda/
│       ├── p2g_cuda.py      #   auto-compile + JAX FFI registration
│       └── kernels/
│           ├── p2g_scatter.cu       # v1: one thread/particle, global atomicAdd
│           ├── p2g_scatter_warp.cu  # v3: __match_any_sync warp reduction
│           ├── p2g_scatter_smem.cu  # v4: shared memory staging (WIP)
│           ├── p2g_fused.cu         # v2: fused stress+scatter (WIP)
│           └── Makefile
└── tests/                   # 24 tests
```

## Architecture

Three embarrassingly parallel phases per timestep:

1. **P2G** — per-particle: stress (SVD) + B-spline weights + affine momentum → scatter to grid
2. **Grid update** — per-node: normalize momentum, gravity, boundary conditions
3. **G2P** — per-particle: gather grid velocities, update position/velocity/F

Each phase is a `jax.vmap` over a single-particle function. The entire frame (multiple substeps) is JIT-compiled as one XLA program via `jax.lax.scan` — zero Python overhead.

The P2G scatter (the only cross-particle reduction) is the CUDA optimisation target. Kernels integrate via `jax.ffi` for zero-copy GPU memory access on the correct CUDA stream. They auto-compile from `.cu` source on first use and cache the resulting shared libraries under `build/ffi`.

## References

- Hu et al., "A Moving Least Squares Material Point Method", ACM TOG 2018
- Stomakhin et al., "A Material Point Method for Snow Simulation", ACM TOG 2013
- Gao et al., "GPU Optimization of Material Point Methods", ACM TOG 2018
