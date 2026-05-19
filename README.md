# MPM-CudaJax

3D MLS-MPM (Moving Least Squares Material Point Method) solver in **JAX**
with hand-written **CUDA** kernels integrated via JAX FFI. Investigates
where JAX/XLA's automatic GPU compilation is sufficient and where custom
CUDA wins.

**Headline:** the fully fused CUDA path (`kernel=cuda_fused`) is **10–15×
faster than the JAX baseline** at 200K–5M particles on an RTX 3080, and
clears the JAX OOM ceiling — the JAX path runs out at ~5M particles on
10 GB, cuda_fused keeps going past 10M.

## Quickstart

You need [pixi](https://pixi.sh/). Everything else (Python, JAX, CUDA
toolkit deps) is pinned in `pyproject.toml` and `pixi.lock` and managed
by pixi — do **not** run `pip install` directly.

```bash
git clone git@github.com:philipnickel/MPM-CudaJax.git
cd MPM-CudaJax
```

**No GPU?** Install the default (CPU) env and run a short simulation:
```bash
pixi install
pixi run python simulate.py sim.num_frames=20
```
A jelly cube falls onto a sticky floor and renders to
`output/jelly_jax.gif`. With `sim.num_frames=20` it takes a few seconds.

**Have an NVIDIA GPU (Linux)?** Install the `gpu` env (this also builds
the custom CUDA kernels via CMake — `nvcc` and `gxx` ship from
conda-forge inside the env, no system module load needed):
```bash
pixi install -e gpu
pixi run -e gpu python simulate.py kernel=cuda_fused timing_mode=per_stage
```

To benchmark instead of rendering:
```bash
pixi run -e gpu python simulate.py \
    kernel=cuda_fused timing_mode=per_stage \
    sim.n_particles=500000 sim.num_grids=64 sim.num_frames=15 \
    benchmark=true
```
Prints `total_steps`, `elapsed_s`, `steps_per_sec`, and average
`ms/step`. No GIF, no per-frame state capture — just wall-clock timing.

Outputs:
- GIF renders → `output/<tag>_<kernel>.gif`
- Hydra logs / config snapshots → `outputs/<date>/<run>/`
- Multirun sweep results → `multirun/<date>/<run>/`
- Built CUDA `.so` files → `mpm_jax/cuda/_lib/` (rebuilds on `.cu` edit via `editable.rebuild=true`)
- wandb logs (online by default; set `WANDB_MODE=disabled` for offline)

If you want a guided tour of the kernel variants and what each one does,
see [Kernel variants](#kernel-variants) below.

## Setup

Requires [pixi](https://pixi.sh/).

```bash
git clone git@github.com:philipnickel/MPM-CudaJax.git
cd MPM-CudaJax
```

**Local (CPU only):**
```bash
pixi install
pixi run python simulate.py sim.num_frames=5
```

**GPU (Linux):**
```bash
pixi install -e gpu        # builds CUDA kernels via CMake at install time
pixi run -e gpu python simulate.py
```

CUDA kernels are built by [scikit-build-core](https://scikit-build-core.readthedocs.io/)
+ CMake during `pixi install -e gpu`. Output `.so` files land in
`mpm_jax/cuda/_lib/` and are loaded at runtime via
`jax.ffi.register_ffi_target`. The build is best-effort: when `nvcc` is
missing (the default CPU env) CMake's `check_language(CUDA)` returns
early, the wheel installs cleanly, and the JAX baseline still works.

Override the CUDA architecture at install time:
```bash
MPM_CUDA_ARCH=sm_86 pixi install -e gpu     # Ampere
MPM_CUDA_ARCH=sm_90 pixi install -e gpu     # Hopper
# default is 'native' (CMake auto-detects the local GPU)
```

**DTU HPC:** no `module load` is needed — conda-forge ships `cuda-nvcc`
and `gxx` inside the `gpu` env. Just:
```bash
MPM_CUDA_ARCH=sm_90 pixi install -e gpu
```

## Usage

```bash
# Default run (renders GIF to ./output)
pixi run -e gpu python simulate.py

# Benchmark mode (no GIF, no per-frame state capture, wall-clock timing)
pixi run -e gpu python simulate.py benchmark=true

# Pick a kernel
pixi run -e gpu python simulate.py kernel=jax        # XLA baseline
pixi run -e gpu python simulate.py kernel=cuda_fused timing_mode=per_stage  # fully fused (the winner)
pixi run -e gpu python simulate.py kernel=cuda_v1    # naive atomicAdd scatter
pixi run -e gpu python simulate.py kernel=cuda_v2    # warp-reduced scatter
pixi run -e gpu python simulate.py kernel=cuda_v4    # smem-tile scatter (slow — argsort overhead)

# Override sim params
pixi run -e gpu python simulate.py sim.n_particles=1000000 sim.num_grids=64
```

`kernel=cuda_fused` requires `timing_mode=per_stage`. The fused kernel
replaces the entire P2G + G2P pipeline so it doesn't fit the monolithic
`lax.scan` shape that `timing_mode=per_frame` uses.

## Kernel variants

Numbered `cuda_vN` labels follow the project plan (course lectures L1–L4).
`cuda_fused` is a separate, exploratory path that fully fuses P2G + G2P.

| `kernel=` | What it does |
|---|---|
| `jax` | Pure JAX/XLA. cuSOLVER SVD, vmap'd compute, `jnp.at[].add()` scatter. |
| `cuda_v1` | JAX compute, CUDA naive atomicAdd scatter (L1: CUDA Basics). |
| `cuda_v2` | JAX compute, CUDA warp-reduced scatter (`__match_any_sync`) (L2: Warp Shuffles). |
| `cuda_v4` | JAX argsort + CSR build, CUDA smem-tile scatter (L3: Shared Memory). |
| **`cuda_fused`** | **Fully fused: SVD + plasticity + corotated stress + APIC + B-spline weights + scatter in one kernel launch, plus a matching fused G2P kernel. No `(N, 27, *)` tensors materialised in HBM.** |

## Benchmark results

RTX 3080 (sm_86, 10 GB), 3D MLS-MPM, G=64³ grid, `timing_mode=per_stage`,
`benchmark=true`, wall-clock after warmup, jelly material (Corotated +
Identity plasticity), 64³ background grid, dt = 3e-4 s, 10 substeps/frame.
100–150 timed substeps per row.

**ms per substep:**

| N (particles) | `jax` | `cuda_v1` | `cuda_v2` | `cuda_v4` | **`cuda_fused`** | fused vs jax |
|---:|---:|---:|---:|---:|---:|---:|
| 5,000     | 1.41   | 1.47   | 1.31   | 3.69   | **0.15** | **9.4×** |
| 50,000    | 13.53  | 14.19  | 13.16  | 36.74  | **1.02** | **13.3×** |
| 200,000   | 51.71  | 57.28  | 52.87  | 141.46 | **3.59** | **14.4×** |
| 500,000   | 129.45 | 134.78 | 130.02 | 294.63 | **8.72** | **14.8×** |
| 1,000,000 | 242.44 | 257.53 | 249.77 | 462.76 | **16.34**| **14.8×** |

`cuda_fused` keeps going past N=1M — measured up to N=10M on this same
card (the JAX path OOMs at ~5M because it materialises `(N, 27, 3)`
tensors across the FFI boundary).

**What the numbers show:**

- **`cuda_v1` ≈ `cuda_v2` ≈ `jax`**: replacing only the scatter buys
  almost nothing. XLA's `jnp.at[].add()` is already at parity with
  hand-written warp-reduced atomicAdds on this GPU.
- **`cuda_v4` is 2-3× slower** because the JAX-side `argsort` and CSR
  build wipe out any shared-memory tiling win.
- **`cuda_fused` is 10-15× faster** at every size from N=5K up. The
  structural reason: it never materialises the `(N, 27, 3)` momentum
  tensor (or the matching weight/dweight/dpos tensors in G2P) across
  any kernel boundary. Each thread runs the full per-particle pipeline
  in registers.

The takeaway: the scatter itself is not the bottleneck on this GPU — XLA's
scatter is already near-optimal. The real cost is **materialising the
`(N, 27, 3)` momentum / weight / dweight tensors between JAX-compute and
CUDA-scatter every substep**. `cuda_fused` skips that entirely by doing the
whole pipeline in registers, one thread per particle. That's where the
order-of-magnitude win comes from.

Only `cuda_fused` supports CorotatedElasticity with Identity or Snow
plasticity (constitutive model is hard-coded inside the kernel).

## Sweeps

Pre-baked Hydra multirun sweeps:

```bash
pixi run -e gpu python simulate.py -cn sweep_per_stage   # all kernels × particle counts × both timing modes
pixi run -e gpu python simulate.py -cn sweep_cuda_fused     # cuda_fused across particle counts
pixi run -e gpu python simulate.py -cn sweep_baseline    # JAX-only scaling
pixi run -e gpu python simulate.py -cn sweep_all
pixi run -e gpu python simulate.py -cn sweep_quick
pixi run -e gpu python simulate.py -cn sweep_profile
```

Each combination gets its own `multirun/<date>/<run>/` subdir. Sweeps
must use Hydra multirun (not a bash `for` loop) so wandb runs and log
parsers see the structure they expect.

## Profiling

Three profilers are wired in via the `profile=` config:

```bash
pixi run -e gpu python simulate.py profile=nsys benchmark=true \
    kernel=cuda_fused timing_mode=per_stage sim.n_particles=200000

pixi run -e gpu python simulate.py profile=ncu  benchmark=true \
    kernel=cuda_fused timing_mode=per_stage sim.n_particles=10000 sim.num_frames=1

pixi run -e gpu python simulate.py profile=jax  benchmark=true \
    kernel=cuda_fused timing_mode=per_stage
```

For `nsys` / `ncu`, `simulate.py` **re-launches itself under the profiler**
(gated by an `_MPM_INSIDE_PROFILER` env var so the inner process knows
not to do the same thing again). The inner process is passed
`hydra.run.dir=<outer_outdir>` so its simulate.log, wandb run, and the
profile report all land in the same Hydra run dir:

```
outputs/<YYYY-MM-DD>/<HH-MM-SS>/
  ├── .hydra/                         # config snapshot
  ├── simulate.log                    # python output
  ├── profile_cuda_fused_N200000.nsys-rep   # (with profile=nsys)
  └── profile_cuda_fused_N10000.csv         # (with profile=ncu)
```

The report is also uploaded to wandb as an artifact (with `profile=jax`,
the TensorBoard trace dir is the artifact). Use the multirun output dir
naming for sweeps: each Hydra run gets its own subdir under
`multirun/<date>/<run>/`, with the same colocated structure.

Notes:
- `ncu --set full` instruments every kernel and is very slow — use
  `sim.num_frames=1` (and a small `sim.n_particles`) so it finishes.
- `nsys` only collects between `cudaProfilerStart` / `cudaProfilerStop`
  brackets (managed for you in `simulate.py`).
- For sweeps, profile one kernel at a time — `profile=nsys -cn sweep_*`
  will launch each combination under nsys independently.

`simulate.py` itself does not produce a per-stage timing breakdown in
benchmark mode — only total wall-clock. Use nsys/ncu when you need to
attribute time to P2G / grid_update / G2P.

## Config

Hydra config groups in `conf/`:

| Group | Options | Description |
|---|---|---|
| `material` | `jelly` (default), `sand` | Constitutive model |
| `sim` | `default` | n_particles, num_grids, dt, BCs, ... |
| `kernel` | `jax` (default), `cuda_v1`, `cuda_v2`, `cuda_v4`, `cuda_fused` | P2G implementation |
| `profile` | `none` (default), `nsys`, `ncu`, `jax` | GPU profiler |

Top-level fields: `benchmark`, `timing_mode` (`per_frame` or `per_stage`),
`tag`, `output_dir`. All overridable from CLI:

```bash
pixi run -e gpu python simulate.py sim.n_particles=100000 kernel=cuda_fused timing_mode=per_stage benchmark=true
```

## Tests

```bash
pixi run test
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
├── pyproject.toml           # scikit-build-core build + pixi cpu / gpu envs
├── pixi.lock                # locked deps for both envs (commit this)
├── CMakeLists.txt           # CUDA kernel build (called by scikit-build-core)
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

**`cuda_fused` collapses P2G and G2P each into a single CUDA kernel launch.**
Each thread runs the whole per-particle pipeline in registers — no
intermediate tensors of shape `(N, 27, 3)` ever exist in HBM. That's the
key structural advantage: with the other CUDA variants (v1/v3/v4) only
the scatter is replaced, and XLA still has to materialise the
`(N, 27, 3)` momentum tensor across the FFI boundary to feed it. cuda_fused
also computes its own 3×3 Jacobi SVD in-thread instead of calling
cuSOLVER, because cuSOLVER is host-side and would force the same
materialisation.

## References

- Hu et al., "A Moving Least Squares Material Point Method", ACM TOG 2018
- Stomakhin et al., "A Material Point Method for Snow Simulation", ACM TOG 2013
- Gao et al., "GPU Optimization of Material Point Methods", ACM TOG 2018
- McAdams et al., "Computing the Singular Value Decomposition of 3×3 matrices with minimal branching and elementary floating point operations", 2011
