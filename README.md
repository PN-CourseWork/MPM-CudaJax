# MPM-CudaJax

3D MLS-MPM (Moving Least Squares Material Point Method) solver in **JAX**
with hand-written **CUDA** kernels integrated via JAX FFI. Investigates
where JAX/XLA's automatic GPU compilation is sufficient and where custom
CUDA wins.

The current CLI uses one fully JIT-compiled frame path. Use `profile=jax`
to emit a TensorBoard trace with JAX host annotations and compiled
`jax.named_scope` regions for P2G, grid update, G2P, and related stages.

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
pixi run -e gpu python simulate.py kernel=cuda_v3_inline material=jelly_jacobi
```

To benchmark instead of rendering:
```bash
pixi run -e gpu python simulate.py \
    kernel=cuda_v3_inline material=jelly_jacobi \
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
pixi run -e gpu python simulate.py kernel=jax_v1_5   # scan over stencil offsets
pixi run -e gpu python simulate.py kernel=warp_v1_inline material=jelly_jacobi
pixi run -e gpu python simulate.py kernel=warp_v2_tile material=jelly_jacobi sim.n_particles=1000000
pixi run -e gpu python simulate.py kernel=warp_v3_supercell_tile material=jelly_jacobi
pixi run -e gpu python simulate.py kernel=warp_bonus_graph material=jelly_jacobi benchmark=true
pixi run -e gpu python simulate.py kernel=warp_bonus_v2_graph material=jelly_jacobi benchmark=true
pixi run -e gpu python simulate.py kernel=cuda_v2_inline material=jelly_jacobi
pixi run -e gpu python simulate.py kernel=cuda_v3_inline material=jelly_jacobi

# Override sim params
pixi run -e gpu python simulate.py sim.n_particles=1000000 sim.num_grids=64
```

`kernel=cuda_fused` is deprecated in the CLI path. The benchmark driver now
uses one fully JIT-compiled frame shape and relies on JAX traces for stage
breakdown.

## Kernel variants

Numbered `cuda_vN_inline` labels follow the project plan (course lectures L1–L4).
The old scatter-only `cuda_v1`, `cuda_v2`, and `cuda_v4` kernels were removed
because they kept the JAX-side `(N, 27, *)` materialisation bottleneck.
`cuda_fused` is a deprecated exploratory path that fully fused P2G + G2P.

| `kernel=` | What it does |
|---|---|
| `jax` | Pure JAX/XLA. cuSOLVER SVD, vmap'd compute, `jnp.at[].add()` scatter. |
| `jax_v1_5` | Pure JAX/XLA, but P2G scans over the 27 stencil offsets to avoid a large P2G intermediate. |
| `warp_v1_inline` | Inline P2G authored as an NVIDIA Warp kernel and called from inside JAX JIT through `warp.jax_experimental.jax_kernel`. |
| `warp_v2_tile` | Experimental Warp tile P2G called through `warp.jax_experimental.jax_callable`; tile-loads 64-particle blocks before Warp-native atomic scatter. |
| `warp_v3_supercell_tile` | Super-cell-owned Warp tile P2G: sort by home super-cell, accumulate a 4^3 shared tile with `tile_scatter_add`, then flush to global grid. |
| `warp_bonus_graph` | Pure Warp prototype: bins particles by super-cell, runs tiled P2G + grid update + G2P in Warp, and replays captured CUDA graphs without JAX. Currently supports `material=jelly_jacobi`. |
| `warp_bonus_v2_graph` | Pure Warp graph path that sorts particle ids only, then gathers state in tiled P2G/G2P to avoid copying sorted `x/v/C/F` buffers. Currently supports `material=jelly_jacobi`. |
| `cuda_v*_inline` | Inline-weight CUDA P2G variants that avoid the `(N, 27, *)` P2G materialisation; paired with fused CUDA G2P in the fully JITted frame path. |
| `cuda_fused` | Deprecated CLI path; retained in lower-level code/tests as the historical fully fused CUDA experiment. |

## Benchmark results

RTX 3080 (sm_86, 10 GB), 3D MLS-MPM, G=64³ grid,
`benchmark=true`, wall-clock after warmup, jelly material (Corotated +
Identity plasticity), 64³ background grid, dt = 3e-4 s, 10 substeps/frame.
100–150 timed substeps per row.

**What the numbers showed:**

The removed scatter-only CUDA variants were not the right optimization target:
replacing only XLA's scatter kept the large JAX-side `(N, 27, *)` intermediates
and bought little or nothing. The current CUDA variants move the stencil work
inside the custom kernel so the 27 contributions stay register-local.

Only `cuda_fused` supports CorotatedElasticity with Identity or Snow
plasticity (constitutive model is hard-coded inside the kernel).

## Sweeps

Pre-baked Hydra multirun sweeps:

```bash
pixi run -e gpu python simulate.py -cn sweep_baseline    # JAX-only scaling
pixi run -e gpu python simulate.py -cn sweep_all
pixi run -e gpu python simulate.py -cn sweep_quick
pixi run -e gpu python simulate.py -cn sweep_scaling
pixi run -e gpu python simulate.py -cn sweep_profile
```

Each combination gets its own `multirun/<date>/<run>/` subdir. Sweeps
should use Hydra multirun so log parsers see the structure they expect.

## Profiling

The JAX profiler is wired in via the `profile=` config:

```bash
pixi run -e gpu python simulate.py profile=jax  benchmark=true \
    kernel=cuda_v3_inline material=jelly_jacobi
```

`profile=jax` writes a TensorBoard trace into the Hydra run directory:

```
outputs/<YYYY-MM-DD>/<HH-MM-SS>/
  ├── .hydra/                         # config snapshot
  ├── simulate.log                    # python output
  ├── results.json
  └── jax_trace/
```

Use the multirun output dir naming for sweeps: each Hydra run gets its own subdir under
`multirun/<date>/<run>/`, with the same colocated structure.

The trace includes host `TraceAnnotation` sections for build/warmup/benchmark
and compiled `jax.named_scope` labels for the simulation stages.

## Config

Hydra config groups in `conf/`:

| Group | Options | Description |
|---|---|---|
| `material` | `jelly` (default), `sand` | Constitutive model |
| `sim` | `default` | n_particles, num_grids, dt, BCs, ... |
| `kernel` | `jax` (default), `jax_v1_5`, inline CUDA variants | P2G implementation |
| `profile` | `none` (default), `jax` | JAX TensorBoard trace |

Top-level fields: `benchmark`, `tag`, `output_dir`. All overridable from CLI:

```bash
pixi run -e gpu python simulate.py sim.n_particles=100000 kernel=cuda_v3_inline benchmark=true
```

## Tests

```bash
pixi run test
```

Run the focused GPU checks with:

```bash
pixi run -e gpu pytest tests/test_cuda_ffi_loader.py tests/test_jax_v1_5.py tests/test_cuda_v2_inline_matches_v1.py -q
```

## Project Structure

```
MPM-CudaJax/
├── simulate.py              # Hydra entry + JAX trace capture
├── pyproject.toml           # scikit-build-core build + pixi cpu / gpu envs
├── pixi.lock                # locked deps for both envs (commit this)
├── CMakeLists.txt           # CUDA kernel build (called by scikit-build-core)
├── conf/
│   ├── config.yaml
│   ├── material/            # jelly.yaml, sand.yaml
│   ├── sim/default.yaml
│   ├── kernel/              # jax.yaml, jax_v1_5.yaml, warp/cuda inline kernels
│   ├── profile/             # none / jax
│   └── sweep_*.yaml
├── mpm_jax/
│   ├── solver.py            # vmap single-particle fns + build_jit_frame + build_jit_stages
│   ├── warp_p2g.py          # Warp P2G kernel wrapped with warp.jax_experimental
│   ├── constitutive.py      # 5 elasticity + 4 plasticity models
│   ├── boundary.py
│   └── cuda/
│       ├── p2g_cuda.py      # FFI registration + make_fused_stages
│       ├── _lib/            # built .so files (gitignored)
│       └── kernels/
│           ├── p2g_fused.cu          # v2: fused P2G in one kernel launch
│           ├── p2g_inline.cu         # inline P2G scatter
│           ├── p2g_v2_inline.cu      # inline P2G + warp coalescing
│           ├── p2g_v3_inline.cu      # inline P2G + Morton sort
│           ├── p2g_v4_inline.cu      # cell-major inline P2G
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

The deprecated `cuda_fused` path collapses P2G and G2P each into a single CUDA kernel launch.
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
