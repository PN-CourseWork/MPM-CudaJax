# Refactor: Pluggable P2G with cuda.core Integration

**Date:** 2026-03-24
**Branch:** `dev`
**Status:** Design approved

## Goal

Simplify the MPM-CudaJax codebase while preserving all functionality. Replace the monolithic JIT'd timestep loop with a Python-level loop calling individually-JIT'd stages. Make P2G pluggable so JAX baseline and CUDA kernel variants share the same interface. Replace JAX FFI with `cuda.core` for native CUDA kernel compilation and launch. Ensure all metrics are observable via wandb.

## Key Decisions

1. **Python timestep loop** — no `lax.scan` over the full frame. Each stage (`p2g`, `grid_update`, `g2p`) is called individually, enabling per-stage timing and CUDA kernel interop.
2. **Pluggable P2G** — all variants implement `(state, params) -> (grid_mv, grid_m)`. Selected at runtime via Hydra config.
3. **`cuda.core`** — replaces JAX FFI for kernel compilation and launch. Runtime JIT from `.cu` source, cached after first compile. GPU pointer interop via `jax.dlpack` / `__cuda_array_interface__` (see JAX–CUDA Interop section).
4. **Shared CUDA compute** — `p2g_compute.cuh` device header contains B-spline weights, SVD stress, APIC momentum. Scatter variants only differ in how they write to the grid.
5. **Wandb logging** — dict-based `wandb.log()` for time series, `wandb.summary.update()` for headline metrics. Hydra config auto-captured.
6. **Removed:** shared-memory kernel (v4). **Kept:** JAX baseline, CUDA naive scatter, CUDA warp-reduced scatter.
7. **Naming:** existing `cuda_v1`/`cuda_v3` are renamed to `cuda_naive`/`cuda_warp` for clarity. The old `cuda_v2` (fused WIP) is superseded — the new CUDA variants are all fused (compute+scatter in one kernel launch) by design.

## Project Structure

```
MPM-CudaJax/
├── simulate.py              # Hydra entry, timestep loop, wandb logging
├── mpm_jax/
│   ├── state.py             # MPMState, MPMParams (NamedTuples)
│   ├── grid_update.py       # @jax.jit grid_update(grid_mv, grid_m, params) -> grid_v
│   ├── g2p.py               # @jax.jit g2p(state, grid_v, params) -> MPMState
│   ├── constitutive.py      # stress models (kept as-is)
│   ├── boundary.py          # boundary conditions (kept as-is)
│   ├── p2g/
│   │   ├── __init__.py      # get_p2g_fn(cfg) -> callable
│   │   ├── jax.py           # Pure JAX P2G (baseline)
│   │   ├── cuda_naive.py    # CUDA naive scatter (atomicAdd per node)
│   │   └── cuda_warp.py     # CUDA warp-reduced scatter (__shfl_down_sync)
│   └── cuda/
│       ├── runtime.py       # cuda.core device/stream/compile helpers
│       └── kernels/
│           ├── p2g_compute.cuh       # Shared device function
│           ├── p2g_scatter_naive.cu  # Naive: atomicAdd for all 27 nodes
│           └── p2g_scatter_warp.cu   # Warp shuffle reduce, fewer atomics
├── conf/
│   ├── config.yaml
│   ├── kernel/              # jax.yaml, cuda_naive.yaml, cuda_warp.yaml
│   ├── material/            # jelly.yaml, sand.yaml
│   └── sim/                 # default.yaml
└── tests/
```

## P2G Interface

All P2G variants implement the same signature:

```python
def p2g(state: MPMState, params: MPMParams) -> tuple[jax.Array, jax.Array]:
    """
    Particle-to-grid transfer.
    Returns (grid_mv, grid_m):
        grid_mv: (num_grids³, 3) momentum
        grid_m:  (num_grids³,)   mass
    """
```

### What moves inside P2G

The current codebase computes stress and weights in `step()` before calling `p2g()`. In the new design, **all of that moves inside each P2G variant**. Each variant is responsible for:
1. Computing B-spline weights and grid indices
2. Computing stress from deformation gradient F (calls constitutive model)
3. Computing APIC momentum contributions
4. Scattering to the grid

This means the constitutive model (`elasticity_fn`) is called within P2G — passed in during factory creation or captured in the closure. The JAX baseline wraps all of this in a single `@jax.jit`. The CUDA variants do it all on the GPU in one kernel launch.

### JAX Baseline (`p2g/jax.py`)

Pure JAX implementation. Computes weights, stress, and APIC momentum via `jax.vmap`, then scatters to grid with `jnp.ndarray.at[].add()`. Individually JIT'd.

### CUDA Naive (`p2g/cuda_naive.py`)

Factory function `make_cuda_naive_p2g(cfg, runtime)` compiles the kernel once, returns a closure:

```python
def make_cuda_naive_p2g(cfg, runtime: CudaRuntime):
    kernel = runtime.compile_kernel("kernels/p2g_scatter_naive.cu", "p2g_scatter_naive")

    def p2g(state, params):
        # Allocate output grid arrays
        grid_mv = jnp.zeros((params.num_grids**3, 3), dtype=jnp.float32)
        grid_m = jnp.zeros((params.num_grids**3,), dtype=jnp.float32)

        # Launch CUDA kernel (compute + scatter in one kernel)
        # Get device pointers (see JAX–CUDA Interop section)
        runtime.launch(kernel, grid=ceil(n_particles/256), block=256,
            get_ptr(state.x), get_ptr(state.v),
            get_ptr(state.C), get_ptr(state.F),
            get_ptr(grid_mv), get_ptr(grid_m),
            params.dt, params.vol, params.p_mass,
            params.inv_dx, params.num_grids, n_particles)

        return grid_mv, grid_m

    return p2g
```

### CUDA Warp (`p2g/cuda_warp.py`)

Same pattern as naive, different kernel. Warp-level reduction via `__shfl_down_sync` means fewer `atomicAdd` calls.

## CUDA Kernel Structure

### Shared Compute (`p2g_compute.cuh`)

Device function computing 27 grid node contributions for one particle:

```cuda
struct ParticleContrib {
    float mv[3];    // momentum contribution
    float m;        // mass contribution
    int grid_idx;   // flat grid index
};

__device__ void p2g_compute(
    const float* x, const float* v, const float* C, const float* F,
    float dt, float vol, float p_mass, float inv_dx, int num_grids,
    ParticleContrib out[27]
);
```

Contains: quadratic B-spline weight calculation, SVD of deformation gradient F, stress computation via constitutive model, APIC momentum assembly.

**Constitutive model in CUDA:** The initial CUDA kernels hardcode the corotated elasticity model (jelly). Sand and other models can be added later via compile-time flags (`#ifdef`) or template parameters. This matches the project's focus — jelly is the primary benchmark material, and the CUDA course goal is optimizing the scatter, not the stress computation.

### Naive Scatter (`p2g_scatter_naive.cu`)

```cuda
#include "p2g_compute.cuh"

extern "C" __global__ void p2g_scatter_naive(/* ... */) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= n_particles) return;

    ParticleContrib contrib[27];
    p2g_compute(..., contrib);

    for (int i = 0; i < 27; i++) {
        atomicAdd(&grid_mv[contrib[i].grid_idx * 3 + 0], contrib[i].mv[0]);
        atomicAdd(&grid_mv[contrib[i].grid_idx * 3 + 1], contrib[i].mv[1]);
        atomicAdd(&grid_mv[contrib[i].grid_idx * 3 + 2], contrib[i].mv[2]);
        atomicAdd(&grid_m[contrib[i].grid_idx], contrib[i].m);
    }
}
```

### Warp-Reduced Scatter (`p2g_scatter_warp.cu`)

Same compute, but scatter uses `__match_any_sync` to find warp lanes targeting the same grid node, reduces via `__shfl_down_sync`, single lane does `atomicAdd`.

## CUDA Runtime (`cuda/runtime.py`)

```python
from cuda.core import Device, Program, ProgramOptions, LaunchConfig, launch

class CudaRuntime:
    def __init__(self):
        self.dev = Device()
        self.dev.set_current()
        self.stream = self.dev.create_stream()
        self._cache = {}

    def compile_kernel(self, source_path: str, kernel_name: str):
        if kernel_name not in self._cache:
            code = Path(source_path).read_text()
            prog = Program(code, code_type="c++", options=ProgramOptions(
                std="c++17", arch=f"sm_{self.dev.arch}"
            ))
            mod = prog.compile("cubin")
            self._cache[kernel_name] = mod.get_kernel(kernel_name)
        return self._cache[kernel_name]

    def launch(self, kernel, grid, block, *args):
        config = LaunchConfig(grid=grid, block=block)
        launch(self.stream, config, kernel, *args)
        self.stream.sync()
```

## grid_update and g2p Signatures

These are simplified from the current multi-argument signatures to use `MPMState` and `MPMParams`:

```python
@jax.jit
def grid_update(grid_mv: jax.Array, grid_m: jax.Array, params: MPMParams) -> jax.Array:
    """Normalize momentum, apply gravity, boundary conditions, damping.
    Returns grid_v: (num_grids³, 3) velocity."""

@jax.jit
def g2p(state: MPMState, grid_v: jax.Array, params: MPMParams) -> MPMState:
    """Gather velocities from grid, update particle positions, velocities, C, F.
    Returns updated MPMState."""
```

The current signatures pass individual scalars (gravity, dt, damping, inv_dx, etc.) — these are consolidated into `MPMParams`. Boundary conditions are captured in closures built at init time, same as current design.

## JAX–CUDA Interop

**Getting device pointers:** Use `jax.dlpack` or `__cuda_array_interface__` to extract raw GPU pointers from JAX arrays. Avoid `unsafe_buffer_pointer()` as it is an internal API. Preferred approach:

```python
def get_ptr(arr: jax.Array) -> int:
    """Extract raw GPU device pointer from a JAX array."""
    return arr.__cuda_array_interface__['data'][0]
```

**Output arrays:** CUDA kernels write into pre-allocated output buffers. These must be allocated outside JAX's functional semantics — use `cuda.core` device buffers or CuPy arrays, then wrap back into JAX via `jax.dlpack.from_dlpack()` after the kernel completes. Alternatively, allocate via `jnp.zeros()` and accept that writing to the pointer is a side-effect (acceptable here since the array is freshly created and not aliased).

**Stream synchronization:** The `CudaRuntime` creates its own CUDA stream. Two sync points are needed:
1. **Before kernel launch:** `jax.block_until_ready()` on all input arrays to ensure JAX's stream has finished writing them.
2. **After kernel launch:** `stream.sync()` to ensure the CUDA kernel has finished before JAX reads the output arrays.

For tighter integration, we can explore extracting JAX's stream handle and launching on the same stream — but explicit double-sync is correct and simpler for the initial implementation.

## Initialization

```python
# In simulate.py, before the timestep loop:
runtime = None
if cfg.kernel.name.startswith("cuda"):
    from mpm_jax.cuda.runtime import CudaRuntime
    runtime = CudaRuntime()

p2g_fn = get_p2g_fn(cfg, runtime)  # runtime passed to factory, None for JAX
```

`CudaRuntime` is created once. The `get_p2g_fn` dispatcher passes it to CUDA factory functions:

```python
def get_p2g_fn(cfg, runtime=None):
    if cfg.kernel.name == "jax":
        return make_jax_p2g(cfg)
    elif cfg.kernel.name == "cuda_naive":
        return make_cuda_naive_p2g(cfg, runtime)
    elif cfg.kernel.name == "cuda_warp":
        return make_cuda_warp_p2g(cfg, runtime)
```

## Error Handling

- **Compilation failure:** `CudaRuntime.compile_kernel()` catches NVRTC errors and raises a clear message with the kernel source path and compiler output. Falls back gracefully — if CUDA is unavailable, `get_p2g_fn` raises an error at init time (not mid-simulation).
- **Kernel source paths:** Resolved relative to the `mpm_jax/cuda/` package directory using `Path(__file__).parent`, not relative to CWD (which Hydra changes).
- **No GPU fallback:** If `cfg.kernel.name` is a CUDA variant but no GPU is available, fail fast at `CudaRuntime()` init with a clear error. No silent fallback to JAX.

## Timestep Loop

```python
p2g_fn = get_p2g_fn(cfg, runtime)

step_timings = []
for frame in range(num_frames):
    for step in range(steps_per_frame):
        t0 = time.perf_counter()
        grid_mv, grid_m = p2g_fn(state, params)
        jax.block_until_ready((grid_mv, grid_m))
        t1 = time.perf_counter()
        grid_v = grid_update(grid_mv, grid_m, params)
        jax.block_until_ready(grid_v)
        t2 = time.perf_counter()
        state = g2p(state, grid_v, params)
        jax.block_until_ready(state.x)
        t3 = time.perf_counter()

        step_timings.append({
            "p2g_ms": (t1 - t0) * 1000,
            "grid_update_ms": (t2 - t1) * 1000,
            "g2p_ms": (t3 - t2) * 1000,
            "step_ms": (t3 - t0) * 1000,
        })
```

## Wandb Logging

```python
wandb.init(project="mpm-cuda", config=OmegaConf.to_container(cfg))

# Per-frame time series (aggregate substeps per frame to avoid excessive logging)
for frame_timings in frame_timing_list:
    wandb.log(frame_timings)

# Summary
wandb.summary.update({
    "mean_p2g_ms": np.mean([t["p2g_ms"] for t in step_timings]),
    "mean_grid_update_ms": np.mean([t["grid_update_ms"] for t in step_timings]),
    "mean_g2p_ms": np.mean([t["g2p_ms"] for t in step_timings]),
    "mean_step_ms": np.mean([t["step_ms"] for t in step_timings]),
    "total_steps": total_steps,
    "steps_per_sec": total_steps / total_time,
    "n_particles": cfg.sim.n_particles,
    "kernel": cfg.kernel.name,
})
```

## Hydra Config

```yaml
# conf/kernel/jax.yaml
name: jax

# conf/kernel/cuda_naive.yaml
name: cuda_naive
block_size: 256

# conf/kernel/cuda_warp.yaml
name: cuda_warp
block_size: 256
```

Material, sim configs unchanged. Profile configs removed (timing is now built-in).

## What's Removed

- `lax.scan` / `build_jit_step()` / `build_jit_frame()` — replaced by Python loop
- JAX FFI integration (`jax.ffi.register_ffi_target`) — replaced by `cuda.core`
- Shared-memory kernel (v4 / `p2g_scatter_smem.cu`) — dropped
- `StageTimer` class — replaced by simple `time.perf_counter()` diffs
- Nsys/ncu profiler integration in simulate.py — can still be used externally

## What's Kept

- All constitutive models and boundary conditions
- Hydra configuration system
- Wandb integration (simplified)
- Test suite (updated for new module structure)
- GIF rendering for non-benchmark runs

## Testing

- **Numerical correctness:** Compare CUDA P2G output against JAX baseline for a small particle set (e.g., 100 particles, 1 step). Assert `grid_mv` and `grid_m` match within tolerance (`atol=1e-5` for float32).
- **Module tests:** Each split module (`grid_update.py`, `g2p.py`, `p2g/jax.py`) gets unit tests verifying the same behavior as the current monolithic solver.
- **Integration test:** Full 10-frame jelly simulation with each kernel variant, verify particles fall under gravity and all values are finite.
- **No-GPU CI:** Tests for JAX baseline run without a GPU. CUDA kernel tests are skipped via `pytest.mark.skipif` when no GPU is detected.
