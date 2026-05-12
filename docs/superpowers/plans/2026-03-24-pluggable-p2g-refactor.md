# Pluggable P2G Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor MPM-CudaJax into a clean Python timestep loop with pluggable P2G implementations (JAX baseline + CUDA variants via `cuda.core`), per-stage timing, and wandb logging.

**Architecture:** Break the monolithic `solver.py` into focused modules (`state.py`, `grid_update.py`, `g2p.py`, `p2g/`). Each P2G variant implements `(state, params) -> (grid_mv, grid_m)`. CUDA kernels compile at runtime via `cuda.core`. The timestep loop lives in `simulate.py` with `time.perf_counter()` between stages, all logged to wandb.

**Tech Stack:** JAX, cuda-core (NVIDIA cuda-python), Hydra, wandb, pytest

**Spec:** `docs/superpowers/specs/2026-03-24-refactor-pluggable-p2g-design.md`

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `mpm_jax/state.py` | `MPMState`, `MPMParams` NamedTuples + `make_params()` |
| `mpm_jax/grid_update.py` | `grid_update(grid_mv, grid_m, params) -> grid_v` (JIT'd) |
| `mpm_jax/g2p.py` | `g2p(state, grid_v, params) -> MPMState` (JIT'd) |
| `mpm_jax/p2g/__init__.py` | `get_p2g_fn(cfg, runtime=None) -> callable` dispatcher |
| `mpm_jax/p2g/jax.py` | Pure JAX P2G baseline |
| `mpm_jax/p2g/cuda_naive.py` | CUDA naive scatter P2G wrapper |
| `mpm_jax/p2g/cuda_warp.py` | CUDA warp-reduced scatter P2G wrapper |
| `mpm_jax/cuda/runtime.py` | `CudaRuntime` class (cuda.core compile/launch) |
| `mpm_jax/cuda/kernels/p2g_compute.cuh` | Shared CUDA device function for P2G compute |
| `tests/test_state.py` | Tests for state module |
| `tests/test_grid_update.py` | Tests for grid_update |
| `tests/test_g2p.py` | Tests for g2p |
| `tests/test_p2g_jax.py` | Tests for JAX P2G |
| `tests/test_p2g_cuda.py` | Tests for CUDA P2G variants |

### Modified Files
| File | Changes |
|------|---------|
| `simulate.py` | Rewrite: Python timestep loop, wandb logging, remove profiler wrappers |
| `conf/config.yaml` | Update kernel defaults |
| `conf/kernel/*.yaml` | Rename: `cuda_v1` → `cuda_naive`, `cuda_v3` → `cuda_warp`, drop `cuda_v4` |
| `pyproject.toml` | Add `cuda-core` optional dependency |

### Deleted Files
| File | Reason |
|------|--------|
| `mpm_jax/solver.py` | Split into state.py, grid_update.py, g2p.py, p2g/ |
| `mpm_jax/cuda/p2g_cuda.py` | Replaced by cuda/runtime.py + p2g/*.py |
| `mpm_jax/cuda/p2g_custom_op.py` | Old PyCUDA wrapper, no longer needed |
| `mpm_jax/cuda/p2g_kernel.cu` | Old float64 benchmark kernel |
| `mpm_jax/cuda/kernels/p2g_scatter_smem.cu` | v4 dropped per spec |
| `mpm_jax/cuda/kernels/p2g_fused.cu` | Superseded by new fused design |
| `conf/kernel/cuda_v1.yaml` | Renamed to cuda_naive.yaml |
| `conf/kernel/cuda_v3.yaml` | Renamed to cuda_warp.yaml |
| `conf/kernel/cuda_v2.yaml` | Dropped (fused WIP, superseded) |
| `conf/kernel/cuda_v4.yaml` | Dropped |
| `conf/profile/*.yaml` | Profiler integration removed from simulate.py |
| `conf/sweep_*.yaml` | Will be recreated if needed |
| `tests/test_solver.py` | Replaced by per-module tests |
| `tests/test_integration.py` | Rewritten for new structure |

---

### Task 1: Create dev branch and add cuda-core dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Create dev branch**

```bash
cd MPM-CudaJax
git checkout -b dev
```

- [ ] **Step 2: Add cuda-core to optional dependencies in pyproject.toml**

In `pyproject.toml`, add `cuda-core` to a new `cuda` extra and to `jax-cuda`:

```toml
[project.optional-dependencies]
jax = ["jax>=0.4.20", "jaxlib>=0.4.20"]
jax-cuda = ["jax[cuda12]"]
cuda = ["cuda-core"]
all = ["jax[cuda12]", "cuda-core"]
```

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: create dev branch, add cuda-core dependency"
```

---

### Task 2: Extract state.py — MPMState, MPMParams, make_params

**Files:**
- Create: `mpm_jax/state.py`
- Create: `tests/test_state.py`
- Reference: `mpm_jax/solver.py:1-54` (current definitions)

- [ ] **Step 1: Write failing test**

Create `tests/test_state.py`:

```python
import jax.numpy as jnp
import numpy as np
from mpm_jax.state import MPMState, MPMParams, make_params


def test_mpm_state_is_namedtuple():
    N = 10
    state = MPMState(
        x=jnp.zeros((N, 3)),
        v=jnp.zeros((N, 3)),
        C=jnp.zeros((N, 3, 3)),
        F=jnp.tile(jnp.eye(3), (N, 1, 1)),
    )
    assert state.x.shape == (N, 3)
    assert state.F.shape == (N, 3, 3)


def test_make_params():
    p = make_params(n_particles=100, num_grids=25)
    assert p.dx == 1.0 / 25
    assert p.inv_dx == 25.0
    assert p.vol > 0
    assert p.p_mass > 0
    assert p.n_particles == 100
    # clip_bound is scaled by dx
    assert p.clip_bound == 0.5 * p.dx


def test_make_params_volume_formula():
    """vol = prod(size) / n_particles — matches original solver."""
    p = make_params(n_particles=100, size=[1.0, 1.0, 1.0])
    assert np.isclose(p.vol, 1.0 / 100)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra jax --with pytest python -m pytest tests/test_state.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mpm_jax.state'`

- [ ] **Step 3: Write mpm_jax/state.py**

Extract from `mpm_jax/solver.py` lines 1-54 **exactly as-is**, preserving the volume formula and argument order:

```python
"""MPM state and parameter definitions."""
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np

class MPMState(NamedTuple):
    x: jax.Array      # (N, 3) positions
    v: jax.Array      # (N, 3) velocities
    C: jax.Array      # (N, 3, 3) APIC affine matrix
    F: jax.Array      # (N, 3, 3) deformation gradient

class MPMParams(NamedTuple):
    num_grids: int
    dt: float
    gravity: jax.Array
    dx: float
    inv_dx: float
    clip_bound: float
    damping: float
    vol: float
    p_mass: float
    n_particles: int

# 27 stencil offsets for 3x3x3 neighborhood
OFFSET_27 = jnp.array(
    [[i, j, k] for i in range(3) for j in range(3) for k in range(3)],
    dtype=jnp.float32,
)  # (27, 3)

def make_params(
    n_particles: int,
    num_grids: int = 25,
    dt: float = 3e-4,
    gravity: list = [0.0, 0.0, -9.8],
    rho: float = 1000.0,
    clip_bound: float = 0.5,
    damping: float = 1.0,
    center: list = [0.5, 0.5, 0.5],
    size: list = [1.0, 1.0, 1.0],
) -> MPMParams:
    dx = 1.0 / num_grids
    vol = float(np.prod(size)) / n_particles
    return MPMParams(
        num_grids=num_grids,
        dt=dt,
        gravity=jnp.array(gravity, dtype=jnp.float32),
        dx=dx,
        inv_dx=float(num_grids),
        clip_bound=clip_bound * dx,
        damping=damping,
        vol=vol,
        p_mass=rho * vol,
        n_particles=n_particles,
    )
```

**IMPORTANT:** `n_particles` is the first positional arg, `vol = prod(size) / n_particles`, and `clip_bound = clip_bound * dx`. These match the original `solver.py` exactly.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra jax --with pytest python -m pytest tests/test_state.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mpm_jax/state.py tests/test_state.py
git commit -m "refactor: extract state.py with MPMState, MPMParams, make_params"
```

---

### Task 3: Extract grid_update.py

**Files:**
- Create: `mpm_jax/grid_update.py`
- Create: `tests/test_grid_update.py`
- Reference: `mpm_jax/solver.py:212-221` (current grid_update)

- [ ] **Step 1: Write failing test**

Create `tests/test_grid_update.py`:

```python
import jax.numpy as jnp
from mpm_jax.state import make_params
from mpm_jax.grid_update import grid_update


def test_grid_update_normalizes_momentum():
    """momentum / mass = velocity, then apply gravity."""
    params = make_params(num_grids=4, dt=1e-3)
    G = 4
    grid_mv = jnp.ones((G**3, 3))
    grid_m = jnp.full((G**3,), 2.0)

    grid_v = grid_update(grid_mv, grid_m, params)

    # v = mv/m + gravity*dt = 0.5 + gravity*dt
    expected_z = 0.5 + params.gravity[2] * params.dt
    assert grid_v.shape == (G**3, 3)
    assert jnp.allclose(grid_v[:, 2], expected_z, atol=1e-5)


def test_grid_update_zero_mass_gives_zero_velocity():
    params = make_params(num_grids=4, dt=1e-3)
    G = 4
    grid_mv = jnp.ones((G**3, 3))
    grid_m = jnp.zeros((G**3,))

    grid_v = grid_update(grid_mv, grid_m, params)
    assert jnp.allclose(grid_v, 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra jax --with pytest python -m pytest tests/test_grid_update.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mpm_jax.grid_update'`

- [ ] **Step 3: Write mpm_jax/grid_update.py**

Extract from `solver.py` lines 212-221. Keep the exact same logic:

```python
"""Grid update: normalize momentum by mass, apply gravity and damping."""
import jax
import jax.numpy as jnp
from mpm_jax.state import MPMParams


@jax.jit
def grid_update(
    grid_mv: jnp.ndarray,
    grid_m: jnp.ndarray,
    params: MPMParams,
) -> jnp.ndarray:
    """Normalize momentum by mass, apply gravity and damping.

    Matches solver.py:212-221 logic exactly:
    - Divide momentum by mass where mass > 1e-15
    - Apply damping factor
    - Add gravity * dt

    Args:
        grid_mv: (num_grids³, 3) momentum
        grid_m: (num_grids³,) mass
        params: MPMParams

    Returns:
        grid_v: (num_grids³, 3) velocity (used as grid_mv downstream for BCs)
    """
    valid = grid_m > 1e-15
    grid_v = jnp.where(valid[:, None], grid_mv / grid_m[:, None], grid_mv)
    grid_v = params.damping * (grid_v + params.dt * params.gravity)
    return grid_v
```

**IMPORTANT:** The original grid_update does NOT zero out invalid cells — it keeps the original momentum. The damping+gravity is applied to ALL cells. This must match exactly for numerical equivalence.

Note: Boundary conditions (`post_grid_fn(grid_v, grid_m, time)`) are applied AFTER grid_update in the timestep loop (Task 10). The post_grid_fn takes `(grid_mv, grid_m, time)` where `grid_mv` is actually the velocity at this point.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra jax --with pytest python -m pytest tests/test_grid_update.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mpm_jax/grid_update.py tests/test_grid_update.py
git commit -m "refactor: extract grid_update.py with simplified params signature"
```

---

### Task 4: Extract g2p.py

**Files:**
- Create: `mpm_jax/g2p.py`
- Create: `tests/test_g2p.py`
- Reference: `mpm_jax/solver.py:135-162` (`_single_particle_g2p`), `solver.py:224-229` (`g2p`)

- [ ] **Step 1: Write failing test**

Create `tests/test_g2p.py`:

```python
import jax.numpy as jnp
from mpm_jax.state import MPMState, make_params
from mpm_jax.g2p import g2p


def test_g2p_returns_mpm_state():
    N = 10
    params = make_params(num_grids=4, n_particles=N)
    G = params.num_grids
    state = MPMState(
        x=jnp.full((N, 3), 0.5),
        v=jnp.zeros((N, 3)),
        C=jnp.zeros((N, 3, 3)),
        F=jnp.tile(jnp.eye(3), (N, 1, 1)),
    )
    grid_v = jnp.zeros((G**3, 3))

    new_state = g2p(state, grid_v, params)

    assert isinstance(new_state, MPMState)
    assert new_state.x.shape == (N, 3)
    assert new_state.v.shape == (N, 3)
    assert new_state.C.shape == (N, 3, 3)
    assert new_state.F.shape == (N, 3, 3)


def test_g2p_updates_position():
    """With nonzero grid velocity, particles should move."""
    N = 5
    params = make_params(num_grids=4, n_particles=N, dt=1e-2)
    G = params.num_grids
    state = MPMState(
        x=jnp.full((N, 3), 0.5),
        v=jnp.zeros((N, 3)),
        C=jnp.zeros((N, 3, 3)),
        F=jnp.tile(jnp.eye(3), (N, 1, 1)),
    )
    # Uniform downward grid velocity
    grid_v = jnp.zeros((G**3, 3)).at[:, 2].set(-1.0)

    new_state = g2p(state, grid_v, params)

    # Particles should have moved down
    assert jnp.all(new_state.x[:, 2] < 0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra jax --with pytest python -m pytest tests/test_g2p.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mpm_jax.g2p'`

- [ ] **Step 3: Write mpm_jax/g2p.py**

Extract from `solver.py:61-107` (weights) and `solver.py:135-162` (g2p). **Key change:** the current g2p receives pre-computed weights from `step()`. In the new design, g2p recomputes weights internally since they are no longer shared with P2G. This is cheap (B-spline evaluation).

```python
"""Grid-to-particle transfer: gather velocities, update particle state."""
import jax
import jax.numpy as jnp
from mpm_jax.state import MPMState, MPMParams, OFFSET_27


def _single_particle_g2p(grid_v, F_p, x_p, dt, inv_dx, clip_bound):
    """G2P for one particle. Recomputes weights internally.

    Matches solver.py:135-162 logic, inlining weight computation
    from solver.py:61-107.
    """
    # Recompute B-spline weights
    px = x_p * inv_dx
    base = jnp.floor(px - 0.5).astype(int)
    fx = px - base.astype(jnp.float32)
    dx = 1.0 / inv_dx
    num_grids = jnp.int32(inv_dx)

    w = jnp.stack([
        0.5 * (1.5 - fx) ** 2,
        0.75 - (fx - 1.0) ** 2,
        0.5 * (fx - 0.5) ** 2,
    ])  # (3, 3)

    dw = jnp.stack([
        fx - 1.5,
        -2.0 * (fx - 1.0),
        fx - 0.5,
    ])  # (3, 3)

    offsets = OFFSET_27.astype(int)
    weight = w[offsets[:, 0], 0] * w[offsets[:, 1], 1] * w[offsets[:, 2], 2]

    dweight = inv_dx * jnp.stack([
        dw[offsets[:, 0], 0] *  w[offsets[:, 1], 1] *  w[offsets[:, 2], 2],
         w[offsets[:, 0], 0] * dw[offsets[:, 1], 1] *  w[offsets[:, 2], 2],
         w[offsets[:, 0], 0] *  w[offsets[:, 1], 1] * dw[offsets[:, 2], 2],
    ], axis=-1)  # (27, 3)

    dpos = (OFFSET_27 - fx[None, :]) * dx  # (27, 3)

    idx_3d = base[None, :] + offsets
    index = idx_3d[:, 0] * num_grids * num_grids + idx_3d[:, 1] * num_grids + idx_3d[:, 2]
    index = jnp.clip(index, 0, num_grids ** 3 - 1)

    # G2P gather (matches solver.py:154-161 exactly)
    gv = grid_v[index]  # (27, 3)
    new_v = (weight[:, None] * gv).sum(axis=0)
    new_C = 4.0 * inv_dx * inv_dx * (
        weight[:, None, None] * jnp.einsum('ij,ik->ijk', gv, dpos)
    ).sum(axis=0)
    grad_v = jnp.einsum('ij,ik->ijk', gv, dweight).sum(axis=0)

    new_x = jnp.clip(x_p + new_v * dt, clip_bound, 1.0 - clip_bound)
    new_F = jnp.clip(F_p + dt * grad_v @ F_p, -2.0, 2.0)

    return new_x, new_v, new_C, new_F


@jax.jit
def g2p(state: MPMState, grid_v: jnp.ndarray, params: MPMParams) -> MPMState:
    """Grid-to-particle: gather velocities and update particle state."""
    new_x, new_v, new_C, new_F = jax.vmap(
        _single_particle_g2p,
        in_axes=(None, 0, 0, None, None, None),
    )(grid_v, state.F, state.x, params.dt, params.inv_dx, params.clip_bound)

    return MPMState(x=new_x, v=new_v, C=new_C, F=new_F)
```

**IMPORTANT:** `grid_v` is broadcast (None), particle arrays F, x are mapped (0), scalars are broadcast (None). Note `clip_bound` from `make_params` is already scaled by `dx`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra jax --with pytest python -m pytest tests/test_g2p.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mpm_jax/g2p.py tests/test_g2p.py
git commit -m "refactor: extract g2p.py with simplified params signature"
```

---

### Task 5: Create p2g/jax.py — pure JAX P2G baseline

**Files:**
- Create: `mpm_jax/p2g/__init__.py`
- Create: `mpm_jax/p2g/jax.py`
- Create: `tests/test_p2g_jax.py`
- Reference: `mpm_jax/solver.py:61-209` (weights, p2g_compute, p2g_scatter, p2g)
- Reference: `mpm_jax/constitutive.py` (elasticity/plasticity functions)

- [ ] **Step 1: Write failing test**

Create `tests/test_p2g_jax.py`:

```python
import jax.numpy as jnp
from omegaconf import OmegaConf
from mpm_jax.state import MPMState, make_params
from mpm_jax.p2g.jax import make_jax_p2g


def _make_state(N, params):
    return MPMState(
        x=jnp.full((N, 3), 0.5),
        v=jnp.zeros((N, 3)),
        C=jnp.zeros((N, 3, 3)),
        F=jnp.tile(jnp.eye(3), (N, 1, 1)),
    )


def test_jax_p2g_returns_correct_shapes():
    N = 50
    params = make_params(num_grids=8, n_particles=N)
    state = _make_state(N, params)
    cfg = OmegaConf.create({
        "material": {
            "elasticity": {"name": "corotated_elasticity", "E": 2e6, "nu": 0.4},
            "plasticity": {"name": "identity_plasticity"},
        }
    })
    p2g_fn = make_jax_p2g(cfg)

    grid_mv, grid_m = p2g_fn(state, params)

    G = params.num_grids
    assert grid_mv.shape == (G**3, 3)
    assert grid_m.shape == (G**3,)


def test_jax_p2g_nonzero_mass():
    """Particles at center should contribute mass to nearby grid nodes."""
    N = 50
    params = make_params(num_grids=8, n_particles=N)
    state = _make_state(N, params)
    cfg = OmegaConf.create({
        "material": {
            "elasticity": {"name": "corotated_elasticity", "E": 2e6, "nu": 0.4},
            "plasticity": {"name": "identity_plasticity"},
        }
    })
    p2g_fn = make_jax_p2g(cfg)

    grid_mv, grid_m = p2g_fn(state, params)

    assert jnp.any(grid_m > 0)
    assert jnp.all(jnp.isfinite(grid_mv))
    assert jnp.all(jnp.isfinite(grid_m))


def test_jax_p2g_mass_conservation():
    """Total grid mass should equal N * p_mass."""
    N = 100
    params = make_params(num_grids=8, n_particles=N)
    state = _make_state(N, params)
    cfg = OmegaConf.create({
        "material": {
            "elasticity": {"name": "corotated_elasticity", "E": 2e6, "nu": 0.4},
            "plasticity": {"name": "identity_plasticity"},
        }
    })
    p2g_fn = make_jax_p2g(cfg)

    grid_mv, grid_m = p2g_fn(state, params)

    total_mass = jnp.sum(grid_m)
    expected_mass = N * params.p_mass
    assert jnp.allclose(total_mass, expected_mass, rtol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra jax --with pytest python -m pytest tests/test_p2g_jax.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mpm_jax.p2g'`

- [ ] **Step 3: Create mpm_jax/p2g/__init__.py**

```python
"""Pluggable P2G implementations."""
from mpm_jax.p2g.jax import make_jax_p2g


def get_p2g_fn(cfg, runtime=None):
    """Return a P2G function based on config.

    All P2G functions have signature:
        p2g(state: MPMState, params: MPMParams) -> (grid_mv, grid_m)
    """
    kernel_name = cfg.kernel.name
    if kernel_name == "jax":
        return make_jax_p2g(cfg)
    elif kernel_name == "cuda_naive":
        from mpm_jax.p2g.cuda_naive import make_cuda_naive_p2g
        return make_cuda_naive_p2g(cfg, runtime)
    elif kernel_name == "cuda_warp":
        from mpm_jax.p2g.cuda_warp import make_cuda_warp_p2g
        return make_cuda_warp_p2g(cfg, runtime)
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")
```

- [ ] **Step 4: Write mpm_jax/p2g/jax.py**

Port from `solver.py` lines 61-209. The key change: stress and weight computation move inside the P2G function. The constitutive model is captured in the closure via `make_jax_p2g(cfg)`.

```python
"""Pure JAX P2G baseline implementation."""
import jax
import jax.numpy as jnp
from mpm_jax.state import MPMState, MPMParams, OFFSET_27
from mpm_jax.constitutive import get_constitutive


def make_jax_p2g(cfg):
    """Build a JIT-compiled JAX P2G function.

    Captures the constitutive model from cfg.material.
    """
    elasticity_fn = get_constitutive(cfg.material.elasticity)
    plasticity_fn = get_constitutive(cfg.material.plasticity)

    def _single_particle_p2g(x_p, v_p, C_p, F_p, inv_dx, dt, vol, p_mass):
        """Compute 27 grid node contributions for one particle."""
        base = (x_p * inv_dx - 0.5).astype(jnp.int32)
        fx = x_p * inv_dx - base.astype(jnp.float32)

        # Quadratic B-spline weights
        w0 = 0.5 * (1.5 - fx) ** 2
        w1 = 0.75 - (fx - 1.0) ** 2
        w2 = 0.5 * (fx - 0.5) ** 2
        w = jnp.stack([w0, w1, w2], axis=0)  # (3, 3)

        # Stress computation (moved inside P2G)
        stress = elasticity_fn(F_p[None])[0]  # unbatch/rebatch for single particle
        # Note: plasticity applied to F before stress in the full loop
        # For initial implementation, apply plasticity here

        num_grids = jnp.int32(inv_dx)

        # Compute contributions for 27 nodes
        mv_contrib = jnp.zeros((27, 3))
        m_contrib = jnp.zeros(27)
        indices = jnp.zeros(27, dtype=jnp.int32)

        def body_fn(i, carry):
            mv_c, m_c, idx = carry
            offset = OFFSET_27[i]
            node = base + offset
            dpos = (offset.astype(jnp.float32) - fx) / inv_dx

            weight = w[offset[0], 0] * w[offset[1], 1] * w[offset[2], 2]
            gid = node[0] * num_grids * num_grids + node[1] * num_grids + node[2]

            affine = p_mass * C_p
            momentum = p_mass * v_p + affine @ dpos
            mv = weight * (momentum + dt * stress @ dpos * vol)
            m = weight * p_mass

            mv_c = mv_c.at[i].set(mv)
            m_c = m_c.at[i].set(m)
            idx = idx.at[i].set(gid)
            return mv_c, m_c, idx

        mv_contrib, m_contrib, indices = jax.lax.fori_loop(
            0, 27, body_fn, (mv_contrib, m_contrib, indices)
        )
        return mv_contrib, m_contrib, indices

    @jax.jit
    def p2g(state: MPMState, params: MPMParams):
        # Apply plasticity to F
        F = plasticity_fn(state.F)

        # Compute per-particle contributions (vmap)
        mv_all, m_all, idx_all = jax.vmap(
            _single_particle_p2g,
            in_axes=(0, 0, 0, 0, None, None, None, None),
        )(state.x, state.v, state.C, F,
          params.inv_dx, params.dt, params.vol, params.p_mass)

        # Scatter onto grid
        G = params.num_grids
        grid_mv = jnp.zeros((G**3, 3))
        grid_m = jnp.zeros((G**3,))

        # Flatten: (N, 27, 3) -> (N*27, 3), indices: (N, 27) -> (N*27,)
        mv_flat = mv_all.reshape(-1, 3)
        m_flat = m_all.reshape(-1)
        idx_flat = idx_all.reshape(-1)

        grid_mv = grid_mv.at[idx_flat].add(mv_flat)
        grid_m = grid_m.at[idx_flat].add(m_flat)

        return grid_mv, grid_m

    return p2g
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run --extra jax --with pytest python -m pytest tests/test_p2g_jax.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add mpm_jax/p2g/__init__.py mpm_jax/p2g/jax.py tests/test_p2g_jax.py
git commit -m "refactor: create p2g/ package with JAX baseline and dispatcher"
```

---

### Task 6: Integration test — full timestep with new modules

**Files:**
- Create: `tests/test_integration.py` (rewrite)

This test validates that the split modules (p2g/jax, grid_update, g2p) work together correctly for a full simulation.

- [ ] **Step 1: Write integration test**

Create `tests/test_integration.py`:

```python
import jax.numpy as jnp
from omegaconf import OmegaConf
from mpm_jax.state import MPMState, make_params
from mpm_jax.p2g.jax import make_jax_p2g
from mpm_jax.grid_update import grid_update
from mpm_jax.g2p import g2p


def _make_cfg():
    return OmegaConf.create({
        "material": {
            "elasticity": {"name": "corotated_elasticity", "E": 2e6, "nu": 0.4},
            "plasticity": {"name": "identity_plasticity"},
        },
        "kernel": {"name": "jax"},
    })


def test_full_timestep_jax():
    """One full P2G -> grid_update -> G2P cycle produces valid state."""
    N = 100
    params = make_params(n_particles=N, num_grids=10, dt=3e-4)
    state = MPMState(
        x=jnp.full((N, 3), 0.5),
        v=jnp.zeros((N, 3)),
        C=jnp.zeros((N, 3, 3)),
        F=jnp.tile(jnp.eye(3), (N, 1, 1)),
    )
    cfg = _make_cfg()
    p2g_fn = make_jax_p2g(cfg)

    grid_mv, grid_m = p2g_fn(state, params)
    grid_v = grid_update(grid_mv, grid_m, params)
    new_state = g2p(state, grid_v, params)

    assert jnp.all(jnp.isfinite(new_state.x))
    assert jnp.all(jnp.isfinite(new_state.v))
    assert jnp.all(jnp.isfinite(new_state.F))


def test_jelly_10_frames():
    """10 frames of jelly simulation: particles should fall under gravity."""
    N = 200
    params = make_params(n_particles=N, num_grids=15, dt=3e-4)
    state = MPMState(
        x=jnp.full((N, 3), 0.5),
        v=jnp.zeros((N, 3)),
        C=jnp.zeros((N, 3, 3)),
        F=jnp.tile(jnp.eye(3), (N, 1, 1)),
    )
    cfg = _make_cfg()
    p2g_fn = make_jax_p2g(cfg)

    initial_z = state.x[:, 2].mean()

    for frame in range(10):
        for step in range(10):
            grid_mv, grid_m = p2g_fn(state, params)
            grid_v = grid_update(grid_mv, grid_m, params)
            state = g2p(state, grid_v, params)

    # Particles should have fallen
    final_z = state.x[:, 2].mean()
    assert final_z < initial_z
    assert jnp.all(jnp.isfinite(state.x))
    assert jnp.all(jnp.isfinite(state.v))
```

- [ ] **Step 2: Run test**

Run: `uv run --extra jax --with pytest python -m pytest tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for split module timestep loop"
```

---

### Task 7: Create cuda/runtime.py — cuda.core wrapper

**Files:**
- Create: `mpm_jax/cuda/runtime.py`

- [ ] **Step 1: Write mpm_jax/cuda/runtime.py**

```python
"""CUDA runtime: kernel compilation and launch via cuda.core."""
from pathlib import Path

KERNELS_DIR = Path(__file__).parent / "kernels"


def get_ptr(arr) -> int:
    """Extract raw GPU device pointer from a JAX array."""
    return arr.__cuda_array_interface__['data'][0]


class CudaRuntime:
    """Manages cuda.core device, stream, and kernel compilation cache."""

    def __init__(self):
        try:
            from cuda.core import Device
        except ImportError:
            raise ImportError(
                "cuda-core is required for CUDA kernels. "
                "Install with: pip install cuda-core"
            )

        self.dev = Device()
        self.dev.set_current()
        self.stream = self.dev.create_stream()
        self._cache = {}

    def compile_kernel(self, source_path: str, kernel_name: str):
        """Compile a .cu file and extract a kernel by name.

        Args:
            source_path: Path to .cu file, relative to kernels/ directory
            kernel_name: Name of the __global__ function to extract

        Returns:
            Compiled kernel handle
        """
        if kernel_name in self._cache:
            return self._cache[kernel_name]

        from cuda.core import Program, ProgramOptions

        cu_path = KERNELS_DIR / source_path
        if not cu_path.exists():
            raise FileNotFoundError(f"Kernel source not found: {cu_path}")

        code = cu_path.read_text()
        try:
            prog = Program(code, code_type="c++", options=ProgramOptions(
                std="c++17",
                arch=f"sm_{self.dev.arch}",
            ))
            mod = prog.compile("cubin")
        except Exception as e:
            raise RuntimeError(
                f"Failed to compile {cu_path}: {e}"
            ) from e

        kernel = mod.get_kernel(kernel_name)
        self._cache[kernel_name] = kernel
        return kernel

    def launch(self, kernel, grid, block, *args):
        """Launch a kernel on the runtime's stream and synchronize.

        Args:
            kernel: Compiled kernel handle
            grid: Grid dimensions (int or tuple)
            block: Block dimensions (int or tuple)
            *args: Kernel arguments (ints for pointers, scalars for values)
        """
        from cuda.core import LaunchConfig, launch

        config = LaunchConfig(grid=grid, block=block)
        launch(self.stream, config, kernel, *args)
        self.stream.sync()
```

- [ ] **Step 2: Commit**

```bash
git add mpm_jax/cuda/runtime.py
git commit -m "feat: add cuda.core runtime wrapper for kernel compilation and launch"
```

---

### Task 8: Port CUDA kernels — shared compute header + scatter variants

**Files:**
- Create: `mpm_jax/cuda/kernels/p2g_compute.cuh`
- Modify: `mpm_jax/cuda/kernels/p2g_scatter_naive.cu` (rename from p2g_scatter.cu, rewrite)
- Modify: `mpm_jax/cuda/kernels/p2g_scatter_warp.cu` (rewrite)

This task ports the existing CUDA kernels to the new fused design where each kernel does compute+scatter. The XLA FFI wrapper code is removed; kernels become standalone `extern "C"` functions callable via `cuda.core`.

- [ ] **Step 1: Write p2g_compute.cuh**

Create `mpm_jax/cuda/kernels/p2g_compute.cuh`. This header contains:
- `ParticleContrib` struct (27 grid node contributions)
- B-spline weight computation
- 3x3 SVD (simplified Jacobi rotations for symmetric eigendecomposition)
- Corotated elasticity stress computation
- APIC momentum assembly

```cuda
#pragma once
#include <math.h>

#define STENCIL 27

struct ParticleContrib {
    float mv[3];
    float m;
    int grid_idx;
};

// --- 3x3 SVD helpers (Jacobi rotations) ---
// Reference: McAdams et al. 2011, "Computing the Singular Value Decomposition
// of 3x3 matrices with minimal branching and elementary floating point operations"
//
// Implementation note: This is a simplified version suitable for MPM where F
// is close to a rotation. For production use, consider a more robust SVD.

__device__ void svd3x3(
    const float F[9],  // column-major 3x3
    float U[9], float S[3], float V[9]
);

// Full implementation of svd3x3 to be written here.
// For initial bring-up, use a simplified polar decomposition:
// F = R * S where R is the rotation part.

__device__ void p2g_compute(
    float x0, float x1, float x2,          // particle position
    float v0, float v1, float v2,           // particle velocity
    const float* C,                          // (3,3) APIC matrix, row-major
    const float* F,                          // (3,3) deformation gradient, row-major
    float dt, float vol, float p_mass,
    float inv_dx, int num_grids,
    float mu_0, float lambda_0,
    ParticleContrib out[STENCIL]
) {
    // 1. Quadratic B-spline weights
    float fx[3], base_f[3];
    int base[3];
    base_f[0] = x0 * inv_dx - 0.5f;
    base_f[1] = x1 * inv_dx - 0.5f;
    base_f[2] = x2 * inv_dx - 0.5f;
    base[0] = (int)base_f[0];
    base[1] = (int)base_f[1];
    base[2] = (int)base_f[2];
    fx[0] = x0 * inv_dx - (float)base[0];
    fx[1] = x1 * inv_dx - (float)base[1];
    fx[2] = x2 * inv_dx - (float)base[2];

    // Weights: w[axis][node] for 3 nodes per axis
    float w[3][3];
    for (int d = 0; d < 3; d++) {
        w[d][0] = 0.5f * (1.5f - fx[d]) * (1.5f - fx[d]);
        w[d][1] = 0.75f - (fx[d] - 1.0f) * (fx[d] - 1.0f);
        w[d][2] = 0.5f * (fx[d] - 0.5f) * (fx[d] - 0.5f);
    }

    // 2. Stress via corotated elasticity (SVD of F)
    // stress = U @ diag(2*mu*(S-1)*S + lambda*(J-1)*J) @ V^T (Kirchhoff)
    float F_col[9]; // column-major for SVD
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            F_col[j * 3 + i] = F[i * 3 + j]; // transpose row->col major

    float U[9], S_vals[3], V[9];
    svd3x3(F_col, U, S_vals, V);

    float J = S_vals[0] * S_vals[1] * S_vals[2];
    float stress[9] = {0};

    // Kirchhoff stress = 2*mu*(F - R) @ F^T + lambda*(J-1)*J*I
    // Simplified: work in rotated frame
    float diag[3];
    for (int i = 0; i < 3; i++) {
        diag[i] = 2.0f * mu_0 * (S_vals[i] - 1.0f) * S_vals[i]
                 + lambda_0 * (J - 1.0f) * J;
    }

    // stress = U @ diag @ V^T (row-major)
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            float s = 0;
            for (int k = 0; k < 3; k++)
                s += U[i * 3 + k] * diag[k] * V[j * 3 + k]; // V^T
            stress[i * 3 + j] = s;
        }

    // 3. Compute 27 node contributions
    int idx = 0;
    for (int di = 0; di < 3; di++)
    for (int dj = 0; dj < 3; dj++)
    for (int dk = 0; dk < 3; dk++) {
        int ni = base[0] + di;
        int nj = base[1] + dj;
        int nk = base[2] + dk;

        float dpos[3] = {
            ((float)di - fx[0]) / inv_dx,
            ((float)dj - fx[1]) / inv_dx,
            ((float)dk - fx[2]) / inv_dx,
        };

        float weight = w[0][di] * w[1][dj] * w[2][dk];
        int gid = ni * num_grids * num_grids + nj * num_grids + nk;

        // APIC momentum: p_mass * v + p_mass * C @ dpos + dt * stress @ dpos * vol
        float mv[3];
        for (int d = 0; d < 3; d++) {
            float C_dpos = 0;
            float stress_dpos = 0;
            for (int k = 0; k < 3; k++) {
                C_dpos += C[d * 3 + k] * dpos[k];
                stress_dpos += stress[d * 3 + k] * dpos[k];
            }
            mv[d] = weight * (p_mass * (d == 0 ? v0 : d == 1 ? v1 : v2)
                             + p_mass * C_dpos
                             + dt * stress_dpos * vol);
        }

        out[idx].mv[0] = mv[0];
        out[idx].mv[1] = mv[1];
        out[idx].mv[2] = mv[2];
        out[idx].m = weight * p_mass;
        out[idx].grid_idx = gid;
        idx++;
    }
}
```

**CRITICAL: The `svd3x3` function is declared but not implemented above.** This is ~100-150 lines of non-trivial CUDA code. Use the Jacobi rotation approach from McAdams et al. 2011 ("Computing the SVD of 3x3 matrices with minimal branching"). A complete, tested CUDA implementation is available in the Taichi MPM codebase and in NVIDIA's cuda-samples. The implementor MUST:
1. Find or write a complete `svd3x3` implementation
2. Validate it against JAX's `jnp.linalg.svd` on random 3x3 matrices (atol=1e-4 for float32)
3. Include it directly in `p2g_compute.cuh` before the `p2g_compute` function

This is the most time-consuming part of the CUDA kernel work. Budget accordingly.

- [ ] **Step 2: Write p2g_scatter_naive.cu**

Create `mpm_jax/cuda/kernels/p2g_scatter_naive.cu`:

```cuda
#include "p2g_compute.cuh"

extern "C" __global__ void p2g_scatter_naive(
    const float* __restrict__ x,      // (N, 3)
    const float* __restrict__ v,      // (N, 3)
    const float* __restrict__ C,      // (N, 3, 3)
    const float* __restrict__ F,      // (N, 3, 3)
    float* __restrict__ grid_mv,      // (G^3, 3)
    float* __restrict__ grid_m,       // (G^3,)
    float dt, float vol, float p_mass,
    float inv_dx, int num_grids,
    float mu_0, float lambda_0,
    int n_particles
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= n_particles) return;

    ParticleContrib contrib[STENCIL];
    p2g_compute(
        x[pid * 3 + 0], x[pid * 3 + 1], x[pid * 3 + 2],
        v[pid * 3 + 0], v[pid * 3 + 1], v[pid * 3 + 2],
        &C[pid * 9], &F[pid * 9],
        dt, vol, p_mass, inv_dx, num_grids,
        mu_0, lambda_0,
        contrib
    );

    for (int i = 0; i < STENCIL; i++) {
        atomicAdd(&grid_mv[contrib[i].grid_idx * 3 + 0], contrib[i].mv[0]);
        atomicAdd(&grid_mv[contrib[i].grid_idx * 3 + 1], contrib[i].mv[1]);
        atomicAdd(&grid_mv[contrib[i].grid_idx * 3 + 2], contrib[i].mv[2]);
        atomicAdd(&grid_m[contrib[i].grid_idx], contrib[i].m);
    }
}
```

- [ ] **Step 3: Write p2g_scatter_warp.cu**

Create `mpm_jax/cuda/kernels/p2g_scatter_warp.cu`:

```cuda
#include "p2g_compute.cuh"

// Reduce val across lanes in `mask`. Uses popcount-based iteration
// to handle non-contiguous peer groups correctly.
__device__ float warp_reduce_masked(unsigned mask, float val) {
    // Iterate over active lanes in the mask using ballot-style reduction
    unsigned remaining = mask;
    while (__popc(remaining) > 1) {
        // Find the last set bit
        int last = 31 - __clz(remaining);
        // Clear it
        remaining &= ~(1u << last);
        // The first set bit receives the value
        int first = __ffs(remaining) - 1;
        float other = __shfl_sync(mask, val, last);
        if ((threadIdx.x % 32) == first) val += other;
    }
    return val;
}
// NOTE: The existing p2g_scatter_warp.cu in the codebase has a tested
// warp reduction. Port that implementation rather than writing from scratch.
// The above is illustrative; verify against the original.

extern "C" __global__ void p2g_scatter_warp(
    const float* __restrict__ x,
    const float* __restrict__ v,
    const float* __restrict__ C,
    const float* __restrict__ F,
    float* __restrict__ grid_mv,
    float* __restrict__ grid_m,
    float dt, float vol, float p_mass,
    float inv_dx, int num_grids,
    float mu_0, float lambda_0,
    int n_particles
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= n_particles) return;

    ParticleContrib contrib[STENCIL];
    p2g_compute(
        x[pid * 3 + 0], x[pid * 3 + 1], x[pid * 3 + 2],
        v[pid * 3 + 0], v[pid * 3 + 1], v[pid * 3 + 2],
        &C[pid * 9], &F[pid * 9],
        dt, vol, p_mass, inv_dx, num_grids,
        mu_0, lambda_0,
        contrib
    );

    unsigned lane = threadIdx.x % 32;

    for (int i = 0; i < STENCIL; i++) {
        int gid = contrib[i].grid_idx;
        unsigned peers = __match_any_sync(0xFFFFFFFF, gid);
        unsigned leader = __ffs(peers) - 1;

        float mv0 = warp_reduce_masked(peers, contrib[i].mv[0]);
        float mv1 = warp_reduce_masked(peers, contrib[i].mv[1]);
        float mv2 = warp_reduce_masked(peers, contrib[i].mv[2]);
        float m   = warp_reduce_masked(peers, contrib[i].m);

        if (lane == leader) {
            atomicAdd(&grid_mv[gid * 3 + 0], mv0);
            atomicAdd(&grid_mv[gid * 3 + 1], mv1);
            atomicAdd(&grid_mv[gid * 3 + 2], mv2);
            atomicAdd(&grid_m[gid], m);
        }
    }
}
```

- [ ] **Step 4: Remove old kernel files**

```bash
rm mpm_jax/cuda/kernels/p2g_scatter.cu         # replaced by p2g_scatter_naive.cu
rm mpm_jax/cuda/kernels/p2g_scatter_smem.cu     # v4 dropped
rm mpm_jax/cuda/kernels/p2g_fused.cu            # superseded
rm mpm_jax/cuda/p2g_kernel.cu                   # old float64 benchmark
rm mpm_jax/cuda/p2g_custom_op.py                # old PyCUDA wrapper
```

- [ ] **Step 5: Commit**

```bash
git add mpm_jax/cuda/kernels/p2g_compute.cuh \
        mpm_jax/cuda/kernels/p2g_scatter_naive.cu \
        mpm_jax/cuda/kernels/p2g_scatter_warp.cu
git rm mpm_jax/cuda/kernels/p2g_scatter.cu \
       mpm_jax/cuda/kernels/p2g_scatter_smem.cu \
       mpm_jax/cuda/kernels/p2g_fused.cu \
       mpm_jax/cuda/p2g_kernel.cu \
       mpm_jax/cuda/p2g_custom_op.py
git commit -m "feat: port CUDA kernels to standalone fused design with shared compute header"
```

---

### Task 9: Create P2G CUDA wrappers

**Files:**
- Create: `mpm_jax/p2g/cuda_naive.py`
- Create: `mpm_jax/p2g/cuda_warp.py`
- Create: `tests/test_p2g_cuda.py`

- [ ] **Step 1: Write mpm_jax/p2g/cuda_naive.py**

```python
"""CUDA P2G with naive atomicAdd scatter."""
import math
import jax
import jax.numpy as jnp
from mpm_jax.state import MPMState, MPMParams
from mpm_jax.cuda.runtime import CudaRuntime, get_ptr
from mpm_jax.constitutive import get_constitutive


def make_cuda_naive_p2g(cfg, runtime: CudaRuntime):
    """Compile naive scatter kernel and return P2G closure."""
    kernel = runtime.compile_kernel("p2g_scatter_naive.cu", "p2g_scatter_naive")

    # Get material parameters for CUDA kernel
    mat = cfg.material.elasticity
    E, nu = float(mat.E), float(mat.nu)
    mu_0 = E / (2.0 * (1.0 + nu))
    lambda_0 = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    block_size = cfg.kernel.get("block_size", 256)

    def p2g(state: MPMState, params: MPMParams):
        N = params.n_particles
        G = params.num_grids

        # Ensure JAX has finished writing inputs
        jax.block_until_ready((state.x, state.v, state.C, state.F))

        # Allocate output on GPU
        grid_mv = jnp.zeros((G ** 3, 3), dtype=jnp.float32)
        grid_m = jnp.zeros((G ** 3,), dtype=jnp.float32)

        grid_dim = math.ceil(N / block_size)

        runtime.launch(
            kernel, grid=grid_dim, block=block_size,
            get_ptr(state.x), get_ptr(state.v),
            get_ptr(state.C), get_ptr(state.F),
            get_ptr(grid_mv), get_ptr(grid_m),
            params.dt, params.vol, params.p_mass,
            params.inv_dx, params.num_grids,
            mu_0, lambda_0,
            N,
        )

        return grid_mv, grid_m

    return p2g
```

- [ ] **Step 2: Write mpm_jax/p2g/cuda_warp.py**

```python
"""CUDA P2G with warp-reduced scatter."""
import math
import jax
import jax.numpy as jnp
from mpm_jax.state import MPMState, MPMParams
from mpm_jax.cuda.runtime import CudaRuntime, get_ptr
from mpm_jax.constitutive import get_constitutive


def make_cuda_warp_p2g(cfg, runtime: CudaRuntime):
    """Compile warp-reduced scatter kernel and return P2G closure."""
    kernel = runtime.compile_kernel("p2g_scatter_warp.cu", "p2g_scatter_warp")

    mat = cfg.material.elasticity
    E, nu = float(mat.E), float(mat.nu)
    mu_0 = E / (2.0 * (1.0 + nu))
    lambda_0 = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    block_size = cfg.kernel.get("block_size", 256)

    def p2g(state: MPMState, params: MPMParams):
        N = params.n_particles
        G = params.num_grids

        jax.block_until_ready((state.x, state.v, state.C, state.F))

        grid_mv = jnp.zeros((G ** 3, 3), dtype=jnp.float32)
        grid_m = jnp.zeros((G ** 3,), dtype=jnp.float32)

        grid_dim = math.ceil(N / block_size)

        runtime.launch(
            kernel, grid=grid_dim, block=block_size,
            get_ptr(state.x), get_ptr(state.v),
            get_ptr(state.C), get_ptr(state.F),
            get_ptr(grid_mv), get_ptr(grid_m),
            params.dt, params.vol, params.p_mass,
            params.inv_dx, params.num_grids,
            mu_0, lambda_0,
            N,
        )

        return grid_mv, grid_m

    return p2g
```

- [ ] **Step 3: Write tests/test_p2g_cuda.py**

```python
"""Tests for CUDA P2G variants. Skipped if no GPU available."""
import pytest
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
from mpm_jax.state import MPMState, make_params

try:
    from mpm_jax.cuda.runtime import CudaRuntime
    _has_cuda = True
except (ImportError, RuntimeError):
    _has_cuda = False

requires_cuda = pytest.mark.skipif(not _has_cuda, reason="No CUDA runtime")

_cfg = OmegaConf.create({
    "material": {
        "elasticity": {"name": "corotated_elasticity", "E": 2e6, "nu": 0.4},
        "plasticity": {"name": "identity_plasticity"},
    },
    "kernel": {"name": "cuda_naive", "block_size": 256},
})


def _make_state(N, params):
    return MPMState(
        x=jnp.full((N, 3), 0.5),
        v=jnp.zeros((N, 3)),
        C=jnp.zeros((N, 3, 3)),
        F=jnp.tile(jnp.eye(3), (N, 1, 1)),
    )


@requires_cuda
def test_cuda_naive_shapes():
    from mpm_jax.p2g.cuda_naive import make_cuda_naive_p2g
    runtime = CudaRuntime()
    N = 100
    params = make_params(num_grids=8, n_particles=N)
    state = _make_state(N, params)
    p2g_fn = make_cuda_naive_p2g(_cfg, runtime)

    grid_mv, grid_m = p2g_fn(state, params)

    G = params.num_grids
    assert grid_mv.shape == (G**3, 3)
    assert grid_m.shape == (G**3,)
    assert jnp.all(jnp.isfinite(grid_mv))
    assert jnp.all(jnp.isfinite(grid_m))


@requires_cuda
def test_cuda_naive_matches_jax():
    """CUDA naive P2G should match JAX baseline within tolerance."""
    from mpm_jax.p2g.cuda_naive import make_cuda_naive_p2g
    from mpm_jax.p2g.jax import make_jax_p2g
    runtime = CudaRuntime()
    N = 100
    params = make_params(num_grids=8, n_particles=N)
    state = _make_state(N, params)

    jax_p2g = make_jax_p2g(_cfg)
    cuda_p2g = make_cuda_naive_p2g(_cfg, runtime)

    jax_mv, jax_m = jax_p2g(state, params)
    cuda_mv, cuda_m = cuda_p2g(state, params)

    assert jnp.allclose(jax_m, cuda_m, atol=1e-5)
    assert jnp.allclose(jax_mv, cuda_mv, atol=1e-5)


@requires_cuda
def test_cuda_warp_matches_jax():
    from mpm_jax.p2g.cuda_warp import make_cuda_warp_p2g
    from mpm_jax.p2g.jax import make_jax_p2g
    runtime = CudaRuntime()
    N = 100
    params = make_params(num_grids=8, n_particles=N)
    state = _make_state(N, params)

    warp_cfg = OmegaConf.merge(_cfg, {"kernel": {"name": "cuda_warp"}})
    jax_p2g = make_jax_p2g(_cfg)
    cuda_p2g = make_cuda_warp_p2g(warp_cfg, runtime)

    jax_mv, jax_m = jax_p2g(state, params)
    cuda_mv, cuda_m = cuda_p2g(state, params)

    assert jnp.allclose(jax_m, cuda_m, atol=1e-5)
    assert jnp.allclose(jax_mv, cuda_mv, atol=1e-5)
```

- [ ] **Step 4: Run tests**

Run: `uv run --extra jax --with pytest python -m pytest tests/test_p2g_cuda.py -v`
Expected: PASS on GPU machine, SKIP on CPU-only machine

- [ ] **Step 5: Commit**

```bash
git add mpm_jax/p2g/cuda_naive.py mpm_jax/p2g/cuda_warp.py tests/test_p2g_cuda.py
git commit -m "feat: add CUDA P2G wrappers (naive + warp) with cuda.core"
```

---

### Task 10: Rewrite simulate.py — Python loop + wandb

**Files:**
- Modify: `simulate.py`
- Modify: `conf/config.yaml`
- Create: `conf/kernel/cuda_naive.yaml`
- Create: `conf/kernel/cuda_warp.yaml`
- Delete: `conf/kernel/cuda_v1.yaml`, `conf/kernel/cuda_v3.yaml`, `conf/kernel/cuda_v4.yaml`
- Delete: `conf/profile/*.yaml`
- Delete: `conf/sweep_*.yaml`

- [ ] **Step 1: Update Hydra configs**

`conf/config.yaml`:
```yaml
defaults:
  - material: jelly
  - sim: default
  - kernel: jax
  - _self_

output_dir: ./output
tag: jelly
benchmark: false
```

`conf/kernel/jax.yaml`:
```yaml
name: jax
```

`conf/kernel/cuda_naive.yaml`:
```yaml
name: cuda_naive
block_size: 256
```

`conf/kernel/cuda_warp.yaml`:
```yaml
name: cuda_warp
block_size: 256
```

Delete old configs:
```bash
rm -f conf/kernel/cuda_v1.yaml conf/kernel/cuda_v2.yaml conf/kernel/cuda_v3.yaml conf/kernel/cuda_v4.yaml
rm -rf conf/profile/
rm -f conf/sweep_*.yaml
```

- [ ] **Step 2: Rewrite simulate.py**

Replace the entire file. Key structure:

```python
"""MLS-MPM simulation with pluggable P2G kernels."""
import time
import hydra
import numpy as np
import jax
import jax.numpy as jnp
import wandb
from omegaconf import DictConfig, OmegaConf

from mpm_jax.state import MPMState, make_params
from mpm_jax.grid_update import grid_update
from mpm_jax.g2p import g2p
from mpm_jax.p2g import get_p2g_fn
from mpm_jax.boundary import build_boundary_fns


def get_particles(n_particles, center, size, initial_velocity):
    """Sample particles uniformly in a box."""
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (n_particles, 3))
    x = x * jnp.array(size) + jnp.array(center) - jnp.array(size) / 2
    v = jnp.tile(jnp.array(initial_velocity), (n_particles, 1))
    C = jnp.zeros((n_particles, 3, 3))
    F = jnp.tile(jnp.eye(3), (n_particles, 1, 1))
    return MPMState(x=x, v=v, C=C, F=F)


def simulate(cfg: DictConfig):
    """Run simulation and return step timings."""
    sim = cfg.sim
    params = make_params(
        num_grids=sim.num_grids,
        dt=sim.dt,
        gravity=sim.gravity,
        n_particles=sim.n_particles,
        rho=sim.rho,
        clip_bound=sim.get("clip_bound", 0.5),
        damping=sim.get("damping", 1.0),
    )

    state = get_particles(
        sim.n_particles, sim.center, sim.size, sim.initial_velocity,
    )

    # Initialize CUDA runtime if needed
    runtime = None
    if cfg.kernel.name.startswith("cuda"):
        from mpm_jax.cuda.runtime import CudaRuntime
        runtime = CudaRuntime()

    p2g_fn = get_p2g_fn(cfg, runtime)

    # Build boundary conditions
    # Actual signature: build_boundary_fns(bc_configs, grid_x, dx, init_pos, dt, p_mass)
    G = params.num_grids
    grid_x = jnp.stack(jnp.meshgrid(
        jnp.arange(G), jnp.arange(G), jnp.arange(G), indexing='ij'
    ), axis=-1).reshape(-1, 3).astype(jnp.float32)
    pre_particle_fn, post_grid_fn = build_boundary_fns(
        sim.get("boundary_conditions", []),
        grid_x, params.dx, state.x, params.dt, params.p_mass,
    )
    # pre_particle_fn(x, v, time) -> (x, v)
    # post_grid_fn(grid_v, grid_m, time) -> grid_v

    # Warm-up (JIT compile)
    grid_mv, grid_m = p2g_fn(state, params)
    grid_v = grid_update(grid_mv, grid_m, params)
    _ = g2p(state, grid_v, params)
    jax.block_until_ready(grid_v)

    # Simulation loop
    step_timings = []
    frames = []
    total_start = time.perf_counter()

    for frame in range(sim.num_frames):
        for step in range(sim.steps_per_frame):
            # Apply pre-particle BCs: pre_particle_fn(x, v, time) -> (x, v)
            sim_time = (frame * sim.steps_per_frame + step) * params.dt
            new_x, new_v = pre_particle_fn(state.x, state.v, sim_time)
            state = state._replace(x=new_x, v=new_v)

            t0 = time.perf_counter()
            grid_mv, grid_m = p2g_fn(state, params)
            jax.block_until_ready((grid_mv, grid_m))
            t1 = time.perf_counter()

            grid_v = grid_update(grid_mv, grid_m, params)
            # Apply post-grid BCs: post_grid_fn(grid_v, grid_m, time) -> grid_v
            grid_v = post_grid_fn(grid_v, grid_m, sim_time)
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

        if not cfg.benchmark:
            frames.append(np.array(state.x))

    total_time = time.perf_counter() - total_start
    return step_timings, frames, total_time


def log_to_wandb(cfg, step_timings, total_time):
    """Log metrics to wandb."""
    wandb.init(project="mpm-cuda", config=OmegaConf.to_container(cfg))

    sim = cfg.sim
    steps_per_frame = sim.steps_per_frame

    # Per-frame aggregated time series
    for f in range(sim.num_frames):
        frame_steps = step_timings[f * steps_per_frame : (f + 1) * steps_per_frame]
        wandb.log({
            "frame_p2g_ms": sum(t["p2g_ms"] for t in frame_steps),
            "frame_grid_update_ms": sum(t["grid_update_ms"] for t in frame_steps),
            "frame_g2p_ms": sum(t["g2p_ms"] for t in frame_steps),
            "frame_step_ms": sum(t["step_ms"] for t in frame_steps),
        })

    # Summary
    total_steps = len(step_timings)
    wandb.summary.update({
        "mean_p2g_ms": np.mean([t["p2g_ms"] for t in step_timings]),
        "mean_grid_update_ms": np.mean([t["grid_update_ms"] for t in step_timings]),
        "mean_g2p_ms": np.mean([t["g2p_ms"] for t in step_timings]),
        "mean_step_ms": np.mean([t["step_ms"] for t in step_timings]),
        "total_steps": total_steps,
        "total_elapsed_s": total_time,
        "steps_per_sec": total_steps / total_time,
        "n_particles": cfg.sim.n_particles,
        "kernel": cfg.kernel.name,
    })

    wandb.finish()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    step_timings, frames, total_time = simulate(cfg)
    log_to_wandb(cfg, step_timings, total_time)

    if not cfg.benchmark and frames:
        # Reuse the existing visualize_frames() from simulate.py
        # (extract to a utility or keep inline — same matplotlib scatter plot + GIF)
        visualize_frames(frames, cfg.output_dir, cfg.tag)


if __name__ == "__main__":
    main()
```

Note: The boundary condition integration needs to match the existing `build_boundary_fns` API. Check `boundary.py:144-198` for the exact interface — `pre_particle_fn(state, time)` and `post_grid_fn(grid_v)`. Adapt as needed.

- [ ] **Step 3: Commit**

```bash
git add simulate.py conf/config.yaml conf/kernel/cuda_naive.yaml conf/kernel/cuda_warp.yaml conf/kernel/jax.yaml
git rm conf/kernel/cuda_v1.yaml conf/kernel/cuda_v2.yaml conf/kernel/cuda_v3.yaml conf/kernel/cuda_v4.yaml
git rm -rf conf/profile/ conf/sweep_*.yaml
git commit -m "refactor: rewrite simulate.py with Python timestep loop and wandb logging"
```

---

### Task 11: Clean up old files

**Files:**
- Delete: `mpm_jax/solver.py`
- Delete: `mpm_jax/cuda/p2g_cuda.py`
- Delete: `tests/test_solver.py`

- [ ] **Step 1: Remove old solver and FFI code**

```bash
git rm mpm_jax/solver.py
git rm mpm_jax/cuda/p2g_cuda.py
git rm tests/test_solver.py
```

- [ ] **Step 2: Update mpm_jax/__init__.py if it imports from solver**

Check `mpm_jax/__init__.py` — if it re-exports from solver, update imports to point to new modules.

- [ ] **Step 3: Run full test suite**

Run: `uv run --extra jax --with pytest python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git rm mpm_jax/solver.py mpm_jax/cuda/p2g_cuda.py tests/test_solver.py
git commit -m "chore: remove old solver.py, FFI code, and legacy tests"
```

---

### Task 12: Smoke test — end-to-end JAX simulation

- [ ] **Step 1: Run full JAX simulation**

```bash
cd MPM-CudaJax
uv run --extra jax python simulate.py kernel=jax sim.num_frames=5 benchmark=true
```

Expected: Completes without error, logs to wandb with per-frame and summary metrics.

- [ ] **Step 2: Verify wandb dashboard**

Check that the wandb run contains:
- Per-frame time series (`frame_p2g_ms`, `frame_grid_update_ms`, `frame_g2p_ms`)
- Summary metrics (`mean_p2g_ms`, `steps_per_sec`, etc.)
- Config captured from Hydra

- [ ] **Step 3: Run all tests one final time**

```bash
uv run --extra jax --with pytest python -m pytest tests/ -v
```

Expected: All PASS

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: address smoke test issues"
```

---

## Task Dependency Order

```
Task 1 (branch + deps)
  └→ Task 2 (state.py)
      ├→ Task 3 (grid_update.py)
      ├→ Task 4 (g2p.py)
      └→ Task 5 (p2g/jax.py)
          └→ Task 6 (integration test)
              ├→ Task 7 (cuda/runtime.py)
              │   └→ Task 8 (CUDA kernels)
              │       └→ Task 9 (CUDA P2G wrappers)
              └→ Task 10 (simulate.py rewrite)
                  └→ Task 11 (cleanup)
                      └→ Task 12 (smoke test)
```

Tasks 3, 4, 5 can run in parallel after Task 2.
Tasks 7-9 run sequentially (7 → 8 → 9). Task 10 depends on Tasks 6 AND 9 (imports from both JAX and CUDA P2G modules).
