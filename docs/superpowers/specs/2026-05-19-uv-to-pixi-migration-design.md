# uv → pixi migration

## Goal

Replace `uv` with `pixi` as the package manager so the project has a first-class GPU environment (JAX + CUDA toolchain + nvcc + gxx, all from conda-forge) and a separate CPU environment for laptop development. The migration is self-contained: no functional change to the solver, kernels, configs, or simulation behavior.

## Why pixi

- A dedicated `gpu` environment can pin the full CUDA toolchain (nvcc, cuda-version, cuda libs, matching gxx) from conda-forge. No more `module load nvhpc/26.1 gcc/15.2` shim on DTU HPC.
- The two extras we have today (`jax` / `jax-cuda`) are mutually exclusive in spirit but uv resolves them in the same lockfile, which is awkward. Pixi environments are first-class and live side by side with their own lock state.
- Conda-forge ships `jaxlib=*=*cuda12*` so we no longer depend on PyPI's CUDA wheel layout.

## Out of scope

- No changes to `mpm_jax/`, `simulate.py`, `conf/`, `tests/`, or the CUDA kernel sources.
- No changes to `CMakeLists.txt` build logic. Scikit-build-core still drives the kernel build.
- No new sweep configs or kernels.
- README + CLAUDE.md updates beyond replacing `uv` snippets are out of scope; that is purely mechanical.

## Design

### 1. `pyproject.toml`

Keep `[build-system]` and `[project]` as-is — scikit-build-core still builds the wheel that compiles CUDA kernels. Drop `[project.optional-dependencies]` (`jax`, `jax-cuda`); their role moves to pixi features.

Add a `[tool.pixi.*]` section:

```toml
[tool.pixi.project]
name = "mpm-cudajax"
channels = ["conda-forge"]
platforms = ["linux-64", "linux-aarch64", "osx-arm64"]

[tool.pixi.dependencies]
python = ">=3.10,<3.13"
hydra-core = ">=1.3"
omegaconf = "*"
matplotlib = "*"
tqdm = "*"
numpy = "*"
# Build tools so the editable install can compile the wheel even on CPU envs
# (the CUDA portion is gated by `check_language(CUDA)` and is skipped if nvcc
# isn't on PATH).
cmake = ">=3.24"
scikit-build-core = ">=0.10"
ninja = "*"

[tool.pixi.pypi-dependencies]
# Editable install of the project itself. Lives in every env so `mpm_jax` is
# always importable; CMake skips the CUDA build when nvcc is absent.
mpm-cudajax = { path = ".", editable = true }

[tool.pixi.feature.cpu.dependencies]
jax = ">=0.4.20"
jaxlib = ">=0.4.20"

[tool.pixi.feature.gpu.dependencies]
jax = ">=0.4.20"
jaxlib = { version = ">=0.4.20", build = "*cuda12*" }
cuda-version = "12.*"
cuda-nvcc = "*"
gxx = "*"

[tool.pixi.feature.gpu.system-requirements]
cuda = "12.0"

[tool.pixi.feature.dev.dependencies]
pytest = "*"
ruff = "*"

[tool.pixi.environments]
default = { features = ["cpu", "dev"], solve-group = "cpu" }
gpu = { features = ["gpu", "dev"], solve-group = "gpu" }
```

Notes:
- The `gpu` environment is only solvable on `linux-64` / `linux-aarch64` (conda-forge has no CUDA builds for macOS). Pixi skips it automatically on `osx-arm64`.
- Both envs include `dev` so `pytest` / `ruff` work without an extra activation.
- `solve-group` keeps shared deps (numpy, hydra) at the same version across envs.

### 2. Tasks (replacing the Makefile)

```toml
[tool.pixi.tasks]
test = "pytest tests/ -v"
lint = "ruff check ."
clean = """
rm -f mpm_jax/cuda/_lib/*.so mpm_jax/cuda/kernels/*.so && \
rm -rf build/ multirun/ outputs/ output/ && \
rm -f *.sqlite && \
find . -type d -name __pycache__ -exec rm -rf {} +
"""

[tool.pixi.feature.gpu.tasks]
sim = "python simulate.py"
sweep = "python simulate.py -cn sweep_baseline"
sweep-quick = "python simulate.py -cn sweep_quick"
sweep-all = "python simulate.py -cn sweep_all"
```

`pixi run <task>` resolves to the right env automatically; GPU-feature tasks only exist in the `gpu` env.

### 3. Files removed

- `Makefile` — deleted.
- `uv.lock` — deleted (replaced by `pixi.lock`).
- `mpm_cudajax.egg-info/` — stale, deleted.
- `[project.optional-dependencies]` block in `pyproject.toml`.

### 4. Files added

- `pixi.lock` — committed (this is the pixi equivalent of `uv.lock`; pixi expects it in git).
- `.gitignore` entries: `.pixi/` (per-env install dir).

### 5. Documentation updates (mechanical)

- `README.md` Quickstart: swap all `uv sync …` and `uv run …` commands for `pixi install` / `pixi run …`.
- `.claude/CLAUDE.md` (project): rewrite "Package manager: uv" section to "Package manager: pixi", update every example, drop the DTU HPC `module load` block (no longer needed — conda-forge supplies nvcc + gxx).
- Mention the `default` / `gpu` env split explicitly.

### 6. MPM_CUDA_ARCH

Keep as an opt-in env var. CMake auto-detects when unset (`native`). For Hopper cross-builds, document `MPM_CUDA_ARCH=sm_90 pixi install -e gpu` in the README, same shape as today.

Optionally, set it in `[tool.pixi.feature.gpu.activation.env]` for the gpu env if we want a default — leave it unset for now to preserve current behavior.

## Verification

Each must pass before declaring done:

1. **CPU env, macOS:** `pixi run test` — all tests pass that don't require CUDA. The cuda_equivalence tests should skip gracefully.
2. **GPU env, DTU HPC:** `pixi install -e gpu` succeeds without any `module load`. `pixi run -e gpu sim` runs a default render. `pixi run -e gpu sweep-quick` completes. `jax.devices()` returns a CUDA device inside the env.
3. **CUDA kernels actually built:** after `pixi install -e gpu`, `ls mpm_jax/cuda/_lib/*.so` is non-empty.
4. **`kernel=cuda_v2` parity:** one short benchmark run matches the existing performance ballpark (within noise).

## Risk + fallback

The main risk is conda-forge `jaxlib=*=*cuda*` not having a build for `linux-aarch64` (DTU's Grace nodes are aarch64). If that's the case, the gpu env on aarch64 falls back to a `pypi-dependencies` entry `jax[cuda12]` plus `cuda-nvcc` from conda-forge — still pixi-native, just one pypi shim. Decide after the first `pixi install -e gpu` attempt on HPC.
