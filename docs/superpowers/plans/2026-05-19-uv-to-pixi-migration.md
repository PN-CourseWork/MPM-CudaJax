# uv → pixi Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `uv` with `pixi` as the package manager, with a CPU `default` environment and a Linux-only `gpu` environment (JAX + CUDA toolchain + nvcc + gxx all from conda-forge).

**Architecture:** Pixi config lives under `[tool.pixi.*]` in the existing `pyproject.toml`. The `[build-system]` and `[project]` blocks stay so scikit-build-core still builds the CUDA wheel. Two environments via features (`cpu`, `gpu`, plus a shared `dev` feature). The local project is an editable pypi-dependency installed in both envs; CMake skips the CUDA build cleanly when `nvcc` is absent (so the CPU env works on macOS).

**Tech Stack:** pixi 0.68+, conda-forge channel, scikit-build-core, JAX, CUDA 12.

**Spec:** `docs/superpowers/specs/2026-05-19-uv-to-pixi-migration-design.md`

---

## File Structure

**Modified:**
- `pyproject.toml` — add `[tool.pixi.*]` sections; drop `[project.optional-dependencies]`.
- `.gitignore` — add `.pixi/`, drop `uv.lock`.
- `README.md` — replace every `uv …` snippet with the pixi equivalent.
- `.claude/CLAUDE.md` — rewrite the "Package manager: uv" section.

**Created:**
- `pixi.lock` — committed to git (pixi's lockfile, equivalent of `uv.lock`).

**Deleted:**
- `Makefile`
- `uv.lock`
- `mpm_cudajax.egg-info/` (stale)

---

## Task 1: Add pixi config (CPU-only first cut)

**Files:**
- Modify: `pyproject.toml:19-21` (replace the `[project.optional-dependencies]` block with `[tool.pixi.*]` sections)

- [ ] **Step 1: Replace the `[project.optional-dependencies]` block**

Remove these three lines from `pyproject.toml`:
```toml
[project.optional-dependencies]
jax = ["jax>=0.4.20", "jaxlib>=0.4.20"]
jax-cuda = ["jax[cuda12]>=0.4.20"]
```

Append the following to the end of `pyproject.toml`:

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
cmake = ">=3.24"
scikit-build-core = ">=0.10"
ninja = "*"

[tool.pixi.pypi-dependencies]
mpm-cudajax = { path = ".", editable = true }

[tool.pixi.feature.cpu.dependencies]
jax = ">=0.4.20"
jaxlib = ">=0.4.20"

[tool.pixi.feature.dev.dependencies]
pytest = "*"
ruff = "*"

[tool.pixi.environments]
default = { features = ["cpu", "dev"], solve-group = "cpu" }
```

- [ ] **Step 2: Solve and install the default env**

Run: `pixi install`
Expected: pixi resolves `linux-64`, `linux-aarch64`, `osx-arm64`, writes `pixi.lock`, installs deps into `.pixi/envs/default/`, and runs an editable install of `mpm-cudajax` (which triggers scikit-build-core → CMake → `check_language(CUDA)` fails → `return()` → wheel installs without kernels).

If `pixi install` fails because conda-forge has no `jax` for `osx-arm64`, fall back: edit `pyproject.toml` to wrap the cpu feature deps in a platform-specific block as documented in pixi docs (`[tool.pixi.feature.cpu.target.osx-arm64.dependencies]`).

- [ ] **Step 3: Verify `mpm_jax` is importable**

Run: `pixi run python -c "import mpm_jax; import jax; print(jax.devices())"`
Expected: prints `[CpuDevice(id=0)]` (or similar) and no import error.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml pixi.lock
git commit -m "build: add pixi config with cpu default environment"
```

---

## Task 2: Add the `gpu` feature + environment

**Files:**
- Modify: `pyproject.toml` (append two more sections)

- [ ] **Step 1: Append gpu feature and environment**

Append to `pyproject.toml`:

```toml
[tool.pixi.feature.gpu.dependencies]
jax = ">=0.4.20"
jaxlib = { version = ">=0.4.20", build = "*cuda12*" }
cuda-version = "12.*"
cuda-nvcc = "*"
gxx = "*"

[tool.pixi.feature.gpu.system-requirements]
cuda = "12.0"

[tool.pixi.feature.gpu.target.linux-64.dependencies]

[tool.pixi.feature.gpu.target.linux-aarch64.dependencies]
```

Edit the `[tool.pixi.environments]` block to add a `gpu` env:

```toml
[tool.pixi.environments]
default = { features = ["cpu", "dev"], solve-group = "cpu" }
gpu = { features = ["gpu", "dev"], solve-group = "gpu" }
```

(The two empty `target.*` blocks are placeholders that make the dependency explicit — pixi will reject the gpu env on `osx-arm64` automatically because `*cuda12*` jaxlib builds aren't available there.)

- [ ] **Step 2: Verify the spec parses**

Run: `pixi info`
Expected: lists both `default` and `gpu` environments. The `gpu` env may show "no compatible platform" on macOS — that is fine and expected; we'll resolve it on a Linux GPU host.

If `pixi info` errors on the gpu env spec, drop the empty `target.*` blocks (they were just documentation).

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add gpu pixi environment with conda-forge CUDA toolchain"
```

---

## Task 3: Add pixi tasks (replacing Makefile targets)

**Files:**
- Modify: `pyproject.toml` (append `[tool.pixi.tasks]` and `[tool.pixi.feature.gpu.tasks]`)

- [ ] **Step 1: Append tasks**

Append to `pyproject.toml`:

```toml
[tool.pixi.tasks]
test = "pytest tests/ -v"
lint = "ruff check ."
clean = { cmd = "rm -f mpm_jax/cuda/_lib/*.so mpm_jax/cuda/kernels/*.so && rm -rf build multirun outputs output && rm -f *.nsys-rep *.sqlite && find . -type d -name __pycache__ -exec rm -rf {} +" }

[tool.pixi.feature.gpu.tasks]
sim = "python simulate.py"
sweep = "python simulate.py -cn sweep_baseline"
sweep-quick = "python simulate.py -cn sweep_quick"
sweep-all = "python simulate.py -cn sweep_all"
```

- [ ] **Step 2: Verify task list**

Run: `pixi task list`
Expected: lists `test`, `lint`, `clean` under both envs, plus `sim`, `sweep`, `sweep-quick`, `sweep-all` under `gpu` only.

- [ ] **Step 3: Run lint as a sanity check**

Run: `pixi run lint`
Expected: ruff runs and exits with 0 (assuming no existing lint errors; if it surfaces pre-existing errors, leave them — out of scope).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml pixi.lock
git commit -m "build: add pixi tasks replacing Makefile targets"
```

---

## Task 4: Remove uv artifacts

**Files:**
- Delete: `Makefile`
- Delete: `uv.lock`
- Delete: `mpm_cudajax.egg-info/` (entire directory)
- Modify: `.gitignore` (replace `uv.lock` with `.pixi/`)

- [ ] **Step 1: Delete files**

```bash
rm Makefile uv.lock
rm -rf mpm_cudajax.egg-info
```

- [ ] **Step 2: Update `.gitignore`**

In `.gitignore`, replace the line `uv.lock` with `.pixi/`. Final relevant lines should be:

```
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.venv/
.pixi/
*.egg
mpm_jax/cuda/_lib/
output/
outputs/
.pytest_cache/

multirun/
outputs/
```

- [ ] **Step 3: Verify `git status` is clean of those files**

Run: `git status`
Expected: shows `Makefile`, `uv.lock` deleted; `mpm_cudajax.egg-info/` gone (it was gitignored, so it won't appear); `.gitignore` modified.

- [ ] **Step 4: Commit**

```bash
git add -u .gitignore Makefile uv.lock
git commit -m "build: drop Makefile and uv artifacts in favor of pixi"
```

---

## Task 5: Validate on a small simulation (CPU env, local)

**Files:**
- None modified — this is verification only.

- [ ] **Step 1: Clear any stale outputs**

Run: `pixi run clean`
Expected: removes `output/`, `outputs/`, `multirun/`, and any built `.so` files.

- [ ] **Step 2: Run a short JAX simulation**

Run: `pixi run python simulate.py sim.num_frames=5 benchmark=true`
Expected: runs, prints `total_steps`, `elapsed_s`, `steps_per_sec`, `ms/step`. No GIF (benchmark mode suppresses rendering). No crash. JAX uses the CPU device.

If this fails because `jax.devices()` shows no CPU device, check that `jaxlib` got installed: `pixi run python -c "import jaxlib; print(jaxlib.__version__)"`.

- [ ] **Step 3: Run a short rendered simulation**

Run: `pixi run python simulate.py sim.num_frames=5 sim.n_particles=2000`
Expected: writes `output/jelly_jax.gif` (or similar), file size > 1 KB. Takes a few seconds.

Verify the GIF exists:
```bash
ls -lh output/
```
Expected: a `.gif` file is present.

- [ ] **Step 4: Run the test suite**

Run: `pixi run test`
Expected: ~28 CPU tests pass. The 4 CUDA equivalence tests should skip (they check for nvcc / built kernels and skip when absent).

If a test fails because of a missing dep, add it to `[tool.pixi.feature.dev.dependencies]` and re-run.

- [ ] **Step 5: Commit** (only if .lock changed; otherwise skip)

```bash
git add pixi.lock 2>/dev/null || true
git diff --cached --quiet || git commit -m "build: refresh pixi.lock after validation"
```

---

## Task 6: Update README.md

**Files:**
- Modify: `README.md` (replace every `uv …` invocation)

- [ ] **Step 1: Rewrite the `Quickstart` section**

Replace lines 14-57 (the `## Quickstart` section) with:

```markdown
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
the custom CUDA kernels via CMake — `nvcc` and `gxx` come from
conda-forge, no system module load needed):
```bash
pixi install -e gpu
pixi run -e gpu python simulate.py kernel=cuda_v2 timing_mode=per_stage
```

To benchmark instead of rendering:
```bash
pixi run -e gpu python simulate.py \
    kernel=cuda_v2 timing_mode=per_stage \
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
```

- [ ] **Step 2: Rewrite the `## Setup` section**

Replace lines 59-99 (the `## Setup` section) with:

```markdown
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
missing (the CPU env) CMake's `check_language(CUDA)` returns early, the
wheel installs cleanly, and the JAX baseline still works.

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
```

- [ ] **Step 3: Find-and-replace remaining `uv` references**

In `README.md`, replace these patterns (using whatever editor invocations the model needs; the substitutions are mechanical):

- `uv run --extra jax-cuda python simulate.py` → `pixi run -e gpu python simulate.py`
- `uv run --extra jax python simulate.py` → `pixi run python simulate.py`
- `uv run --extra jax --with pytest python -m pytest tests/ -v` → `pixi run test`
- `uv sync --extra jax-cuda` → `pixi install -e gpu`
- `uv sync --extra jax` → `pixi install`

Also: in the "Project Structure" section, remove the `Makefile` line from the tree (lines around 281).

In the `pyproject.toml` comment in that tree, change `# scikit-build-core build, jax / jax-cuda extras` to `# scikit-build-core build + pixi cpu / gpu envs`.

- [ ] **Step 4: Verify no `uv` references remain**

Run: `grep -n "uv " README.md ; grep -n "uv\." README.md`
Expected: no matches (or only matches inside example code / URLs that aren't relevant).

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README for pixi-based setup"
```

---

## Task 7: Update `.claude/CLAUDE.md`

**Files:**
- Modify: `.claude/CLAUDE.md` (rewrite the "Package manager: uv" section and update every `uv …` example)

- [ ] **Step 1: Rewrite the "Package manager" section**

Find the section starting `## Package manager: uv` (around line 9 of `.claude/CLAUDE.md`). Replace the whole section, from that heading through the line before `## Layout`, with:

```markdown
## Package manager: pixi

**Always use `pixi` to install, sync, and run.** Never invoke `pip`,
`pip install`, `python -m pip`, or a bare `python` from the system
interpreter — those will miss the project's locked environment.

Two environments, defined in `[tool.pixi.environments]` in
`pyproject.toml`:

- `default` — CPU. JAX from conda-forge, no CUDA. Use on macOS / laptop.
- `gpu` — Linux only. JAX with `*cuda12*` jaxlib build, `cuda-nvcc`,
  `gxx`, full CUDA 12 toolchain from conda-forge. No `module load`
  required on DTU HPC — everything ships from conda-forge.

Common patterns:

```bash
pixi install                                      # default (CPU) env
pixi install -e gpu                               # GPU env (Linux)
pixi run python simulate.py ...                   # default env
pixi run -e gpu python simulate.py ...            # gpu env
pixi run test                                     # pytest
pixi run -e gpu sim                               # task alias for `python simulate.py`
pixi run -e gpu sweep-quick                       # task alias for `simulate.py -cn sweep_quick`
pixi add <pkg>                                    # add a runtime dep (edits pyproject.toml)
pixi add --feature gpu <pkg>                      # add to the gpu feature only
```

### CUDA kernel build (scikit-build-core + CMake)

CUDA kernels in `mpm_jax/cuda/kernels/*.cu` build via `CMakeLists.txt`
driven by **scikit-build-core** at `pixi install` time. Output `.so`
files land in `mpm_jax/cuda/_lib/` (gitignored) and are loaded by
`mpm_jax/cuda/p2g_cuda.py` which registers them with JAX FFI
(`jax.ffi.register_ffi_target` / `ffi_call`).

Key knobs:

- `MPM_CUDA_ARCH=sm_86` (or `sm_90`, etc.) at install time → CMake picks
  that arch. Default is `native` (CMake auto-detects the local GPU). Set
  this before `pixi install -e gpu` on cross-build hosts.
- If `nvcc` is not on PATH (the default CPU env), CMake's
  `check_language(CUDA)` returns early and the wheel installs fine
  without CUDA kernels — the JAX baseline still works. Useful for
  CPU-only dev.
- `editable.rebuild = true` in `pyproject.toml` means edits to `.cu`
  sources trigger a rebuild on the next `import mpm_jax.cuda.p2g_cuda`.
- `[build-system].requires` pulls in `scikit-build-core>=0.10`,
  `cmake>=3.24`, and `jax>=0.4.20` (jax is needed at build time so CMake
  can `import jax.ffi` to find the FFI headers).
```

- [ ] **Step 2: Find-and-replace remaining `uv` references**

In `.claude/CLAUDE.md`, apply the same substitutions as the README:

- `uv run --extra jax-cuda python simulate.py` → `pixi run -e gpu python simulate.py`
- `uv run --extra jax python simulate.py` → `pixi run python simulate.py`
- `uv run --extra jax --with pytest python -m pytest tests/ -v` → `pixi run test`
- `uv sync --extra jax-cuda` → `pixi install -e gpu`
- `uv sync --extra jax` → `pixi install`
- `uv add` → `pixi add`
- `uv add --optional jax` → `pixi add --feature cpu`
- `uv add --optional jax-cuda` → `pixi add --feature gpu`

Also: in the "Layout" section, remove the `Makefile` line.

In the "DTU HPC notes" section, replace the contents with:

```markdown
## DTU HPC notes

The `gpu` environment is fully self-contained — no `module load` is
needed because conda-forge provides `cuda-nvcc`, `gxx`, and the cuda
runtime libs inside the env.

```bash
MPM_CUDA_ARCH=sm_90 pixi install -e gpu    # build kernels for Hopper
pixi run -e gpu sim                        # smoke-test
```

CMake auto-detects the local GPU arch when `MPM_CUDA_ARCH` is unset.
```

- [ ] **Step 3: Update the "Don't" section**

Change the first bullet of `## Don't` from:

```
- Don't run `pip install` — use `uv add` / `uv sync`.
```

to:

```
- Don't run `pip install` — use `pixi add` / `pixi install`.
```

And change the bullet:

```
- Don't commit `build/`, `output/`, `outputs/`, `multirun/`, `*.nsys-rep`, `*.sqlite`, or `uv.lock` (`.gitignore` covers these).
```

to:

```
- Don't commit `build/`, `output/`, `outputs/`, `multirun/`, `*.nsys-rep`, `*.sqlite`, or `.pixi/` (`.gitignore` covers these). DO commit `pixi.lock`.
```

- [ ] **Step 4: Verify no `uv` references remain**

Run: `grep -n "uv " .claude/CLAUDE.md ; grep -n "\buv\b" .claude/CLAUDE.md`
Expected: no matches.

- [ ] **Step 5: Commit**

```bash
git add .claude/CLAUDE.md
git commit -m "docs: rewrite project CLAUDE.md for pixi"
```

---

## Task 8: Final validation

**Files:**
- None modified.

- [ ] **Step 1: Re-run the small simulation end-to-end**

Run: `pixi run python simulate.py sim.num_frames=5 sim.n_particles=2000`
Expected: completes, GIF written to `output/`.

- [ ] **Step 2: Re-run the test suite**

Run: `pixi run test`
Expected: same pass count as Task 5 Step 4. No regressions.

- [ ] **Step 3: `git status` clean**

Run: `git status`
Expected: clean working tree, all commits on the `pixi` branch.

- [ ] **Step 4: Final commit (only if anything changed)**

If pixi.lock or any other file changed, commit it:
```bash
git diff --quiet || git commit -am "build: final pixi.lock refresh"
```

---

## Out of scope (do not do in this plan)

- GPU env validation on DTU HPC (Linux + Hopper) — user will run this separately on the cluster.
- Adding new sweep configs, kernels, or features.
- Removing CUDA-related profiler logic from `simulate.py`.
- Rewriting `CMakeLists.txt`.
