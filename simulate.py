import os
import time
import subprocess
import ctypes
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb


# ---------------------------------------------------------------------------
# CUDA profiler markers (for nsys --capture-range=cudaProfilerApi)
# ---------------------------------------------------------------------------

def _get_cudart():
    """Load libcudart for profiler start/stop. Returns None if unavailable."""
    for name in ["libcudart.so", "libcudart.so.12", "libcudart.dylib"]:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


_cudart = None


def cuda_profiler_start():
    global _cudart
    if _cudart is None:
        _cudart = _get_cudart()
    if _cudart:
        _cudart.cudaProfilerStart()


def cuda_profiler_stop():
    if _cudart:
        _cudart.cudaProfilerStop()


def get_particles(n_particles, center, size):
    """Sample n_particles uniformly in a box."""
    start = np.array(center) - np.array(size) / 2
    end = np.array(center) + np.array(size) / 2
    rng = np.random.RandomState(42)
    return start + rng.rand(n_particles, 3) * (end - start)


def visualize_frames(frames, export_path, center=[0.5, 0.5, 0.5],
                     size=[2.0, 2.0, 2.0], c='blue', s=20, fps=30):
    xlim = [center[0] - size[0]/2, center[0] + size[0]/2]
    ylim = [center[1] - size[1]/2, center[1] + size[1]/2]
    zlim = [center[2] - size[2]/2, center[2] + size[2]/2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    def update(frame):
        ax.cla()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.scatter(frames[frame][:, 0], frames[frame][:, 1], frames[frame][:, 2], s=s, c=c)
        ax.set_title(f'Frame {frame}')

    ani = FuncAnimation(fig, update, frames=len(frames), blit=False)
    ani.save(export_path, writer='pillow', fps=fps)
    plt.close()


# ---------------------------------------------------------------------------
# Per-stage timing helpers
# ---------------------------------------------------------------------------

class StageTimer:
    """Accumulates per-stage wall-clock times across substeps."""

    def __init__(self):
        self.stages = {}
        self._start = None
        self._current = None

    def start(self, name):
        self._current = name
        self._start = time.perf_counter()

    def stop(self):
        elapsed = time.perf_counter() - self._start
        if self._current not in self.stages:
            self.stages[self._current] = []
        self.stages[self._current].append(elapsed)
        self._current = None

    def flush_frame(self):
        """Pop accumulated times for the current frame and return per-stage totals in ms."""
        out = {}
        for name, times in self.stages.items():
            out[name] = sum(times) * 1000  # ms
        self.stages.clear()
        return out

    def summary_from_frames(self, frame_timings):
        """Compute overall summary from a list of per-frame dicts."""
        all_stages = {}
        for ft in frame_timings:
            for name, ms in ft.items():
                all_stages.setdefault(name, []).append(ms)
        out = {}
        for name, vals in all_stages.items():
            arr = np.array(vals)
            out[name] = {
                'mean_ms': float(arr.mean()),
                'std_ms': float(arr.std()),
                'total_ms': float(arr.sum()),
                'count': len(vals),
            }
        return out


# ---------------------------------------------------------------------------
# Backend-specific runners
# ---------------------------------------------------------------------------

def _maybe_enable_cuda_graphs(kernel_name):
    """Toggle XLA command-buffer capture (= CUDA Graphs) for the v6 kernel.

    Must be called BEFORE the first `import jax`, otherwise XLA has already
    parsed XLA_FLAGS and the new value is ignored. Routed from run_jax() at
    its very top, and (defensively) from main() before any jax import.

    Project plan v6: capture the repeating P2G -> grid_update -> G2P substep
    as a CUDA Graph and replay it. Concretely we ask XLA to wrap FUSION,
    CUSTOM_CALL (our FFI scatter / fused G2P) and WHILE (the lax.scan
    substep loop) into command buffers, which the GPU runtime executes as
    a single replayed graph per substep.
    """
    if kernel_name != 'cuda_v6_inline':
        return
    extra = "--xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL,WHILE"
    cur = os.environ.get("XLA_FLAGS", "")
    if extra not in cur:
        os.environ["XLA_FLAGS"] = (cur + " " + extra).strip()
        print(f"cuda_v6_inline: enabling XLA CUDA Graph capture via XLA_FLAGS={os.environ['XLA_FLAGS']}")


def run_jax(cfg: DictConfig):
    # Must come BEFORE `import jax` — XLA reads XLA_FLAGS once at startup.
    kernel_name = cfg.get('kernel', {}).get('name', 'jax')
    _maybe_enable_cuda_graphs(kernel_name)

    import jax
    import jax.numpy as jnp
    from mpm_jax.solver import (
        MPMState, make_params, build_jit_step, build_jit_frame, build_jit_stages,
    )
    from mpm_jax.constitutive import get_constitutive
    from mpm_jax.boundary import build_boundary_fns

    sim = cfg.sim
    mat = cfg.material
    bench = cfg.get('benchmark', False)

    # Build p2g_fn based on kernel config. cuda_fused does its own thing
    # because the kernel covers stress + plasticity + scatter and doesn't fit
    # the (v, C, stress, weight, ...) -> (grid_mv, grid_m) signature.
    p2g_fn = None         # None = default JAX implementation
    fused_stages = None   # set only for cuda_fused

    if kernel_name == 'cuda_v1':
        from mpm_jax.cuda.p2g_cuda import make_cuda_p2g
        p2g_fn = make_cuda_p2g(sim.num_grids, kernel='scatter')
        if p2g_fn is None:
            raise RuntimeError(
                "kernel=cuda_v1 requested but CUDA kernel failed to compile/register. "
                "Run `pixi install -e gpu` in an env where nvcc is on PATH."
            )
        print("Using CUDA P2G scatter kernel (v1, naive atomicAdd)")
    elif kernel_name == 'cuda_v2':
        from mpm_jax.cuda.p2g_cuda import make_cuda_p2g
        p2g_fn = make_cuda_p2g(sim.num_grids, kernel='warp')
        if p2g_fn is None:
            raise RuntimeError(
                "kernel=cuda_v2 requested but CUDA kernel failed to compile/register. "
                "Run `pixi install -e gpu` in an env where nvcc is on PATH."
            )
        print("Using CUDA P2G warp-reduction scatter kernel (v2)")
    elif kernel_name == 'cuda_v4':
        from mpm_jax.cuda.p2g_cuda import make_cuda_p2g
        p2g_fn = make_cuda_p2g(sim.num_grids, kernel='smem')
        if p2g_fn is None:
            raise RuntimeError(
                "kernel=cuda_v4 requested but CUDA kernel failed to compile/register."
            )
        print("Using CUDA P2G shared-memory scatter kernel (v4)")
    elif kernel_name == 'cuda_fused':
        # cuda_fused collapses stress / weights / compute / scatter / plasticity
        # into a single CUDA kernel launch — no XLA-side intermediates.
        # Implemented only on the per-stage path; per_frame would need a new
        # build_jit_frame_fused() that we don't have.
        print("Using CUDA fully fused P2G + G2P kernel — stress + scatter in one launch")
    elif kernel_name == 'cuda_v1_inline':
        # cuda_v1_inline: one CUDA kernel for inline-weights + 27-stencil
        # atomic scatter, one thread per particle, register-resident loop.
        # JAX does stress upstream (use material=jelly_jacobi for the fast
        # Jacobi SVD). Wired into the per-frame path: the FFI call lives
        # inside lax.scan so the whole frame compiles to one XLA program.
        print("Using CUDA inline P2G kernel (cuda_v1_inline) — JAX stress + register-resident scatter")
    elif kernel_name == 'cuda_v2_inline':
        # cuda_v2_inline: same as cuda_v1_inline but with warp-shuffle
        # reduction (__match_any_sync + __shfl_xor_sync) folded into every
        # atomicAdd inside the 27-stencil loop. Tests whether warp coalescing
        # helps once the (N, 27, *) materialisation overhead is gone.
        print("Using CUDA inline P2G kernel with warp shuffle (cuda_v2_inline)")
    elif kernel_name == 'cuda_v3_inline':
        # cuda_v3_inline: cuda_v1_inline + Morton (Z-order) spatial sort of
        # particles before each substep + warp-shuffle atomic coalescing.
        # Hypothesis: spatially-close particles in the same warp share more
        # stencil-node targets, so `__match_any_sync` collapses 4-8x of the
        # atomics into one. Tradeoff: an argsort per substep (O(N log N)).
        print("Using CUDA inline P2G kernel with Morton sort + warp shuffle (cuda_v3_inline)")
    elif kernel_name == 'cuda_v6_inline':
        # cuda_v6_inline: exactly the cuda_v3_inline pipeline (Morton sort +
        # warp-shuffle inline scatter + fused G2P), but with XLA's command
        # buffer / CUDA Graph capture enabled via XLA_FLAGS. The substep
        # body (FUSION + CUSTOM_CALL + WHILE) is wrapped into a CUDA Graph
        # and replayed each substep, eliminating most of the per-kernel
        # launch dispatch overhead. Project plan v6 (L4: Streams & Graphs).
        print("Using cuda_v3_inline pipeline replayed through XLA CUDA Graph capture (cuda_v6_inline)")
    elif kernel_name == 'cuda_v4_inline':
        # cuda_v4_inline: cell-major scheduling + 4^3 shared-memory tile +
        # inline weights. JAX sorts particles by home cell every substep and
        # passes (sorted x, v, C, stress, cell_start) to one CUDA launch.
        # Per-frame only (the sort + ffi call live inside lax.scan).
        print("Using CUDA cell-major + smem-tile inline P2G kernel (cuda_v4_inline)")
    elif kernel_name == 'jax_v1_5':
        # Pure JAX, but the (N, 27, *) momentum/mass intermediate is replaced
        # by a lax.scan over the 27 stencil offsets. Per-stage only — the
        # P2G stage has a structurally different shape from solver.build_jit_frame.
        print("Using JAX P2G with lax.scan over 27 stencil offsets (jax_v1_5)")
    else:
        print("Using JAX P2G kernel")

    n = sim.n_particles
    cube_np = get_particles(n, center=list(sim.center), size=[0.5, 0.5, 0.5])
    particles = jnp.array(cube_np, dtype=jnp.float32)
    print(f"N={n}, G={sim.num_grids}")

    params = make_params(
        n_particles=n, num_grids=sim.num_grids, dt=sim.dt,
        gravity=list(sim.gravity), rho=sim.rho,
        clip_bound=sim.clip_bound, damping=sim.damping,
        center=list(sim.center), size=list(sim.size),
    )

    g = jnp.arange(params.num_grids, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(g, g, g, indexing='ij')
    grid_x = jnp.stack([gx, gy, gz], axis=-1).reshape(-1, 3)

    pre_fn, post_fn = build_boundary_fns(
        list(sim.boundary_conditions), grid_x, params.dx,
        particles, params.dt, params.p_mass,
    )

    elasticity_fn = get_constitutive(mat.elasticity)
    plasticity_fn = get_constitutive(mat.plasticity)

    timing_mode = cfg.get('timing_mode', 'per_frame')
    assert timing_mode in ('per_frame', 'per_stage'), \
        f"timing_mode must be 'per_frame' or 'per_stage', got {timing_mode!r}"

    def make_state():
        return MPMState(
            x=particles,
            v=jnp.broadcast_to(jnp.array(list(sim.initial_velocity)), (n, 3)).copy(),
            C=jnp.zeros((n, 3, 3)),
            F=jnp.tile(jnp.eye(3), (n, 1, 1)),
        )

    if kernel_name == 'cuda_fused' and timing_mode == 'per_frame':
        raise RuntimeError(
            "kernel=cuda_fused is only wired into the per-stage path. "
            "Run with timing_mode=per_stage."
        )
    if kernel_name == 'jax_v1_5' and timing_mode == 'per_frame':
        raise RuntimeError(
            "kernel=jax_v1_5 is only wired into the per-stage path. "
            "Run with timing_mode=per_stage."
        )

    if kernel_name == 'cuda_v1_inline' and timing_mode == 'per_stage':
        raise RuntimeError(
            "kernel=cuda_v1_inline is only wired into the per-frame path "
            "(by design — the whole frame compiles to one XLA program). "
            "Run with timing_mode=per_frame."
        )
    if kernel_name == 'cuda_v2_inline' and timing_mode == 'per_stage':
        raise RuntimeError(
            "kernel=cuda_v2_inline is only wired into the per-frame path "
            "(by design — the whole frame compiles to one XLA program). "
            "Run with timing_mode=per_frame."
        )

    if kernel_name == 'cuda_v3_inline' and timing_mode == 'per_stage':
        raise RuntimeError(
            "kernel=cuda_v3_inline is only wired into the per-frame path. "
            "Run with timing_mode=per_frame."
        )
    if kernel_name == 'cuda_v6_inline' and timing_mode == 'per_stage':
        raise RuntimeError(
            "kernel=cuda_v6_inline is only wired into the per-frame path "
            "(CUDA Graph capture wraps the lax.scan substep loop). "
            "Run with timing_mode=per_frame."
        )

    if kernel_name == 'cuda_v4_inline' and timing_mode == 'per_stage':
        raise RuntimeError(
            "kernel=cuda_v4_inline is only wired into the per-frame path "
            "(the JAX-side sort and FFI call both live inside lax.scan). "
            "Run with timing_mode=per_frame."
        )

    def _warmup_metrics(s):
        """Compile the per-frame metric reads so the first timed frame doesn't
        eat a one-shot trace+compile on jnp.mean / jnp.abs.max."""
        _ = float(s.x[:, 2].mean())
        _ = float(jnp.abs(s.v).max())

    # ---- Build the substep function for whichever timing_mode is selected ----
    if timing_mode == 'per_frame':
        if kernel_name == 'cuda_v1_inline':
            from mpm_jax.cuda.p2g_cuda import build_jit_frame_inline
            jit_frame = build_jit_frame_inline(
                params, elasticity_fn, plasticity_fn,
                pre_fn, post_fn, sim.steps_per_frame)

            def run_frame(s):
                return jit_frame(s)

            state = make_state()
            state = jit_frame(state); jax.block_until_ready(state.x)
            _warmup_metrics(state)
        elif kernel_name == 'cuda_v2_inline':
            from mpm_jax.cuda.p2g_cuda import build_jit_frame_v2_inline
            jit_frame = build_jit_frame_v2_inline(
                params, elasticity_fn, plasticity_fn,
                pre_fn, post_fn, sim.steps_per_frame)

            def run_frame(s):
                return jit_frame(s)

            state = make_state()
            state = jit_frame(state); jax.block_until_ready(state.x)
            _warmup_metrics(state)
        elif kernel_name == 'cuda_v3_inline':
            from mpm_jax.cuda.p2g_cuda import build_jit_frame_v3_inline
            jit_frame = build_jit_frame_v3_inline(
                params, elasticity_fn, plasticity_fn,
                pre_fn, post_fn, sim.steps_per_frame)

            def run_frame(s):
                return jit_frame(s)

            state = make_state()
            state = jit_frame(state); jax.block_until_ready(state.x)
            _warmup_metrics(state)
        elif kernel_name == 'cuda_v6_inline':
            # Same compute path as cuda_v3_inline; the CUDA-Graph capture
            # comes from XLA_FLAGS set in _maybe_enable_cuda_graphs() above.
            from mpm_jax.cuda.p2g_cuda import build_jit_frame_v3_inline
            jit_frame = build_jit_frame_v3_inline(
                params, elasticity_fn, plasticity_fn,
                pre_fn, post_fn, sim.steps_per_frame)

            def run_frame(s):
                return jit_frame(s)

            state = make_state()
            state = jit_frame(state); jax.block_until_ready(state.x)
            _warmup_metrics(state)
        elif kernel_name == 'cuda_v4_inline':
            from mpm_jax.cuda.p2g_cuda import build_jit_frame_v4_inline
            jit_frame = build_jit_frame_v4_inline(
                params, elasticity_fn, plasticity_fn,
                pre_fn, post_fn, sim.steps_per_frame)

            def run_frame(s):
                return jit_frame(s)

            state = make_state()
            state = jit_frame(state); jax.block_until_ready(state.x)
            _warmup_metrics(state)
        else:
            jit_step = build_jit_step(params, elasticity_fn, plasticity_fn,
                                      pre_fn, post_fn, p2g_fn=p2g_fn)
            jit_frame = build_jit_frame(params, elasticity_fn, plasticity_fn,
                                        pre_fn, post_fn, sim.steps_per_frame, p2g_fn=p2g_fn)

            def run_frame(s):
                return jit_frame(s)

            # Warmup: trace+compile everything we'll use in the timed region.
            state = make_state()
            state = jit_step(state); jax.block_until_ready(state.x)
            state = make_state()
            state = jit_frame(state); jax.block_until_ready(state.x)
            _warmup_metrics(state)
    else:  # per_stage
        if kernel_name == 'cuda_fused':
            from mpm_jax.cuda.p2g_cuda import make_fused_stages
            jit_p2g_stage, jit_grid_stage, jit_g2p_stage = make_fused_stages(
                params, mat.elasticity, mat.plasticity, pre_fn, post_fn)
        elif kernel_name == 'jax_v1_5':
            from mpm_jax.p2g_scan import build_jit_stages_scan
            jit_p2g_stage, jit_grid_stage, jit_g2p_stage = build_jit_stages_scan(
                params, elasticity_fn, plasticity_fn, pre_fn, post_fn)
        else:
            jit_p2g_stage, jit_grid_stage, jit_g2p_stage = build_jit_stages(
                params, elasticity_fn, plasticity_fn, pre_fn, post_fn, p2g_fn=p2g_fn)

        def run_frame(s):
            for _ in range(sim.steps_per_frame):
                grid_mv, grid_m, inter = jit_p2g_stage(s)
                grid_v = jit_grid_stage(grid_mv, grid_m)
                s = jit_g2p_stage(s, grid_v, inter)
            return s

        state = make_state()
        grid_mv, grid_m, inter = jit_p2g_stage(state)
        grid_v = jit_grid_stage(grid_mv, grid_m)
        state = jit_g2p_stage(state, grid_v, inter)
        jax.block_until_ready(state.x)
        _warmup_metrics(state)

    # ---- Timed region ----
    # Benchmark mode: dispatch every frame back-to-back with no intra-loop
    # sync. JAX queues the work on its stream; we block once at the end and
    # divide elapsed by num_frames for the average. This gives the GPU the
    # most freedom to pipeline launches.
    #
    # Non-benchmark mode (GIF rendering): we have to materialise state.x
    # to host every frame to capture the trajectory, which forces a sync
    # per frame anyway. Keep the per-frame metrics in that path.
    state = make_state()
    frames = []
    frame_metrics = []

    cuda_profiler_start()

    if bench:
        t0 = time.perf_counter()
        for _ in tqdm(range(sim.num_frames), desc='JAX'):
            state = run_frame(state)
        jax.block_until_ready(state.x)
        elapsed = time.perf_counter() - t0
    else:
        t0 = time.perf_counter()
        for _ in tqdm(range(sim.num_frames), desc='JAX'):
            frames.append(np.array(state.x))  # implicit sync via host readback
            t_frame = time.perf_counter()
            state = run_frame(state)
            jax.block_until_ready(state.x)
            frame_ms = (time.perf_counter() - t_frame) * 1000
            frame_metrics.append({
                'x_mean_z': float(state.x[:, 2].mean()),
                'v_max': float(jnp.abs(state.v).max()),
                'frame_ms': frame_ms,
                'timestep_ms': frame_ms,
            })
        elapsed = time.perf_counter() - t0

    cuda_profiler_stop()
    total_steps = sim.num_frames * sim.steps_per_frame
    avg_frame_ms = elapsed / sim.num_frames * 1000
    summary = {
        'timestep': {
            'mean_ms': avg_frame_ms,
            'std_ms': 0.0,
            'total_ms': elapsed * 1000,
            'count': sim.num_frames,
        }
    }
    return frames, elapsed, total_steps, summary, frame_metrics



# ---------------------------------------------------------------------------
# Wandb logging (all after timing is complete)
# ---------------------------------------------------------------------------

def log_results(backend, elapsed, total_steps, summary, frame_metrics, frames, cfg, export_path):
    """Log all metrics to wandb. Called only after timing is done."""
    steps_per_sec = total_steps / elapsed
    ms_per_step = elapsed / total_steps * 1000
    steps_per_frame = cfg.sim.steps_per_frame

    # Per-frame time series (stage timings + physics metrics)
    for i, fm in enumerate(frame_metrics):
        step_idx = (i + 1) * steps_per_frame
        wandb.log({k: v for k, v in fm.items()}, step=step_idx)

    # Summary scalars
    n_particles = cfg.sim.n_particles
    wandb.log({
        'summary/total_steps': total_steps,
        'summary/elapsed_s': elapsed,
        'summary/steps_per_sec': steps_per_sec,
        'summary/ms_per_step': ms_per_step,
        'summary/n_particles': n_particles,
    })

    # Per-stage breakdown table
    stage_table = wandb.Table(
        columns=["stage", "mean_ms", "std_ms", "total_ms", "count", "pct"],
    )
    total_ms = sum(s['total_ms'] for s in summary.values())
    for stage, stats in sorted(summary.items(), key=lambda x: -x[1]['total_ms']):
        pct = stats['total_ms'] / total_ms * 100 if total_ms > 0 else 0
        stage_table.add_data(stage, round(stats['mean_ms'], 4), round(stats['std_ms'], 4),
                             round(stats['total_ms'], 2), stats['count'], round(pct, 1))
        wandb.log({
            f'stage/{stage}_mean_ms': stats['mean_ms'],
            f'stage/{stage}_pct': pct,
        })
    wandb.log({'stage_breakdown': stage_table})

    # Animation
    if export_path and os.path.exists(export_path):
        wandb.log({'animation': wandb.Video(export_path, format='gif')})


# ---------------------------------------------------------------------------
# Profiler integration (nsys, ncu, jax)
# ---------------------------------------------------------------------------

_ENV_INSIDE_PROFILER = "_MPM_INSIDE_PROFILER"


def _is_inside_profiler():
    return os.environ.get(_ENV_INSIDE_PROFILER) == "1"


def _relaunch_under_profiler(profile_name, cfg):
    """Re-launch this process under nsys or ncu. Exits when done.

    The .nsys-rep / .csv is written into the Hydra run output dir, and the
    inner Python process is told to reuse the same dir so its simulate.log
    / wandb files end up next to the profile report.
    """
    import sys
    from hydra.core.hydra_config import HydraConfig

    kernel_name = cfg.get('kernel', {}).get('name', 'jax')
    N = cfg.sim.n_particles

    # Hydra ≥1.2 doesn't chdir by default — get the run output dir from the
    # HydraConfig API instead of trusting os.getcwd().
    outer_outdir = os.path.abspath(HydraConfig.get().runtime.output_dir)
    os.makedirs(outer_outdir, exist_ok=True)
    report_stem = os.path.join(outer_outdir, f"profile_{kernel_name}_N{N}")

    inner_cmd = [sys.executable] + sys.argv + [
        f"hydra.run.dir={outer_outdir}",
        "hydra.output_subdir=null",  # avoid stomping on the outer .hydra/
    ]

    if profile_name == "nsys":
        wrapper = [
            "nsys", "profile",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--trace=cuda,nvtx",
            "--stats=true",
            "--force-overwrite=true",
            "-o", report_stem,  # absolute path; nsys appends .nsys-rep
        ]
    elif profile_name == "ncu":
        wrapper = [
            "ncu",
            "--set", "full",
            "--csv",
            "--log-file", f"{report_stem}.csv",
            "--force-overwrite",
        ]
    else:
        return  # not a subprocess profiler

    env = os.environ.copy()
    env[_ENV_INSIDE_PROFILER] = "1"

    print(f"\nRe-launching under {profile_name}...")
    print(f"  {' '.join(wrapper + inner_cmd)}\n")
    result = subprocess.run(wrapper + inner_cmd, env=env)
    sys.exit(result.returncode)


def _extract_nsys_stats(cfg):
    """Extract kernel timings from the nsys .nsys-rep file and log to wandb."""
    import glob
    import io
    from hydra.core.hydra_config import HydraConfig

    outdir = os.path.abspath(HydraConfig.get().runtime.output_dir)
    candidates = sorted(
        glob.glob(os.path.join(outdir, "profile_*.nsys-rep")),
        key=os.path.getmtime, reverse=True,
    )
    if not candidates:
        print("No .nsys-rep file found.")
        return

    report_path = candidates[0]
    print(f"\nExtracting kernel stats from {report_path}...")

    try:
        result = subprocess.run(
            ["nsys", "stats", report_path,
             "--report", "cuda_gpu_kern_sum",
             "--format", "csv"],
            capture_output=True, text=True, timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("nsys stats failed.")
        return

    if result.returncode != 0:
        print(f"nsys stats error: {result.stderr[:200]}")
        return

    lines = result.stdout.strip().split("\n")
    csv_lines = [line for line in lines if "," in line and not line.startswith("Processing")]
    if not csv_lines:
        print("No kernel data in nsys report.")
        return

    csv_text = "\n".join(csv_lines)
    try:
        import pandas as pd
        df = pd.read_csv(io.StringIO(csv_text))
        print("\nCUDA Kernel Summary:")
        print(df.to_string(index=False))
        wandb.log({"nsys_kernel_summary": wandb.Table(dataframe=df)})
    except ImportError:
        wandb.log({"nsys_kernel_csv": wandb.Html(f"<pre>{csv_text}</pre>")})

    # Upload raw report as artifact
    artifact = wandb.Artifact(
        f"nsys-{cfg.get('kernel', {}).get('name', 'jax')}-N{cfg.sim.n_particles}",
        type="profile",
    )
    artifact.add_file(report_path)
    wandb.log_artifact(artifact)
    print(f"Uploaded {report_path} as wandb artifact.")


def _parse_ncu_csv(csv_path):
    """Parse the long-format ncu CSV (`ncu --set full --csv` output).

    The file is a regular CSV preceded by `==PROF==` status lines. Each
    data row is (kernel invocation, metric) -> value. We collapse to a
    per-kernel summary keyed on the demangled kernel name.

    Returns a dict: {kernel_name: {
        'launches': int,
        'duration_us_total': float,
        'duration_us_avg': float,
        'metrics': {metric_name: [values...]},  # per-launch averages
    }}
    """
    import csv
    from collections import defaultdict

    rows = []
    header = None
    with open(csv_path) as f:
        # Skip ==PROF== / ==ERROR== prelude until we find the CSV header.
        for line in f:
            stripped = line.strip()
            if stripped.startswith('"ID"') or stripped.startswith('ID,'):
                header = next(csv.reader([line]))
                break
        if header is None:
            return None
        rows = list(csv.DictReader(f, fieldnames=header))

    # Each kernel launch shows up as N rows (one per metric). Group on
    # (kernel name, ID) so we can count distinct launches.
    by_kernel = defaultdict(lambda: {'launch_ids': set(), 'metrics': defaultdict(list)})
    for row in rows:
        kname = row.get('Kernel Name') or ''
        # Trim template args / argument list for readability.
        kname_short = kname.split('(', 1)[0].strip()
        if not kname_short:
            continue
        invocation_id = row.get('ID', '')
        metric = row.get('Metric Name', '')
        value_str = (row.get('Metric Value') or '').replace(',', '').strip()
        unit = (row.get('Metric Unit') or '').strip()
        try:
            value = float(value_str)
        except ValueError:
            continue
        # Normalise duration to microseconds.
        if metric == 'gpu__time_duration.sum':
            if unit == 'ns':
                value /= 1000.0
            elif unit == 'ms':
                value *= 1000.0
            elif unit == 's':
                value *= 1e6
            elif unit != 'us':
                continue  # unknown unit, skip
        by_kernel[kname_short]['launch_ids'].add(invocation_id)
        by_kernel[kname_short]['metrics'][metric].append(value)

    out = {}
    for kname, data in by_kernel.items():
        durations = data['metrics'].get('gpu__time_duration.sum', [])
        out[kname] = {
            'launches': len(data['launch_ids']),
            'duration_us_total': sum(durations),
            'duration_us_avg': (sum(durations) / len(durations)) if durations else 0.0,
            'metrics': {m: vals for m, vals in data['metrics'].items()},
        }
    return out


def _print_ncu_summary(summary):
    """Pretty-print the dict returned by _parse_ncu_csv."""
    if not summary:
        print("(ncu summary is empty)")
        return
    total_us = sum(v['duration_us_total'] for v in summary.values()) or 1.0

    # Headline columns. Show a couple of widely-available throughput metrics
    # if present in the CSV.
    print(f"\n{'Kernel':<40s} {'Launches':>9s} {'Avg µs':>10s} {'Total ms':>10s} {'% time':>7s}  "
          f"{'SM%':>6s}  {'Mem%':>6s}  {'Occ%':>6s}")
    print("-" * 110)

    def _avg(vs):
        return sum(vs) / len(vs) if vs else float('nan')

    rows = sorted(summary.items(), key=lambda kv: -kv[1]['duration_us_total'])
    for kname, d in rows:
        sm_pct = _avg(d['metrics'].get('sm__throughput.avg.pct_of_peak_sustained_elapsed', []))
        mem_pct = _avg(d['metrics'].get('gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed', []))
        occ_pct = _avg(d['metrics'].get('sm__warps_active.avg.pct_of_peak_sustained_active', []))

        def _fmt(x):
            return f"{x:>5.1f}%" if x == x else "    —"  # NaN-safe

        print(f"{kname[:40]:<40s} {d['launches']:>9d} "
              f"{d['duration_us_avg']:>10.2f} {d['duration_us_total']/1000.0:>10.3f} "
              f"{100.0 * d['duration_us_total']/total_us:>6.1f}%  "
              f"{_fmt(sm_pct)}  {_fmt(mem_pct)}  {_fmt(occ_pct)}")


def _extract_ncu_stats(cfg):
    """Extract Nsight Compute CSV results, summarise, and log to wandb."""
    import glob
    from hydra.core.hydra_config import HydraConfig

    outdir = os.path.abspath(HydraConfig.get().runtime.output_dir)
    candidates = sorted(
        glob.glob(os.path.join(outdir, "profile_*.csv")),
        key=os.path.getmtime, reverse=True,
    )
    if not candidates:
        print("No ncu CSV file found.")
        return

    csv_path = candidates[0]
    print(f"\nLoading ncu results from {csv_path}...")

    summary = _parse_ncu_csv(csv_path)
    if not summary:
        # Either no CSV header (None) or no kernel data (empty dict). Both
        # usually mean ncu didn't get to run kernels — most commonly the
        # driver-side perm gate (ERR_NVGPUCTRPERM, RmProfilingAdminOnly=1).
        with open(csv_path) as f:
            head = "".join(line for line, _ in zip(f, range(8)))
        if "ERR_NVGPUCTRPERM" in head:
            print("ncu collected no kernel data: the driver requires admin perf-counter access.")
            print("On the host, reload the nvidia module with NVreg_RestrictProfilingToAdminUsers=0,")
            print("or run inside an environment where the driver isn't locked down.")
        else:
            print("ncu produced no kernel data. First lines of the report:")
            print("\n".join("  " + line.rstrip() for line in head.splitlines() if line.strip()))
    else:
        _print_ncu_summary(summary)
        # Log to wandb as a flat table.
        try:
            table = wandb.Table(columns=["kernel", "launches", "avg_us", "total_ms", "pct_time"])
            total_us = sum(v['duration_us_total'] for v in summary.values()) or 1.0
            for kname, d in sorted(summary.items(), key=lambda kv: -kv[1]['duration_us_total']):
                table.add_data(kname, d['launches'],
                               round(d['duration_us_avg'], 2),
                               round(d['duration_us_total'] / 1000.0, 3),
                               round(100.0 * d['duration_us_total'] / total_us, 1))
            wandb.log({"ncu_kernel_summary": table})
        except Exception as e:
            print(f"(wandb table upload failed: {e})")

    artifact = wandb.Artifact(
        f"ncu-{cfg.get('kernel', {}).get('name', 'jax')}-N{cfg.sim.n_particles}",
        type="profile",
    )
    artifact.add_file(csv_path)
    wandb.log_artifact(artifact)
    print(f"Uploaded {csv_path} as wandb artifact.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    profile_name = cfg.get('profile', {}).get('name', 'none')

    # If nsys/ncu requested and we're not already inside the profiler,
    # re-launch this process wrapped in the profiler.
    if profile_name in ('nsys', 'ncu') and not _is_inside_profiler():
        _relaunch_under_profiler(profile_name, cfg)
        return  # unreachable — _relaunch calls sys.exit

    kernel_name = cfg.get('kernel', {}).get('name', 'jax')
    N = cfg.sim.n_particles
    G = cfg.sim.num_grids

    # CUDA Graphs toggle (v6) must happen before any `import jax` in this
    # process — including the profile=jax branch a few lines down.
    _maybe_enable_cuda_graphs(kernel_name)

    # Init wandb
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(
        project="MPM-CudaJAX",
        name=f"jax_{kernel_name}_N{N}_G{G}",
        config=wandb_cfg,
        tags=[kernel_name, f"N{N}", f"G{G}", profile_name],
    )

    # JAX profiler (in-process, writes TensorBoard trace)
    jax_trace_dir = None
    if profile_name == 'jax':
        import jax
        from hydra.core.hydra_config import HydraConfig
        # Same fix as the nsys/ncu path: Hydra >=1.2 doesn't chdir.
        run_dir = os.path.abspath(HydraConfig.get().runtime.output_dir)
        jax_trace_dir = os.path.join(run_dir, "jax_trace")
        jax.profiler.start_trace(jax_trace_dir)
        print(f"JAX profiler started -> {jax_trace_dir}")

    # Run simulation (timing-critical — no wandb calls inside)
    frames, elapsed, total_steps, summary, frame_metrics = run_jax(cfg)

    # Stop JAX profiler
    if profile_name == 'jax':
        import jax
        jax.profiler.stop_trace()
        print(f"JAX trace saved to {jax_trace_dir}")

    # Print timing summary
    steps_per_sec = total_steps / elapsed
    ms_per_step = elapsed / total_steps * 1000
    print(f"\njax ({kernel_name}): {total_steps} steps in {elapsed:.2f}s ({steps_per_sec:.1f} steps/s, {ms_per_step:.2f} ms/step)")

    total_ms = sum(s['total_ms'] for s in summary.values())
    print(f"\nPer-stage timing (per frame, {cfg.sim.steps_per_frame} substeps each):")
    for stage, stats in sorted(summary.items(), key=lambda x: -x[1]['total_ms']):
        pct = stats['total_ms'] / total_ms * 100 if total_ms > 0 else 0
        print(f"  {stage:15s}: {stats['mean_ms']:8.3f} ms/frame ({pct:5.1f}%  std={stats['std_ms']:.3f}  n={stats['count']})")

    # Render GIF (skip in benchmark mode)
    export_path = None
    if not cfg.get('benchmark', False) and frames:
        orig_cwd = hydra.utils.get_original_cwd()
        output_dir = os.path.join(orig_cwd, cfg.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        export_path = os.path.join(output_dir, f"{cfg.tag}_{kernel_name}.gif")
        print(f"\nRendering to {export_path}...")
        visualize_frames(frames, export_path, size=[1, 1, 1], c=cfg.material.color)
    elif cfg.get('benchmark', False):
        print("\nBenchmark mode: skipping GIF rendering.")

    # Dump a small results.json into the Hydra run dir so multirun callbacks
    # (and any post-hoc aggregation) can pick up the per-run numbers without
    # touching wandb. One file per run, fixed shape.
    import json
    from hydra.core.hydra_config import HydraConfig
    run_dir = os.path.abspath(HydraConfig.get().runtime.output_dir)
    mat_name = cfg.get('material', {}).get('elasticity', {}).get('name', None) \
        or cfg.get('material', {}).get('name', 'unknown')
    results = {
        'kernel': kernel_name,
        'material_elasticity': mat_name,
        'n_particles': int(cfg.sim.n_particles),
        'num_grids': int(cfg.sim.num_grids),
        'timing_mode': cfg.get('timing_mode', 'per_frame'),
        'num_frames': int(cfg.sim.num_frames),
        'steps_per_frame': int(cfg.sim.steps_per_frame),
        'total_steps': int(total_steps),
        'elapsed_s': float(elapsed),
        'ms_per_step': float(ms_per_step),
        'steps_per_sec': float(steps_per_sec),
    }
    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Log timing results to wandb
    log_results(kernel_name, elapsed, total_steps, summary, frame_metrics, frames, cfg, export_path)

    # Extract and log profiler results
    if profile_name == 'nsys':
        _extract_nsys_stats(cfg)
    elif profile_name == 'ncu':
        _extract_ncu_stats(cfg)
    elif profile_name == 'jax' and jax_trace_dir:
        artifact = wandb.Artifact(f"jax-trace-{kernel_name}-N{N}", type="profile")
        artifact.add_dir(jax_trace_dir)
        wandb.log_artifact(artifact)
        print("Uploaded JAX trace as wandb artifact.")

    wandb.finish()


if __name__ == "__main__":
    main()
