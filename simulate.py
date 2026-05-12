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

def run_jax(cfg: DictConfig):
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
    kernel_name = cfg.get('kernel', {}).get('name', 'jax')

    # Build p2g_fn based on kernel config
    p2g_fn = None  # None = default JAX implementation
    if kernel_name == 'cuda_v1':
        from mpm_jax.cuda.p2g_cuda import make_cuda_p2g
        p2g_fn = make_cuda_p2g(sim.num_grids, kernel='scatter')
        if p2g_fn is None:
            raise RuntimeError(
                "kernel=cuda_v1 requested but CUDA kernel failed to compile/register. "
                "Check nvcc is on PATH and module load gcc is done."
            )
        print("Using CUDA P2G scatter kernel (v1)")
    elif kernel_name == 'cuda_v4':
        from mpm_jax.cuda.p2g_cuda import make_cuda_p2g
        p2g_fn = make_cuda_p2g(sim.num_grids, kernel='smem')
        if p2g_fn is None:
            raise RuntimeError(
                "kernel=cuda_v4 requested but CUDA kernel failed to compile/register."
            )
        print("Using CUDA P2G shared-memory scatter kernel (v4)")
    elif kernel_name == 'cuda_v3':
        from mpm_jax.cuda.p2g_cuda import make_cuda_p2g
        p2g_fn = make_cuda_p2g(sim.num_grids, kernel='warp')
        if p2g_fn is None:
            raise RuntimeError(
                "kernel=cuda_v3 requested but CUDA kernel failed to compile/register. "
                "Check nvcc is on PATH and module load gcc is done."
            )
        print("Using CUDA P2G warp-reduction scatter kernel (v3)")
    elif kernel_name == 'cuda_v2':
        from mpm_jax.cuda.p2g_cuda import is_available
        if not is_available('fused'):
            raise RuntimeError(
                "kernel=cuda_v2 requested but CUDA kernel failed to compile/register. "
                "Check nvcc is on PATH and module load gcc is done."
            )
        print("Using CUDA fused P2G kernel (v2)")
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

    if timing_mode == 'per_frame':
        # ---- monolithic JIT'd frame ----
        jit_step = build_jit_step(params, elasticity_fn, plasticity_fn,
                                  pre_fn, post_fn, p2g_fn=p2g_fn)
        jit_frame = build_jit_frame(params, elasticity_fn, plasticity_fn,
                                    pre_fn, post_fn, sim.steps_per_frame, p2g_fn=p2g_fn)

        # Warmup
        state = make_state()
        state = jit_step(state)
        jax.block_until_ready(state.x)
        state = make_state()
        state = jit_frame(state)
        jax.block_until_ready(state.x)

        state = make_state()
        timer = StageTimer()
        frames = []
        frame_metrics = []
        frame_timings = []

        cuda_profiler_start()
        t0 = time.perf_counter()

        for frame in tqdm(range(sim.num_frames), desc='JAX'):
            if not bench:
                frames.append(np.array(state.x))

            timer.start('timestep')
            state = jit_frame(state)
            jax.block_until_ready(state.x)
            timer.stop()

            ft = timer.flush_frame()
            frame_ms = sum(ft.values())
            frame_timings.append(ft)
            frame_metrics.append({
                'x_mean_z': float(state.x[:, 2].mean()),
                'v_max': float(jnp.abs(state.v).max()),
                'frame_ms': frame_ms,
                **{f'{k}_ms': v for k, v in ft.items()},
            })

        elapsed = time.perf_counter() - t0
        cuda_profiler_stop()
        total_steps = sim.num_frames * sim.steps_per_frame
        return frames, elapsed, total_steps, timer.summary_from_frames(frame_timings), frame_metrics

    # ---- per-stage path: Python loop over substeps, sync between stages ----
    jit_p2g_stage, jit_grid_stage, jit_g2p_stage = build_jit_stages(
        params, elasticity_fn, plasticity_fn, pre_fn, post_fn, p2g_fn=p2g_fn)

    # Warmup each stage
    state = make_state()
    grid_mv, grid_m, inter = jit_p2g_stage(state)
    jax.block_until_ready(grid_mv)
    grid_v = jit_grid_stage(grid_mv, grid_m)
    jax.block_until_ready(grid_v)
    state = jit_g2p_stage(state, grid_v, inter)
    jax.block_until_ready(state.x)

    state = make_state()
    timer = StageTimer()
    frames = []
    frame_metrics = []
    frame_timings = []

    cuda_profiler_start()
    t0 = time.perf_counter()

    for frame in tqdm(range(sim.num_frames), desc='JAX'):
        if not bench:
            frames.append(np.array(state.x))

        for _ in range(sim.steps_per_frame):
            timer.start('p2g')
            grid_mv, grid_m, inter = jit_p2g_stage(state)
            jax.block_until_ready(grid_mv)
            timer.stop()

            timer.start('grid_update')
            grid_v = jit_grid_stage(grid_mv, grid_m)
            jax.block_until_ready(grid_v)
            timer.stop()

            timer.start('g2p')
            state = jit_g2p_stage(state, grid_v, inter)
            jax.block_until_ready(state.x)
            timer.stop()

        ft = timer.flush_frame()
        frame_ms = sum(ft.values())
        frame_timings.append(ft)
        frame_metrics.append({
            'x_mean_z': float(state.x[:, 2].mean()),
            'v_max': float(jnp.abs(state.v).max()),
            'frame_ms': frame_ms,
            **{f'{k}_ms': v for k, v in ft.items()},
        })

    elapsed = time.perf_counter() - t0
    cuda_profiler_stop()
    total_steps = sim.num_frames * sim.steps_per_frame
    return frames, elapsed, total_steps, timer.summary_from_frames(frame_timings), frame_metrics



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
    """Re-launch this process under nsys or ncu. Exits when done."""
    import sys

    kernel_name = cfg.get('kernel', {}).get('name', 'jax')
    N = cfg.sim.n_particles
    report_name = f"profile_{kernel_name}_N{N}"

    # Build the inner command (same args, but with env marker)
    inner_cmd = [sys.executable] + sys.argv

    if profile_name == "nsys":
        wrapper = [
            "nsys", "profile",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--trace=cuda,nvtx",
            "--stats=true",
            "--force-overwrite=true",
            "-o", report_name,
        ]
    elif profile_name == "ncu":
        wrapper = [
            "ncu",
            "--set", "full",
            "--csv",
            "--log-file", f"{report_name}.csv",
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

    candidates = sorted(glob.glob("profile_*.nsys-rep") + glob.glob("*.nsys-rep"),
                        key=os.path.getmtime, reverse=True)
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


def _extract_ncu_stats(cfg):
    """Extract Nsight Compute CSV results and log to wandb."""
    import glob

    candidates = sorted(glob.glob("profile_*.csv"), key=os.path.getmtime, reverse=True)
    if not candidates:
        print("No ncu CSV file found.")
        return

    csv_path = candidates[0]
    print(f"\nLoading ncu results from {csv_path}...")

    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(df.to_string(index=False))
        wandb.log({"ncu_kernel_metrics": wandb.Table(dataframe=df)})
    except Exception as e:
        print(f"Failed to parse ncu CSV: {e}")
        # Upload raw file anyway
        pass

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
        jax_trace_dir = os.path.join(os.getcwd(), "jax_trace")
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
