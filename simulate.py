import pyvista as pv
import os
import time
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig


def get_particles(n_particles, center, size):
    """Sample n_particles uniformly in a box."""
    start = np.array(center, dtype=np.float32) - np.array(size, dtype=np.float32) / 2
    end = np.array(center, dtype=np.float32) + np.array(size, dtype=np.float32) / 2
    rng = np.random.RandomState(42)
    return (start + rng.rand(n_particles, 3).astype(np.float32) * (end - start)).astype(np.float32, copy=False)


def visualize_frames(frames, export_path, center=[0.5, 0.5, 0.5],
                     size=[2.0, 2.0, 2.0], c='blue', s=20, fps=30):
    try:
        # Need to start xvfb for pyvista offscreen rendering to work without a display
        # But we can also set the VTK render window to offscreen before plotting
        pv.start_xvfb()
    except Exception:
        pass

    plotter = pv.Plotter(off_screen=True)
    plotter.open_gif(export_path)

    # Initialize point cloud
    points = frames[0]
    cloud = pv.PolyData(points)
    plotter.add_mesh(cloud, color=c, point_size=s, render_points_as_spheres=True)

    # Add bounding box
    bounds = [
        center[0] - size[0]/2, center[0] + size[0]/2,
        center[1] - size[1]/2, center[1] + size[1]/2,
        center[2] - size[2]/2, center[2] + size[2]/2
    ]
    box = pv.Box(bounds)
    plotter.add_mesh(box, style='wireframe', color='black')

    plotter.camera_position = 'iso'
    plotter.show(auto_close=False)

    for i in range(len(frames)):
        cloud.points = frames[i]
        plotter.add_text(f"Frame {i}", position="upper_left", name="time_label")
        plotter.write_frame()

    plotter.close()


# ---------------------------------------------------------------------------
# Backend-specific runners
# ---------------------------------------------------------------------------

def run_warp_bonus(cfg: DictConfig):
    """Run the pure-Warp graph-captured tiled prototype."""
    import warp as wp
    from mpm_jax.warp_bonus import WarpBonusSimulator

    sim = cfg.sim
    kernel_name = cfg.get('kernel', {}).get('name', 'warp_bonus_graph')
    indexed_sort = kernel_name == 'warp_bonus_v2_graph'

    elasticity = cfg.get('material', {}).get('elasticity', {}).get('name', None)
    plasticity = cfg.get('material', {}).get('plasticity', {}).get('name', None)
    if elasticity != 'CorotatedElasticityJacobi' or plasticity != 'IdentityPlasticity':
        raise RuntimeError(
            f"kernel={kernel_name} currently supports material=jelly_jacobi "
            "only: CorotatedElasticityJacobi + IdentityPlasticity."
        )

    if int(sim.num_grids) % 2 != 0:
        raise RuntimeError(f"kernel={kernel_name} requires sim.num_grids divisible by 2.")

    n = int(sim.n_particles)
    cube_np = get_particles(n, center=list(sim.center), size=[0.5, 0.5, 0.5])
    precompute_stress = not (indexed_sort and n >= 150_000_000)
    if indexed_sort:
        print("Using pure NVIDIA Warp graph-captured indexed super-cell tile MPM step (warp_bonus_v2_graph)")
        if not precompute_stress:
            print("warp_bonus_v2_graph: disabling stress precompute to fit large particle count in GPU memory")
    else:
        print("Using pure NVIDIA Warp graph-captured super-cell tile MPM step (warp_bonus_graph)")
    print(f"N={n}, G={sim.num_grids}")

    runner = WarpBonusSimulator(cube_np, cfg, indexed_sort=indexed_sort, precompute_stress=precompute_stress)
    runner.warmup()

    frames = []
    frame_metrics = []
    total_steps = int(sim.num_frames) * int(sim.steps_per_frame)

    if cfg.get('benchmark', False):
        result = runner.run_frames(int(sim.num_frames))
        elapsed = result.elapsed_s
    else:
        t0 = time.perf_counter()
        for frame in tqdm(range(sim.num_frames), desc='Warp'):
            t_frame = time.perf_counter()
            runner.launch_frame()
            wp.synchronize_device(runner.device)
            frame_ms = (time.perf_counter() - t_frame) * 1000
            x_np = runner.x.numpy()
            v_np = runner.v.numpy()
            frames.append(x_np)
            frame_metrics.append({
                'x_mean_z': float(x_np[:, 2].mean()),
                'v_max': float(np.abs(v_np).max()),
                'frame_ms': frame_ms,
                'timestep_ms': frame_ms,
            })
        elapsed = time.perf_counter() - t0

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

    import warp as wp
    wp.init()
    import jax
    import jax.numpy as jnp
    from mpm_jax.solver import (
        MPMState, make_params, build_jit_step, build_jit_frame,
    )
    from mpm_jax.constitutive import get_constitutive
    from mpm_jax.boundary import build_boundary_fns

    sim = cfg.sim
    mat = cfg.material
    bench = cfg.get('benchmark', False)

    p2g_fn = None         # None = default JAX implementation

    if kernel_name in {'cuda_v1', 'cuda_v2', 'cuda_v4'}:
        raise RuntimeError(
            f"kernel={kernel_name} has been removed. Use an inline CUDA kernel "
            "such as cuda_v3_inline, or use the pure JAX kernels."
        )
    elif kernel_name == 'cuda_fused':
        raise RuntimeError(
            "kernel=cuda_fused is deprecated in the CLI path. Use one of the "
            "fully-jitted kernels and profile with profile=jax."
        )
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
    elif kernel_name == 'cuda_v2_fori_inline':
        print("Using CUDA inline P2G kernel with warp shuffle + JAX lax.fori_loop (cuda_v2_fori_inline)")
    elif kernel_name == 'cuda_v3_inline':
        # cuda_v3_inline: cuda_v1_inline + Morton (Z-order) spatial sort of
        # particles before each substep + warp-shuffle atomic coalescing.
        # Hypothesis: spatially-close particles in the same warp share more
        # stencil-node targets, so `__match_any_sync` collapses 4-8x of the
        # atomics into one. Tradeoff: an argsort per substep (O(N log N)).
        print("Using CUDA inline P2G kernel with Morton sort + warp shuffle (cuda_v3_inline)")
    elif kernel_name == 'cuda_v3_fori_inline':
        print("Using CUDA inline P2G kernel with Morton sort + warp shuffle + JAX lax.fori_loop (cuda_v3_fori_inline)")
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
    elif kernel_name == 'warp_v1_inline':
        # warp_v1_inline: one Warp thread per particle, inline weights +
        # 27-stencil atomic scatter, called inside JAX JIT through Warp's
        # experimental JAX FFI.
        print("Using NVIDIA Warp inline P2G kernel via JAX FFI (warp_v1_inline)")
    elif kernel_name == 'warp_v2_tile':
        # warp_v2_tile: launches a Warp tiled kernel from inside JAX JIT via
        # jax_callable. Each block tile-loads 64 particles into shared storage
        # before the per-lane 27-stencil atomic scatter.
        print("Using NVIDIA Warp tiled P2G kernel via JAX FFI (warp_v2_tile)")
    elif kernel_name == 'warp_v3_supercell_tile':
        # warp_v3_supercell_tile: cell-owned Warp tile path. Particles are
        # sorted by home super-cell; one Warp block accumulates a 4^3 shared
        # grid-node tile and flushes it once to global memory.
        print("Using NVIDIA Warp super-cell tile P2G kernel via JAX FFI (warp_v3_supercell_tile)")
    elif kernel_name == 'jax_v1_5':
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

    def make_state():
        return MPMState(
            x=particles,
            v=jnp.broadcast_to(jnp.array(list(sim.initial_velocity)), (n, 3)).copy(),
            C=jnp.zeros((n, 3, 3)),
            F=jnp.tile(jnp.eye(3), (n, 1, 1)),
        )

    def _warmup_metrics(s):
        """Compile the per-frame metric reads so the first timed frame doesn't
        eat a one-shot trace+compile on jnp.mean / jnp.abs.max."""
        _ = float(s.x[:, 2].mean())
        _ = float(jnp.abs(s.v).max())

    with jax.profiler.TraceAnnotation("build_jit_frame", kernel=kernel_name):
        if kernel_name == 'cuda_v1_inline':
            from mpm_jax.cuda.p2g_cuda import build_jit_frame_inline
            jit_frame = build_jit_frame_inline(
                params, elasticity_fn, plasticity_fn,
                pre_fn, post_fn, sim.steps_per_frame)
        elif kernel_name in ('cuda_v2_inline', 'cuda_v2_fori_inline'):
            from mpm_jax.cuda.p2g_cuda import build_jit_frame_v2_inline
            jit_frame = build_jit_frame_v2_inline(
                params, elasticity_fn, plasticity_fn,
                pre_fn, post_fn, sim.steps_per_frame,
                loop_kind='fori' if kernel_name == 'cuda_v2_fori_inline' else 'python')
        elif kernel_name in ('cuda_v3_inline', 'cuda_v3_fori_inline', 'cuda_v6_inline'):
            from mpm_jax.cuda.p2g_cuda import build_jit_frame_v3_inline
            jit_frame = build_jit_frame_v3_inline(
                params, elasticity_fn, plasticity_fn,
                pre_fn, post_fn, sim.steps_per_frame,
                loop_kind='fori' if kernel_name == 'cuda_v3_fori_inline' else 'python')
        elif kernel_name == 'cuda_v4_inline':
            from mpm_jax.cuda.p2g_cuda import build_jit_frame_v4_inline
            jit_frame = build_jit_frame_v4_inline(
                params, elasticity_fn, plasticity_fn,
                pre_fn, post_fn, sim.steps_per_frame)
        elif kernel_name == 'warp_v1_inline':
            from mpm_jax.warp_p2g import build_jit_frame_warp_inline
            jit_frame = build_jit_frame_warp_inline(
                params, elasticity_fn, plasticity_fn,
                pre_fn, post_fn, sim.steps_per_frame)
        elif kernel_name == 'warp_v2_tile':
            from mpm_jax.warp_p2g import build_jit_frame_warp_tile
            jit_frame = build_jit_frame_warp_tile(
                params, elasticity_fn, plasticity_fn,
                pre_fn, post_fn, sim.steps_per_frame)
        elif kernel_name == 'warp_v3_supercell_tile':
            from mpm_jax.warp_p2g import build_jit_frame_warp_supercell_tile
            jit_frame = build_jit_frame_warp_supercell_tile(
                params, elasticity_fn, plasticity_fn,
                pre_fn, post_fn, sim.steps_per_frame)
        elif kernel_name == 'jax_v1_5':
            from mpm_jax.p2g_scan import build_jit_frame_scan
            jit_frame = build_jit_frame_scan(
                params, elasticity_fn, plasticity_fn,
                pre_fn, post_fn, sim.steps_per_frame)
        else:
            jit_step = build_jit_step(params, elasticity_fn, plasticity_fn,
                                      pre_fn, post_fn, p2g_fn=p2g_fn)
            jit_frame = build_jit_frame(params, elasticity_fn, plasticity_fn,
                                        pre_fn, post_fn, sim.steps_per_frame, p2g_fn=p2g_fn)

    def run_frame(s):
        return jit_frame(s)

    with jax.profiler.TraceAnnotation("warmup", kernel=kernel_name):
        state = make_state()
        if kernel_name not in (
            'cuda_v1_inline', 'cuda_v2_inline', 'cuda_v2_fori_inline',
            'cuda_v3_inline', 'cuda_v3_fori_inline',
            'cuda_v4_inline', 'cuda_v6_inline', 'jax_v1_5',
            'warp_v1_inline', 'warp_v2_tile', 'warp_v3_supercell_tile',
        ):
            state = jit_step(state)
            jax.block_until_ready(state.x)
        state = make_state()
        state = jit_frame(state)
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

    if bench:
        with jax.profiler.TraceAnnotation("benchmark", kernel=kernel_name):
            t0 = time.perf_counter()
            for frame in tqdm(range(sim.num_frames), desc='JAX'):
                with jax.profiler.StepTraceAnnotation("frame", step_num=frame):
                    state = run_frame(state)
            jax.block_until_ready(state.x)
            elapsed = time.perf_counter() - t0
    else:
        # Initialize Warp HashGrid for bookkeeping proof-of-concept
        # HashGrid will build an acceleration structure around the JAX-computed positions
        grid = wp.HashGrid(dim_x=sim.num_grids, dim_y=sim.num_grids, dim_z=sim.num_grids)
        t0 = time.perf_counter()
        with jax.profiler.TraceAnnotation("render_loop", kernel=kernel_name):
            for frame in tqdm(range(sim.num_frames), desc='JAX'):
                with jax.profiler.StepTraceAnnotation("frame", step_num=frame):
                    # Bookkeeping with Warp: copy jnp array into a wp array.
                    # Zero-copy via DLPack since both are on the GPU. Fallback to CPU if warp GPU not init.
                    try:
                        # Use standard __dlpack__ protocol since JAX arrays support it natively.
                        wp_x = wp.from_dlpack(state.x)
                        grid.build(wp_x, radius=float(params.dx))
                        frames.append(wp_x.numpy())
                    except Exception:
                        # Fallback if no GPU for warp (e.g. CI environments).
                        frames.append(np.array(state.x))

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
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    profile_name = cfg.get('profile', {}).get('name', 'none')

    if profile_name not in ('none', 'jax'):
        raise RuntimeError(
            f"Unsupported profile={profile_name!r}. Only profile=none and "
            "profile=jax are supported."
        )

    kernel_name = cfg.get('kernel', {}).get('name', 'jax')
    is_warp_bonus = kernel_name in {'warp_bonus_graph', 'warp_bonus_v2_graph'}

    # CUDA Graphs toggle (v6) must happen before any `import jax` in this
    # process — including the profile=jax branch a few lines down.
    _maybe_enable_cuda_graphs(kernel_name)

    if is_warp_bonus and profile_name == 'jax':
        raise RuntimeError(f"kernel={kernel_name} is pure Warp and does not emit a JAX trace.")

    # JAX profiler (in-process, writes TensorBoard trace)
    jax_trace_dir = None
    if profile_name == 'jax':
        import jax
        from hydra.core.hydra_config import HydraConfig
        # Hydra >=1.2 doesn't chdir, so use the output dir from HydraConfig.
        run_dir = os.path.abspath(HydraConfig.get().runtime.output_dir)
        jax_trace_dir = os.path.join(run_dir, "jax_trace")
        jax.profiler.start_trace(jax_trace_dir)
        print(f"JAX profiler started -> {jax_trace_dir}")

    # Run simulation.
    if is_warp_bonus:
        frames, elapsed, total_steps, summary, _frame_metrics = run_warp_bonus(cfg)
    else:
        frames, elapsed, total_steps, summary, _frame_metrics = run_jax(cfg)

    # Stop JAX profiler
    if profile_name == 'jax':
        import jax
        jax.profiler.stop_trace()
        print(f"JAX trace saved to {jax_trace_dir}")

    # Print timing summary
    steps_per_sec = total_steps / elapsed
    ms_per_step = elapsed / total_steps * 1000
    backend_label = "warp" if is_warp_bonus else "jax"
    print(f"\n{backend_label} ({kernel_name}): {total_steps} steps in {elapsed:.2f}s ({steps_per_sec:.1f} steps/s, {ms_per_step:.2f} ms/step)")

    total_ms = sum(s['total_ms'] for s in summary.values())
    print(f"\nWall-clock timing (per frame, {cfg.sim.steps_per_frame} substeps each):")
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
    # and post-hoc aggregation can pick up the per-run numbers. One file per
    # run, fixed shape.
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
        'num_frames': int(cfg.sim.num_frames),
        'steps_per_frame': int(cfg.sim.steps_per_frame),
        'total_steps': int(total_steps),
        'elapsed_s': float(elapsed),
        'ms_per_step': float(ms_per_step),
        'steps_per_sec': float(steps_per_sec),
    }
    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
