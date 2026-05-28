"""Hydra-driven Nsight Python profiler for MPM kernel phases and sweeps."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

from simulate import get_particles

_UNSUPPORTED_ANALYZE_CONFIG_KEYS = {"configs", "derive_metric", "combine_kernel_metrics"}
_SCRIPT_NSIGHT_KEYS = {"phase", "include_step_total", "write_json", "plot", "configs", "analyze"}


def _require_nsight():
    try:
        import nsight
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "nsight-python is not installed. Run `pixi install -e gpu` after "
            "the latest pyproject update, or `pixi run -e gpu python -m pip "
            "install nsight-python` for a one-off local install."
        ) from exc
    return nsight


def _warp_bonus_sim(cfg: DictConfig):
    import warp as wp

    from mpm_jax.warp_bonus import WarpBonusSimulator

    kernel_name = cfg.get("kernel", {}).get("name", "warp_bonus_graph")
    indexed_sort = kernel_name == "warp_bonus_v2_graph"
    n = int(cfg.sim.n_particles)
    precompute_stress = not (indexed_sort and n >= 150_000_000)
    particles = get_particles(n, center=list(cfg.sim.center), size=[0.5, 0.5, 0.5])
    sim = WarpBonusSimulator(
        particles,
        cfg,
        indexed_sort=indexed_sort,
        precompute_stress=precompute_stress,
    )

    sim._substep()
    wp.synchronize_device(sim.device)
    return sim


def _warp_bonus_p2g_runner(cfg: DictConfig):
    import warp as wp

    from mpm_jax import warp_bonus as wb

    sim = _warp_bonus_sim(cfg)
    indexed_sort = sim.indexed_sort

    def run_p2g_once():
        wp.launch(wb._zero_float_kernel, dim=sim.G3, inputs=[sim.grid_m], device=sim.device)
        wp.launch(wb._zero_vec3_kernel, dim=sim.G3, inputs=[sim.grid_mv], device=sim.device)
        if indexed_sort:
            if sim.precompute_stress:
                wp.launch(
                    wb._compute_stress_kernel,
                    dim=sim.n,
                    inputs=[sim.F, sim.stress, sim.mu, sim.la],
                    device=sim.device,
                )
                p2g_inputs = [
                    sim.ids, sim.x, sim.v, sim.C, sim.stress, sim.cell_start,
                    sim.G, sim.dt, sim.vol, sim.p_mass, sim.inv_dx, sim.dx,
                    sim.grid_mv, sim.grid_m,
                ]
            else:
                p2g_inputs = [
                    sim.ids, sim.x, sim.v, sim.C, sim.F, sim.cell_start,
                    sim.G, sim.dt, sim.vol, sim.p_mass, sim.inv_dx, sim.dx,
                    sim.mu, sim.la, sim.grid_mv, sim.grid_m,
                ]
        else:
            p2g_inputs = [
                sim.xs, sim.vs, sim.Cs, sim.Fs, sim.cell_start,
                sim.G, sim.dt, sim.vol, sim.p_mass, sim.inv_dx, sim.dx,
                sim.mu, sim.la, sim.grid_mv, sim.grid_m,
            ]

        wp.launch_tiled(
            sim.p2g_kernel,
            dim=[sim.Gs3],
            inputs=p2g_inputs,
            block_dim=sim.tile_size,
            device=sim.device,
        )
        wp.synchronize_device(sim.device)

    return run_p2g_once


def _warp_bonus_step_runner(cfg: DictConfig, nsight):
    import warp as wp

    from mpm_jax import warp_bonus as wb

    sim = _warp_bonus_sim(cfg)
    total_sim = _warp_bonus_sim(cfg) if bool(cfg.nsight.get("include_step_total", True)) else None

    def _annotated_substep():
        if total_sim is not None:
            with nsight.annotate("step"):
                total_sim._substep()
            wp.synchronize_device(total_sim.device)

        with nsight.annotate("bin"):
            wp.launch(wb._zero_int_kernel, dim=sim.Gs3, inputs=[sim.counts], device=sim.device)
            wp.launch(
                wb._count_supercells_kernel,
                dim=sim.n,
                inputs=[sim.x, sim.counts, sim.G, sim.inv_dx, sim.super_cell_width],
                device=sim.device,
            )
            wp.utils.array_scan(sim.counts, sim.prefix, inclusive=True)
            wp.launch(
                wb._prefix_to_cell_start_kernel,
                dim=sim.Gs3,
                inputs=[sim.prefix, sim.cell_start],
                device=sim.device,
            )
            wp.launch(wb._zero_int_kernel, dim=sim.Gs3, inputs=[sim.cursor], device=sim.device)
            if sim.indexed_sort:
                wp.launch(
                    wb._scatter_supercell_ids_kernel,
                    dim=sim.n,
                    inputs=[
                        sim.x, sim.cell_start, sim.cursor, sim.ids,
                        sim.G, sim.inv_dx, sim.super_cell_width,
                    ],
                    device=sim.device,
                )
            else:
                wp.launch(
                    wb._scatter_supercell_order_kernel,
                    dim=sim.n,
                    inputs=[
                        sim.x, sim.v, sim.C, sim.F,
                        sim.cell_start, sim.cursor,
                        sim.xs, sim.vs, sim.Cs, sim.Fs,
                        sim.G, sim.inv_dx, sim.super_cell_width,
                    ],
                    device=sim.device,
                )

        with nsight.annotate("zero_grid"):
            wp.launch(wb._zero_float_kernel, dim=sim.G3, inputs=[sim.grid_m], device=sim.device)
            wp.launch(wb._zero_vec3_kernel, dim=sim.G3, inputs=[sim.grid_mv], device=sim.device)

        if sim.indexed_sort:
            if sim.precompute_stress:
                with nsight.annotate("stress"):
                    wp.launch(
                        wb._compute_stress_kernel,
                        dim=sim.n,
                        inputs=[sim.F, sim.stress, sim.mu, sim.la],
                        device=sim.device,
                    )
                p2g_inputs = [
                    sim.ids, sim.x, sim.v, sim.C, sim.stress, sim.cell_start,
                    sim.G, sim.dt, sim.vol, sim.p_mass, sim.inv_dx, sim.dx,
                    sim.grid_mv, sim.grid_m,
                ]
            else:
                p2g_inputs = [
                    sim.ids, sim.x, sim.v, sim.C, sim.F, sim.cell_start,
                    sim.G, sim.dt, sim.vol, sim.p_mass, sim.inv_dx, sim.dx,
                    sim.mu, sim.la, sim.grid_mv, sim.grid_m,
                ]
        else:
            p2g_inputs = [
                sim.xs, sim.vs, sim.Cs, sim.Fs, sim.cell_start,
                sim.G, sim.dt, sim.vol, sim.p_mass, sim.inv_dx, sim.dx,
                sim.mu, sim.la, sim.grid_mv, sim.grid_m,
            ]

        with nsight.annotate("p2g"):
            wp.launch_tiled(
                sim.p2g_kernel,
                dim=[sim.Gs3],
                inputs=p2g_inputs,
                block_dim=sim.tile_size,
                device=sim.device,
            )

        with nsight.annotate("grid_update"):
            wp.launch(
                wb._grid_update_kernel,
                dim=sim.G3,
                inputs=[
                    sim.grid_mv, sim.grid_m, sim.G, sim.dt, sim.damping,
                    sim.gravity, sim.floor_bound,
                ],
                device=sim.device,
            )

        with nsight.annotate("g2p"):
            if sim.indexed_sort:
                wp.launch(
                    wb._g2p_indexed_kernel,
                    dim=sim.n,
                    inputs=[
                        sim.ids, sim.x, sim.F, sim.grid_mv,
                        sim.x2, sim.v2, sim.C2, sim.F2,
                        sim.G, sim.dt, sim.inv_dx, sim.dx, sim.clip_bound,
                    ],
                    device=sim.device,
                )
            else:
                wp.launch(
                    wb._g2p_kernel,
                    dim=sim.n,
                    inputs=[
                        sim.xs, sim.Fs, sim.grid_mv,
                        sim.x2, sim.v2, sim.C2, sim.F2,
                        sim.G, sim.dt, sim.inv_dx, sim.dx, sim.clip_bound,
                    ],
                    device=sim.device,
                )
        sim.x, sim.x2 = sim.x2, sim.x
        sim.v, sim.v2 = sim.v2, sim.v
        sim.C, sim.C2 = sim.C2, sim.C
        sim.F, sim.F2 = sim.F2, sim.F

        wp.synchronize_device(sim.device)

    return _annotated_substep


def _variant_without_label(variant):
    if variant is None:
        return {}
    variant_dict = OmegaConf.to_container(OmegaConf.create(variant), resolve=True)
    if not isinstance(variant_dict, Mapping):
        raise RuntimeError("Each nsight.configs entry must be a mapping of Hydra overrides.")
    variant_dict = dict(variant_dict)
    variant_dict.pop("label", None)
    return variant_dict


def _variant_value(variant: Mapping, path: str, default):
    cursor = variant
    for part in path.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def _merge_variant_cfg(
    base_cfg: DictConfig,
    *,
    label: str,
    kernel_name: str,
    n_particles: int,
    num_grids: int,
    steps_per_frame: int,
    overrides,
):
    variant_cfg = OmegaConf.create(deepcopy(OmegaConf.to_container(base_cfg, resolve=True)))
    merged = OmegaConf.merge(variant_cfg, OmegaConf.create(_variant_without_label(overrides)))
    merged.nsight_variant = str(label)
    merged.kernel.name = str(kernel_name)
    merged.sim.n_particles = int(n_particles)
    merged.sim.num_grids = int(num_grids)
    merged.sim.steps_per_frame = int(steps_per_frame)
    return merged


def _nsight_configs(cfg: DictConfig):
    configs = cfg.nsight.get("configs", None)
    if configs is None:
        return None
    if not isinstance(configs, ListConfig | list):
        raise RuntimeError("nsight.configs must be a list of Hydra override mappings.")
    base_kernel = cfg.get("kernel", {}).get("name", "warp_bonus_graph")
    base_n = int(cfg.sim.n_particles)
    base_g = int(cfg.sim.num_grids)
    base_steps = int(cfg.sim.steps_per_frame)
    nsight_configs = []
    for index, variant in enumerate(OmegaConf.to_container(configs, resolve=True)):
        if not isinstance(variant, Mapping):
            raise RuntimeError("Each nsight.configs entry must be a mapping of Hydra overrides.")
        label = str(variant.get("label", f"variant_{index}"))
        kernel_name = str(_variant_value(variant, "kernel.name", base_kernel))
        n_particles = int(_variant_value(variant, "sim.n_particles", base_n))
        num_grids = int(_variant_value(variant, "sim.num_grids", base_g))
        steps_per_frame = int(_variant_value(variant, "sim.steps_per_frame", base_steps))
        nsight_configs.append([
            label,
            kernel_name,
            n_particles,
            num_grids,
            steps_per_frame,
            variant,
        ])
    return nsight_configs


def _nsight_analyze_kwargs(cfg: DictConfig, run_dir: Path, kernel_name: str, phase: str):
    analyze_cfg = cfg.nsight.get("analyze", {})
    kwargs = OmegaConf.to_container(analyze_cfg, resolve=True)
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, Mapping):
        raise RuntimeError("nsight.analyze must be a mapping of nsight.analyze.kernel options.")

    unsupported = _UNSUPPORTED_ANALYZE_CONFIG_KEYS.intersection(kwargs)
    if unsupported:
        keys = ", ".join(sorted(unsupported))
        raise RuntimeError(
            "The Hydra nsight.analyze block only supports YAML-serializable "
            f"nsight.analyze.kernel options; unsupported keys: {keys}."
        )

    kwargs = dict(kwargs)
    kwargs.setdefault("runs", 1)
    kwargs.setdefault("metrics", ["gpu__time_duration.sum"])
    kwargs.setdefault("output", "progress")
    kwargs.setdefault("output_csv", True)
    kwargs.setdefault("output_prefix", str(run_dir / f"nsight_{kernel_name}_{phase}_"))
    kwargs.setdefault("configs", _nsight_configs(cfg))
    return kwargs


def _write_results(results, run_dir: Path, write_json: bool):
    df = results.to_dataframe()
    out_csv = run_dir / "nsight_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    print(df)

    if write_json:
        out_json = run_dir / "nsight_results.json"
        out_json.write_text(json.dumps(json.loads(df.to_json(orient="records")), indent=2))
        print(f"Wrote {out_json}")


def _write_plot(nsight, results, cfg: DictConfig, run_dir: Path):
    plot_cfg = cfg.nsight.get("plot", {})
    if not bool(plot_cfg.get("enabled", False)):
        return
    from nsight.visualization import visualize

    filename = Path(str(plot_cfg.get("filename", "nsight_plot.png")))
    if not filename.is_absolute():
        filename = run_dir / filename

    kwargs = OmegaConf.to_container(plot_cfg, resolve=True)
    kwargs.pop("enabled", None)
    kwargs["filename"] = str(filename)
    visualize(results.to_dataframe(), **kwargs)
    print(f"Wrote {filename}")


def _run_nsight_profile(profiled_func):
    try:
        return profiled_func()
    except Exception as exc:
        if "ERR_NVGPUCTRPERM" in str(exc):
            raise RuntimeError(
                "Nsight Compute denied access to GPU performance counters "
                "(ERR_NVGPUCTRPERM). Enable NVIDIA performance counter access "
                "for this host/user, then rerun this script. See "
                "https://developer.nvidia.com/ERR_NVGPUCTRPERM"
            ) from exc
        raise


@hydra.main(version_base=None, config_path="conf", config_name="nsight_profile")
def main(cfg: DictConfig):
    nsight = _require_nsight()
    kernel_name = cfg.get("kernel", {}).get("name", "warp_bonus_graph")
    phase = cfg.nsight.get("phase", "p2g")
    if phase not in {"p2g", "step"}:
        raise RuntimeError(f"Unsupported nsight.phase={phase!r}; expected 'p2g' or 'step'.")
    if kernel_name not in {"warp_bonus_graph", "warp_bonus_v2_graph"}:
        raise RuntimeError(
            "profile_nsight.py currently profiles phases only for pure Warp bonus "
            f"kernels, got kernel={kernel_name!r}."
        )

    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    analyze_kwargs = _nsight_analyze_kwargs(cfg, run_dir, kernel_name, phase)

    @nsight.analyze.kernel(**analyze_kwargs)
    def profiled_warp_variant(label, kernel_name, n_particles, num_grids, steps_per_frame, overrides):
        profile_cfg = _merge_variant_cfg(
            cfg,
            label=label,
            kernel_name=kernel_name,
            n_particles=n_particles,
            num_grids=num_grids,
            steps_per_frame=steps_per_frame,
            overrides=overrides,
        )
        variant_kernel = profile_cfg.get("kernel", {}).get("name", kernel_name)
        if variant_kernel not in {"warp_bonus_graph", "warp_bonus_v2_graph"}:
            raise RuntimeError(f"Unsupported profiled variant kernel={variant_kernel!r}.")
        if phase == "p2g":
            launcher = _warp_bonus_p2g_runner(profile_cfg)
            annotation = f"{variant_kernel}_p2g"
            with nsight.annotate(annotation):
                launcher()
        else:
            launcher = _warp_bonus_step_runner(profile_cfg, nsight)
            launcher()

    print("Nsight profile config:")
    print(OmegaConf.to_yaml(cfg.nsight))
    unexpected = set(cfg.nsight.keys()) - _SCRIPT_NSIGHT_KEYS
    if unexpected:
        keys = ", ".join(sorted(unexpected))
        raise RuntimeError(f"Unknown nsight config keys: {keys}.")
    results = _run_nsight_profile(profiled_warp_variant)
    _write_results(results, run_dir, write_json=bool(cfg.nsight.get("write_json", True)))
    _write_plot(nsight, results, cfg, run_dir)


if __name__ == "__main__":
    # Keep Nsight output paths relative to Hydra's output directory, not cwd.
    os.environ.setdefault("NSYS_NVTX_PROFILER_REGISTER_ONLY", "0")
    main()
