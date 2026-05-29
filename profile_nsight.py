"""Hydra-driven Nsight Python profiler for MPM kernel phases and sweeps."""

from __future__ import annotations

import json
import os
import itertools
import shlex
import sys
import sysconfig
from contextlib import contextmanager
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

from simulate import get_particles

_UNSUPPORTED_ANALYZE_CONFIG_KEYS = {"configs"}
_SCRIPT_NSIGHT_KEYS = {"phase", "include_step_total", "write_json", "plot", "sweep", "configs", "analyze"}

_WARP_BONUS_KERNELS = {"warp_bonus_graph", "warp_bonus_v2_graph"}
_JAX_P2G_KERNELS = {
    "jax",
    "jax_v1_5",
    "cuda_v1_inline",
    "cuda_v2_inline",
    "cuda_v2_fori_inline",
    "cuda_v3_inline",
    "cuda_v3_fori_inline",
    "cuda_v4_inline",
    "cuda_v6_inline",
    "warp_v1_inline",
    "warp_v2_tile",
    "warp_v3_supercell_tile",
}
_P2G_KERNELS = _WARP_BONUS_KERNELS | _JAX_P2G_KERNELS

_SPEED_OF_LIGHT_METRICS = [
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
]


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


def _p2g_annotation_name(cfg: DictConfig):
    kernel_name = cfg.get("kernel", {}).get("name", "kernel")
    return f"{kernel_name}_p2g"


def _warp_bonus_p2g_runner(cfg: DictConfig, nsight):
    import warp as wp

    from mpm_jax import warp_bonus as wb

    sim = _warp_bonus_sim(cfg)
    indexed_sort = sim.indexed_sort
    annotation_name = _p2g_annotation_name(cfg)

    def run_p2g_once():
        with nsight.annotate(annotation_name):
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
            if indexed_sort:
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


def _jax_problem(cfg: DictConfig):
    from simulate import _maybe_enable_cuda_graphs

    kernel_name = cfg.get("kernel", {}).get("name", "jax")
    _maybe_enable_cuda_graphs(kernel_name)

    import jax.numpy as jnp

    from mpm_jax.boundary import build_boundary_fns
    from mpm_jax.constitutive import get_constitutive
    from mpm_jax.solver import MPMState, make_params

    sim = cfg.sim
    mat = cfg.material
    n = int(sim.n_particles)
    cube_np = get_particles(n, center=list(sim.center), size=[0.5, 0.5, 0.5])
    particles = jnp.array(cube_np, dtype=jnp.float32)

    params = make_params(
        n_particles=n,
        num_grids=int(sim.num_grids),
        dt=float(sim.dt),
        gravity=list(sim.gravity),
        rho=float(sim.rho),
        clip_bound=float(sim.clip_bound),
        damping=float(sim.damping),
        center=list(sim.center),
        size=list(sim.size),
    )

    g = jnp.arange(params.num_grids, dtype=jnp.float32)
    gx, gy, gz = jnp.meshgrid(g, g, g, indexing="ij")
    grid_x = jnp.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    pre_fn, post_fn = build_boundary_fns(
        list(sim.boundary_conditions),
        grid_x,
        params.dx,
        particles,
        params.dt,
        params.p_mass,
    )
    elasticity_fn = get_constitutive(mat.elasticity)
    plasticity_fn = get_constitutive(mat.plasticity)

    state = MPMState(
        x=particles,
        v=jnp.broadcast_to(jnp.array(list(sim.initial_velocity)), (n, 3)).copy(),
        C=jnp.zeros((n, 3, 3)),
        F=jnp.tile(jnp.eye(3), (n, 1, 1)),
    )
    return params, pre_fn, post_fn, elasticity_fn, plasticity_fn, state


def _jax_inline_p2g_stage(kernel_name, params, pre_fn, elasticity_fn):
    import jax
    import jax.numpy as jnp

    from mpm_jax.solver import StepIntermediates

    if kernel_name == "cuda_v1_inline":
        from mpm_jax.cuda.p2g_cuda import cuda_p2g_inline, is_available

        if not is_available("inline"):
            raise RuntimeError("cuda_v1_inline P2G kernel is not registered.")

        @jax.jit
        def jit_p2g_stage(state):
            x, v = pre_fn(state.x, state.v, 0.0)
            stress = elasticity_fn(state.F)
            grid_mv, grid_m = cuda_p2g_inline(
                x, v, state.C, stress,
                params.num_grids, params.dt, params.vol, params.p_mass,
                params.inv_dx, params.dx,
            )
            return grid_mv, grid_m, StepIntermediates(x_post_bc=x, F_pre_plast=state.F)

        return jit_p2g_stage

    if kernel_name in {"cuda_v2_inline", "cuda_v2_fori_inline"}:
        from mpm_jax.cuda.p2g_cuda import cuda_p2g_v2_inline, is_available

        if not is_available("v2_inline"):
            raise RuntimeError(f"{kernel_name} P2G kernel is not registered.")

        @jax.jit
        def jit_p2g_stage(state):
            x, v = pre_fn(state.x, state.v, 0.0)
            stress = elasticity_fn(state.F)
            grid_mv, grid_m = cuda_p2g_v2_inline(
                x, v, state.C, stress,
                params.num_grids, params.dt, params.vol, params.p_mass,
                params.inv_dx, params.dx,
            )
            return grid_mv, grid_m, StepIntermediates(x_post_bc=x, F_pre_plast=state.F)

        return jit_p2g_stage

    if kernel_name in {"cuda_v3_inline", "cuda_v3_fori_inline", "cuda_v6_inline"}:
        from mpm_jax.cuda.p2g_cuda import cuda_p2g_v3_inline, is_available
        from mpm_jax.morton import morton_argsort

        if not is_available("v3_inline"):
            raise RuntimeError(f"{kernel_name} P2G kernel is not registered.")

        @jax.jit
        def jit_p2g_stage(state):
            order = morton_argsort(state.x, params.inv_dx, params.num_grids)
            x_sorted = state.x[order]
            v_sorted = state.v[order]
            C_sorted = state.C[order]
            F_sorted = state.F[order]
            x, v = pre_fn(x_sorted, v_sorted, 0.0)
            stress = elasticity_fn(F_sorted)
            grid_mv, grid_m = cuda_p2g_v3_inline(
                x, v, C_sorted, stress,
                params.num_grids, params.dt, params.vol, params.p_mass,
                params.inv_dx, params.dx,
            )
            return grid_mv, grid_m, StepIntermediates(x_post_bc=x, F_pre_plast=F_sorted)

        return jit_p2g_stage

    if kernel_name == "cuda_v4_inline":
        from mpm_jax.cuda.p2g_cuda import (
            V4_SUPER_CELL_WIDTH,
            _home_super_cell_id,
            cuda_p2g_v4_inline,
            is_available,
        )

        if not is_available("v4_inline"):
            raise RuntimeError("cuda_v4_inline P2G kernel is not registered.")
        if params.num_grids % V4_SUPER_CELL_WIDTH != 0:
            raise RuntimeError(
                f"cuda_v4_inline requires num_grids ({params.num_grids}) divisible by "
                f"{V4_SUPER_CELL_WIDTH}."
            )
        g_super = params.num_grids // V4_SUPER_CELL_WIDTH
        super_boundaries = jnp.arange(g_super ** 3 + 1, dtype=jnp.int32)

        @jax.jit
        def jit_p2g_stage(state):
            x, v = pre_fn(state.x, state.v, 0.0)
            stress = elasticity_fn(state.F)
            super_id = _home_super_cell_id(
                x, params.inv_dx, params.num_grids, V4_SUPER_CELL_WIDTH)
            order = jnp.argsort(super_id)
            x_s = x[order]
            v_s = v[order]
            C_s = state.C[order]
            stress_s = stress[order]
            F_s = state.F[order]
            super_id_sorted = super_id[order]
            cell_start = jnp.searchsorted(super_id_sorted, super_boundaries).astype(jnp.int32)
            grid_mv, grid_m = cuda_p2g_v4_inline(
                x_s, v_s, C_s, stress_s, cell_start,
                params.num_grids, params.dt, params.vol, params.p_mass,
                params.inv_dx, params.dx,
            )
            return grid_mv, grid_m, StepIntermediates(x_post_bc=x_s, F_pre_plast=F_s)

        return jit_p2g_stage

    raise RuntimeError(f"Unsupported CUDA inline P2G kernel={kernel_name!r}.")


def _jax_warp_p2g_stage(kernel_name, params, pre_fn, elasticity_fn):
    import jax
    import jax.numpy as jnp

    from mpm_jax.solver import StepIntermediates

    if kernel_name == "warp_v1_inline":
        from mpm_jax.warp_p2g import warp_p2g_inline

        @jax.jit
        def jit_p2g_stage(state):
            x, v = pre_fn(state.x, state.v, 0.0)
            stress = elasticity_fn(state.F)
            grid_mv, grid_m = warp_p2g_inline(
                x, v, state.C, stress,
                params.num_grids, params.dt, params.vol, params.p_mass,
                params.inv_dx, params.dx,
            )
            return grid_mv, grid_m, StepIntermediates(x_post_bc=x, F_pre_plast=state.F)

        return jit_p2g_stage

    if kernel_name == "warp_v2_tile":
        from mpm_jax.warp_p2g import TILE_SIZE, warp_p2g_inline_tile

        if params.n_particles % TILE_SIZE != 0:
            raise RuntimeError(
                f"warp_v2_tile requires n_particles divisible by {TILE_SIZE}; "
                f"got {params.n_particles}."
            )

        @jax.jit
        def jit_p2g_stage(state):
            x, v = pre_fn(state.x, state.v, 0.0)
            stress = elasticity_fn(state.F)
            grid_mv, grid_m = warp_p2g_inline_tile(
                x, v, state.C, stress,
                params.num_grids, params.dt, params.vol, params.p_mass,
                params.inv_dx, params.dx,
            )
            return grid_mv, grid_m, StepIntermediates(x_post_bc=x, F_pre_plast=state.F)

        return jit_p2g_stage

    if kernel_name == "warp_v3_supercell_tile":
        from mpm_jax.warp_p2g import (
            SUPER_CELL_WIDTH,
            _home_super_cell_id,
            warp_p2g_supercell_tile,
        )

        if params.num_grids % SUPER_CELL_WIDTH != 0:
            raise RuntimeError(
                f"warp_v3_supercell_tile requires num_grids ({params.num_grids}) "
                f"divisible by {SUPER_CELL_WIDTH}."
            )
        g_super = params.num_grids // SUPER_CELL_WIDTH
        super_boundaries = jnp.arange(g_super ** 3 + 1, dtype=jnp.int32)

        @jax.jit
        def jit_p2g_stage(state):
            x, v = pre_fn(state.x, state.v, 0.0)
            stress = elasticity_fn(state.F)
            super_id = _home_super_cell_id(x, params.inv_dx, params.num_grids, SUPER_CELL_WIDTH)
            order = jnp.argsort(super_id)
            x_s = x[order]
            v_s = v[order]
            C_s = state.C[order]
            stress_s = stress[order]
            F_s = state.F[order]
            super_id_sorted = super_id[order]
            cell_start = jnp.searchsorted(super_id_sorted, super_boundaries).astype(jnp.int32)
            grid_mv, grid_m = warp_p2g_supercell_tile(
                x_s, v_s, C_s, stress_s, cell_start,
                params.num_grids, params.dt, params.vol, params.p_mass,
                params.inv_dx, params.dx,
            )
            return grid_mv, grid_m, StepIntermediates(x_post_bc=x_s, F_pre_plast=F_s)

        return jit_p2g_stage

    raise RuntimeError(f"Unsupported Warp/JAX P2G kernel={kernel_name!r}.")


def _jax_p2g_stage_runner(cfg: DictConfig, nsight):
    import jax

    from mpm_jax.solver import build_jit_stages

    kernel_name = cfg.get("kernel", {}).get("name", "jax")
    params, pre_fn, post_fn, elasticity_fn, plasticity_fn, state = _jax_problem(cfg)
    annotation_name = _p2g_annotation_name(cfg)

    if kernel_name == "jax":
        jit_p2g_stage, _, _ = build_jit_stages(
            params, elasticity_fn, plasticity_fn, pre_fn, post_fn)
    elif kernel_name == "jax_v1_5":
        from mpm_jax.p2g_scan import build_jit_stages_scan

        jit_p2g_stage, _, _ = build_jit_stages_scan(
            params, elasticity_fn, plasticity_fn, pre_fn, post_fn)
    elif kernel_name.startswith("cuda_"):
        jit_p2g_stage = _jax_inline_p2g_stage(kernel_name, params, pre_fn, elasticity_fn)
    elif kernel_name.startswith("warp_"):
        jit_p2g_stage = _jax_warp_p2g_stage(kernel_name, params, pre_fn, elasticity_fn)
    else:
        raise RuntimeError(f"Unsupported JAX P2G kernel={kernel_name!r}.")

    warmup = jit_p2g_stage(state)
    jax.block_until_ready(warmup)

    def run_p2g_once():
        with nsight.annotate(annotation_name):
            out = jit_p2g_stage(state)
            jax.block_until_ready(out)

    return run_p2g_once


def _p2g_runner(cfg: DictConfig, nsight):
    kernel_name = cfg.get("kernel", {}).get("name", "warp_bonus_graph")
    if kernel_name in _WARP_BONUS_KERNELS:
        return _warp_bonus_p2g_runner(cfg, nsight)
    if kernel_name in _JAX_P2G_KERNELS:
        return _jax_p2g_stage_runner(cfg, nsight)
    supported = ", ".join(sorted(_P2G_KERNELS))
    raise RuntimeError(f"Unsupported P2G kernel={kernel_name!r}. Supported kernels: {supported}")


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


def _variant_value(variant: Mapping, path: str, default):
    cursor = variant
    for part in path.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def _sweep_values(mapping: Mapping, key: str, default):
    value = mapping.get(key, default)
    if isinstance(value, list | tuple):
        return list(value)
    return [value]


def _merge_variant_cfg(
    base_cfg: DictConfig,
    *,
    kernel_name: str,
    n_particles: int,
    num_grids: int,
    steps_per_frame: int,
):
    variant_cfg = OmegaConf.create(deepcopy(OmegaConf.to_container(base_cfg, resolve=True)))
    merged = variant_cfg
    merged.kernel.name = str(kernel_name)
    merged.sim.n_particles = int(n_particles)
    merged.sim.num_grids = int(num_grids)
    merged.sim.steps_per_frame = int(steps_per_frame)
    return merged


def _sweep_kernel_names(cfg: DictConfig):
    base_kernel = cfg.get("kernel", {}).get("name", "warp_bonus_graph")
    sweep = cfg.nsight.get("sweep", None)
    if sweep is not None:
        sweep_dict = OmegaConf.to_container(sweep, resolve=True)
        if not isinstance(sweep_dict, Mapping):
            raise RuntimeError("nsight.sweep must be a mapping of parameter lists.")
        return [str(value) for value in _sweep_values(sweep_dict, "kernels", [base_kernel])]

    configs = cfg.nsight.get("configs", None)
    if configs is not None:
        kernels = []
        for variant in OmegaConf.to_container(configs, resolve=True):
            if not isinstance(variant, Mapping):
                raise RuntimeError("Each nsight.configs entry must be a mapping of Hydra overrides.")
            kernel_name = str(_variant_value(variant, "kernel.name", base_kernel))
            if kernel_name not in kernels:
                kernels.append(kernel_name)
        return kernels or [base_kernel]

    return [str(base_kernel)]


def _nsight_configs(cfg: DictConfig):
    base_n = int(cfg.sim.n_particles)
    base_g = int(cfg.sim.num_grids)
    base_steps = int(cfg.sim.steps_per_frame)

    sweep = cfg.nsight.get("sweep", None)
    if sweep is not None:
        sweep_dict = OmegaConf.to_container(sweep, resolve=True)
        if not isinstance(sweep_dict, Mapping):
            raise RuntimeError("nsight.sweep must be a mapping of parameter lists.")
        n_particles = [int(value) for value in _sweep_values(sweep_dict, "n_particles", [base_n])]
        num_grids = [int(value) for value in _sweep_values(sweep_dict, "num_grids", [base_g])]
        steps_per_frame = [
            int(value) for value in _sweep_values(sweep_dict, "steps_per_frame", [base_steps])
        ]
        return list(itertools.product(n_particles, num_grids, steps_per_frame))

    configs = cfg.nsight.get("configs", None)
    if configs is None:
        return None
    if not isinstance(configs, ListConfig | list):
        raise RuntimeError("nsight.configs must be a list of Hydra override mappings.")
    nsight_configs = []
    for variant in OmegaConf.to_container(configs, resolve=True):
        if not isinstance(variant, Mapping):
            raise RuntimeError("Each nsight.configs entry must be a mapping of Hydra overrides.")
        n_particles = int(_variant_value(variant, "sim.n_particles", base_n))
        num_grids = int(_variant_value(variant, "sim.num_grids", base_g))
        steps_per_frame = int(_variant_value(variant, "sim.steps_per_frame", base_steps))
        nsight_configs.append((n_particles, num_grids, steps_per_frame))
    return nsight_configs


def _configured_kernel_names(cfg: DictConfig):
    return set(_sweep_kernel_names(cfg))


def _prepare_process_for_kernels(kernel_names: set[str]):
    if "cuda_v6_inline" in kernel_names:
        from simulate import _maybe_enable_cuda_graphs

        _maybe_enable_cuda_graphs("cuda_v6_inline")


def _value_for_metric(metric_values, metrics: list[str], metric: str):
    if metric not in metrics:
        raise RuntimeError(
            f"Configured derive_metric requires metric {metric!r}. "
            f"Configured metrics: {metrics}"
        )
    return float(metric_values[metrics.index(metric)])


def _n_particles_from_config(config_values):
    if not config_values:
        raise RuntimeError("Expected config values to include n_particles.")
    if isinstance(config_values[0], str):
        if len(config_values) < 2:
            raise RuntimeError("Expected legacy config values to include kernel_name and n_particles.")
        return int(config_values[1])
    return int(config_values[0])


def _p2g_throughput_metric(metrics: list[str]):
    """Return a derive_metric callback that adds time and particle throughput."""

    def derive_p2g_throughput(*args):
        metric_values = args[: len(metrics)]
        config_values = args[len(metrics):]
        n_particles = _n_particles_from_config(config_values)
        time_ns = _value_for_metric(metric_values, metrics, "gpu__time_duration.sum")
        seconds = time_ns / 1e9
        return {
            "time_ms": time_ns / 1e6,
            "p2g_mparticles_per_s": (n_particles / seconds) / 1e6,
        }

    return derive_p2g_throughput


def _speed_of_light_metric(metrics: list[str]):
    """Return a derive_metric callback matching nsight-python's metric order."""

    def derive_speed_of_light(*args):
        metric_values = args[: len(metrics)]
        config_values = args[len(metrics):]
        n_particles = _n_particles_from_config(config_values)
        time_ns = _value_for_metric(metric_values, metrics, "gpu__time_duration.sum")
        sm_pct = _value_for_metric(
            metric_values,
            metrics,
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        )
        compute_memory_pct = _value_for_metric(
            metric_values,
            metrics,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        )
        dram_pct = _value_for_metric(
            metric_values,
            metrics,
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        )
        seconds = time_ns / 1e9
        return {
            "time_ms": time_ns / 1e6,
            "p2g_mparticles_per_s": (n_particles / seconds) / 1e6,
            "sol_sm_pct": sm_pct,
            "sol_compute_memory_pct": compute_memory_pct,
            "sol_dram_pct": dram_pct,
            "sol_max_pct": max(sm_pct, compute_memory_pct, dram_pct),
        }

    return derive_speed_of_light


def _derive_metric(name, metrics: list[str]):
    if name is None:
        return None
    if callable(name):
        return name
    if not isinstance(name, str):
        raise RuntimeError("nsight.analyze.derive_metric must be null or a supported preset name.")
    if name in {"throughput", "p2g_throughput"}:
        if "gpu__time_duration.sum" not in metrics:
            raise RuntimeError(
                "derive_metric='throughput' requires "
                "nsight.analyze.metrics=[gpu__time_duration.sum, ...]."
            )
        return _p2g_throughput_metric(metrics)
    if name in {"speed_of_light", "sol"}:
        missing = [metric for metric in _SPEED_OF_LIGHT_METRICS if metric not in metrics]
        if missing:
            raise RuntimeError(
                "derive_metric='speed_of_light' requires these nsight.analyze.metrics: "
                + ", ".join(missing)
            )
        return _speed_of_light_metric(metrics)
    raise RuntimeError(
        f"Unsupported nsight.analyze.derive_metric={name!r}; "
        "supported presets: throughput, speed_of_light"
    )


def _combine_kernel_metrics(name):
    if name is None:
        return None
    if callable(name):
        return name
    if not isinstance(name, str):
        raise RuntimeError("nsight.analyze.combine_kernel_metrics must be null or a preset name.")
    if name in {"sum", "add"}:
        return lambda x, y: x + y
    if name == "max":
        return max
    if name == "min":
        return min
    raise RuntimeError(
        f"Unsupported combine_kernel_metrics={name!r}; supported presets: sum, max, min"
    )


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
    kwargs["metrics"] = list(kwargs["metrics"])
    kwargs["derive_metric"] = _derive_metric(kwargs.get("derive_metric"), kwargs["metrics"])
    kwargs["combine_kernel_metrics"] = _combine_kernel_metrics(
        kwargs.get("combine_kernel_metrics")
    )
    kwargs.setdefault("output", "progress")
    kwargs.setdefault("output_csv", True)
    kwargs.setdefault("output_prefix", str(run_dir / f"nsight_{kernel_name}_{phase}_"))
    kwargs.setdefault("configs", _nsight_configs(cfg))
    return kwargs


def _write_results(results, run_dir: Path, write_json: bool):
    df = results.to_dataframe()
    print("Nsight Python wrote raw and processed CSV files via output_csv=True.")
    print(df)

    if write_json:
        out_json = run_dir / "nsight_results.json"
        out_json.write_text(json.dumps(json.loads(df.to_json(orient="records")), indent=2))
        print(f"Wrote {out_json}")


def _nsight_plot_kwargs(cfg: DictConfig, run_dir: Path):
    plot_cfg = cfg.nsight.get("plot", {})
    filename = Path(str(plot_cfg.get("filename", "nsight_plot.png")))
    if not filename.is_absolute():
        filename = run_dir / filename

    kwargs = OmegaConf.to_container(plot_cfg, resolve=True)
    kwargs.pop("enabled", None)
    kwargs["filename"] = str(filename)

    if "show_aggregate" not in kwargs:
        if kwargs.pop("show_avg", False):
            kwargs["show_aggregate"] = "avg"
        elif kwargs.pop("show_geomean", False):
            kwargs["show_aggregate"] = "geomean"
    else:
        kwargs.pop("show_avg", None)
        kwargs.pop("show_geomean", None)

    return kwargs


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


def _prepare_nsight_child_python(run_dir: Path):
    """Run NCU's target Python without site `.pth` hooks."""
    if os.environ.get("NSPY_NCU_PROFILE"):
        return

    original_python = sys.executable
    wrapper = run_dir / "nsight_python_no_site.sh"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        f"exec {shlex.quote(original_python)} -S \"$@\"\n"
    )
    wrapper.chmod(0o755)

    paths = []
    for path in [str(Path(__file__).resolve().parent), *sys.path]:
        if not path:
            continue
        resolved = str(Path(path).resolve())
        if resolved not in paths and Path(resolved).exists():
            paths.append(resolved)

    existing = os.environ.get("PYTHONPATH")
    if existing:
        for path in existing.split(os.pathsep):
            if path and path not in paths:
                paths.append(path)

    os.environ["PYTHONPATH"] = os.pathsep.join(paths)
    os.environ["NSPY_ORIGINAL_PYTHON"] = original_python
    sys.executable = str(wrapper)


@contextmanager
def _disable_editable_pth_for_nsight():
    """Temporarily hide the scikit-build editable hook from NCU target startup."""
    if os.environ.get("NSPY_NCU_PROFILE"):
        yield
        return

    purelib = Path(sysconfig.get_path("purelib"))
    pth = purelib / "_mpm_cudajax_editable.pth"
    disabled = purelib / f"_mpm_cudajax_editable.pth.nsight-disabled-{os.getpid()}"

    moved = False
    try:
        if pth.exists():
            pth.rename(disabled)
            moved = True
        yield
    finally:
        if moved and disabled.exists():
            disabled.rename(pth)


@hydra.main(version_base=None, config_path="conf", config_name="nsight_profile")
def main(cfg: DictConfig):
    nsight = _require_nsight()
    kernel_name = cfg.get("kernel", {}).get("name", "warp_bonus_graph")
    phase = cfg.nsight.get("phase", "p2g")
    if phase not in {"p2g", "step"}:
        raise RuntimeError(f"Unsupported nsight.phase={phase!r}; expected 'p2g' or 'step'.")
    configured_kernels = _configured_kernel_names(cfg)
    if phase == "p2g":
        unsupported = configured_kernels - _P2G_KERNELS
        if unsupported:
            supported = ", ".join(sorted(_P2G_KERNELS))
            raise RuntimeError(
                f"Unsupported P2G kernels: {', '.join(sorted(unsupported))}. "
                f"Supported kernels: {supported}"
            )
    elif configured_kernels - _WARP_BONUS_KERNELS:
        raise RuntimeError(
            "nsight.phase=step currently profiles only pure Warp bonus kernels; "
            f"got kernels={sorted(configured_kernels)}."
        )
    _prepare_process_for_kernels(configured_kernels)

    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    _prepare_nsight_child_python(run_dir)
    analyze_kwargs = _nsight_analyze_kwargs(cfg, run_dir, kernel_name, phase)
    plot_enabled = bool(cfg.nsight.get("plot", {}).get("enabled", False))
    plot_kwargs = _nsight_plot_kwargs(cfg, run_dir) if plot_enabled else None

    def profiled_variant(n_particles, num_grids, steps_per_frame):
        for variant_kernel in _sweep_kernel_names(cfg):
            profile_cfg = _merge_variant_cfg(
                cfg,
                kernel_name=variant_kernel,
                n_particles=n_particles,
                num_grids=num_grids,
                steps_per_frame=steps_per_frame,
            )
            if phase == "p2g":
                launcher = _p2g_runner(profile_cfg, nsight)
                launcher()
            else:
                if variant_kernel not in _WARP_BONUS_KERNELS:
                    raise RuntimeError(
                        "nsight.phase=step currently supports only pure Warp bonus "
                        f"kernels, got {variant_kernel!r}."
                    )
                launcher = _warp_bonus_step_runner(profile_cfg, nsight)
                launcher()

    profiled_variant = nsight.analyze.kernel(**analyze_kwargs)(profiled_variant)
    if plot_kwargs is not None:
        profiled_variant = nsight.analyze.plot(**plot_kwargs)(profiled_variant)

    print("Nsight profile config:")
    print(OmegaConf.to_yaml(cfg.nsight))
    unexpected = set(cfg.nsight.keys()) - _SCRIPT_NSIGHT_KEYS
    if unexpected:
        keys = ", ".join(sorted(unexpected))
        raise RuntimeError(f"Unknown nsight config keys: {keys}.")
    with _disable_editable_pth_for_nsight():
        results = _run_nsight_profile(profiled_variant)
    _write_results(results, run_dir, write_json=bool(cfg.nsight.get("write_json", True)))
    if plot_kwargs is not None:
        print(f"Wrote {plot_kwargs['filename']}")


if __name__ == "__main__":
    # Keep Nsight output paths relative to Hydra's output directory, not cwd.
    os.environ.setdefault("NSYS_NVTX_PROFILER_REGISTER_ONLY", "0")
    main()
