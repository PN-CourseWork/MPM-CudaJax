"""Hydra callbacks for the MPM-CudaJax benchmark sweeps.

ScalingPlotCallback fires on multirun end. It walks the multirun output
directory, collects every per-run ``results.json`` that ``simulate.py``
dumped, and produces two seaborn plots into the multirun root:

  scaling_ms_per_step.png     ms/step vs N, one line per kernel (log-log)
  scaling_speedup.png         speedup vs `jax` baseline, one line per kernel

Plus ``results.csv`` (long-form pandas dataframe) for downstream tooling.

The callback uses only matplotlib + seaborn + pandas; no wandb dependency.
Failures are caught and printed — the sweep itself is never blocked by a
plotting hiccup.

Usage in a Hydra sweep config:

    hydra:
      callbacks:
        scaling_plot:
          _target_: mpm_jax.callbacks.ScalingPlotCallback
          baseline_kernel: jax     # which kernel's ms/step is the denominator
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from hydra.experimental.callback import Callback
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class ScalingPlotCallback(Callback):
    """Aggregate per-run results.json files into a scaling plot."""

    def __init__(self, baseline_kernel: str = "jax"):
        self.baseline_kernel = baseline_kernel

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        try:
            self._render(config)
        except Exception as exc:
            logger.warning(
                "ScalingPlotCallback failed (this never blocks the sweep): %s",
                exc, exc_info=True,
            )

    def _render(self, config: DictConfig) -> None:
        # Sweep root = the hydra.sweep.dir of the multirun. Each subrun lives
        # in <sweep_root>/<job_index>/ with its own results.json.
        sweep_root = Path(config.hydra.sweep.dir).resolve()
        if not sweep_root.exists():
            logger.warning("Sweep root does not exist: %s", sweep_root)
            return

        result_files = sorted(sweep_root.rglob("results.json"))
        if not result_files:
            logger.warning("No results.json files under %s — nothing to plot.",
                           sweep_root)
            return

        # Lazy imports — keep the module importable in environments without
        # pandas/seaborn (e.g. CPU-only smoke imports).
        import pandas as pd
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        rows = []
        for path in result_files:
            try:
                rows.append(json.loads(path.read_text()))
            except Exception as exc:
                logger.warning("Failed to read %s: %s", path, exc)
                continue

        if not rows:
            logger.warning("All results.json files were unreadable.")
            return

        df = pd.DataFrame(rows)
        df.to_csv(sweep_root / "results.csv", index=False)
        logger.info("Wrote %s with %d rows.", sweep_root / "results.csv", len(df))

        # ---- Plot 1: ms/step vs N (log-log) ----
        sns.set_theme(style="whitegrid", context="talk")
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.lineplot(
            data=df,
            x="n_particles",
            y="ms_per_step",
            hue="kernel",
            marker="o",
            ax=ax,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Particles (N)")
        ax.set_ylabel("ms / step")
        ax.set_title("MPM P2G scaling — A10, jelly cube, G=64")
        ax.legend(title="kernel", loc="best", fontsize=10)
        fig.tight_layout()
        ms_path = sweep_root / "scaling_ms_per_step.png"
        fig.savefig(ms_path, dpi=150)
        plt.close(fig)
        logger.info("Wrote %s", ms_path)

        # ---- Plot 2: speedup vs baseline kernel ----
        # speedup[kernel, N] = baseline_ms[N] / kernel_ms[N]
        baseline = df[df["kernel"] == self.baseline_kernel]
        if baseline.empty:
            logger.warning(
                "Baseline kernel '%s' not in sweep results; skipping speedup plot. "
                "Available kernels: %s",
                self.baseline_kernel, sorted(df["kernel"].unique()),
            )
            return

        baseline_map = (baseline.groupby("n_particles")["ms_per_step"]
                        .mean().to_dict())
        df = df.copy()
        df["speedup_vs_baseline"] = df.apply(
            lambda r: baseline_map.get(r["n_particles"], float("nan"))
                      / r["ms_per_step"],
            axis=1,
        )

        fig, ax = plt.subplots(figsize=(9, 6))
        sns.lineplot(
            data=df,
            x="n_particles",
            y="speedup_vs_baseline",
            hue="kernel",
            marker="o",
            ax=ax,
        )
        ax.set_xscale("log")
        ax.axhline(1.0, ls="--", color="gray", alpha=0.6,
                   label=f"{self.baseline_kernel} baseline")
        ax.set_xlabel("Particles (N)")
        ax.set_ylabel(f"Speedup vs {self.baseline_kernel}")
        ax.set_title(f"MPM P2G speedup vs {self.baseline_kernel} — A10, G=64")
        ax.legend(title="kernel", loc="best", fontsize=10)
        fig.tight_layout()
        sp_path = sweep_root / "scaling_speedup.png"
        fig.savefig(sp_path, dpi=150)
        plt.close(fig)
        logger.info("Wrote %s", sp_path)
