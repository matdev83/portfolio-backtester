"""WFO grid heatmap PNG writers (research protocol artifacts)."""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from portfolio_backtester.research.protocol_config import (
    DoubleOOSWFOProtocolConfig,
    normalize_heatmap_metric_token,
)
from portfolio_backtester.research.results import WFOArchitectureResult

logger = logging.getLogger(__name__)


def _heatmap_metric_slug(normalized_metric: str) -> str:
    base = normalized_metric.strip().lower().replace(" ", "_")
    slug = re.sub(r"[^a-z0-9_]+", "_", base)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "metric"


def heatmap_png_path(
    run_dir: Path, normalized_metric: str, step: int, walk_forward_type: str
) -> Path:
    """Build the canonical heatmap filename for a metric / subgroup."""

    slug = _heatmap_metric_slug(normalized_metric)
    return run_dir / f"wfo_heatmap_{slug}_step_{step}_{walk_forward_type}.png"


def _metric_cell_value(row: WFOArchitectureResult, normalized_metric: str) -> float:
    if normalized_metric == "score":
        return float(row.score)
    if normalized_metric == "robust_score":
        return float("nan") if row.robust_score is None else float(row.robust_score)
    v = row.metrics.get(normalized_metric)
    return float("nan") if v is None else float(v)


def _float_close(a: float, b: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isnan(a) or math.isnan(b):
        return False
    return math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12)


def write_wfo_heatmaps(
    run_dir: Path | str,
    grid_results: Sequence[WFOArchitectureResult],
    protocol_config: DoubleOOSWFOProtocolConfig,
) -> tuple[Path, ...]:
    """Write one PNG heatmap per requested metric and WFO subgroup (step + walk-forward type)."""

    root = Path(run_dir)
    reporting = protocol_config.reporting
    if not reporting.enabled or not reporting.generate_heatmaps:
        return ()

    grid = protocol_config.wfo_window_grid
    train_labels = sorted(grid.train_window_months)
    test_labels = sorted(grid.test_window_months)
    if not train_labels or not test_labels:
        return ()

    metrics_norm = tuple(normalize_heatmap_metric_token(m) for m in reporting.heatmap_metrics)
    subgroups: set[tuple[int, str]] = set()
    for r in grid_results:
        a = r.architecture
        subgroups.add((a.wfo_step_months, a.walk_forward_type))

    written: list[Path] = []
    for step, wft in sorted(subgroups):
        rows_sub = [
            r
            for r in grid_results
            if r.architecture.wfo_step_months == step and r.architecture.walk_forward_type == wft
        ]
        if not rows_sub:
            continue
        for raw_metric in metrics_norm:
            cell_vals: dict[tuple[int, int], float] = {}
            pivot_ok = True
            for r in rows_sub:
                tr = r.architecture.train_window_months
                te = r.architecture.test_window_months
                key = (tr, te)
                val = _metric_cell_value(r, raw_metric)
                if key in cell_vals and not _float_close(cell_vals[key], val):
                    pivot_ok = False
                    break
                cell_vals[key] = val
            if not pivot_ok:
                logger.warning(
                    "Skipping heatmap for metric=%s step=%s type=%s: duplicate grid cells differ",
                    raw_metric,
                    step,
                    wft,
                )
                continue

            mat = np.full((len(test_labels), len(train_labels)), np.nan, dtype=float)
            tr_ix = {v: i for i, v in enumerate(train_labels)}
            te_ix = {v: i for i, v in enumerate(test_labels)}
            for (tr, te), val in cell_vals.items():
                if tr not in tr_ix or te not in te_ix:
                    continue
                mat[te_ix[te], tr_ix[tr]] = val

            out_path = heatmap_png_path(root, raw_metric, step, wft)
            fig, ax = plt.subplots(
                figsize=(max(4.0, len(train_labels) * 0.6), max(3.0, len(test_labels) * 0.5))
            )
            im = ax.imshow(mat, origin="lower", aspect="auto")
            ax.set_xticks(range(len(train_labels)))
            ax.set_xticklabels([str(x) for x in train_labels])
            ax.set_yticks(range(len(test_labels)))
            ax.set_yticklabels([str(y) for y in test_labels])
            ax.set_xlabel("train_window_months")
            ax.set_ylabel("test_window_months")
            ax.set_title(f"{raw_metric} (step={step} {wft})")
            fig.colorbar(im, ax=ax, shrink=0.7)
            fig.tight_layout()
            fig.savefig(out_path, dpi=120)
            plt.close(fig)
            written.append(out_path)

    return tuple(written)
