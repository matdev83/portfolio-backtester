"""PNG chart exports used by optional research reporting (single-run directory artifacts)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats


def write_bootstrap_distribution_chart_pngs(run_dir: Path | str, *, dpi: int = 100) -> list[Path]:
    """Create histogram + normal QQ diagnostics for ``bootstrap_distribution_*.csv`` files."""

    root = Path(run_dir)
    written: list[Path] = []
    for csv_path in sorted(root.glob("bootstrap_distribution_*.csv")):
        slug = csv_path.name.removeprefix("bootstrap_distribution_").removesuffix(".csv")
        try:
            df = pd.read_csv(csv_path)
        except (OSError, ValueError, pd.errors.EmptyDataError):
            continue
        if df.empty or "value" not in df.columns:
            continue
        vals = pd.to_numeric(df["value"], errors="coerce").astype(float).dropna()
        arr = np.asarray(vals.to_numpy(dtype=float), dtype=float)
        if arr.size == 0:
            continue

        fig, ax = plt.subplots(figsize=(5.0, 3.6))
        _bins = max(5, min(30, arr.size))
        ax.hist(arr, bins=_bins, color="#4e79a7", edgecolor="#333333", linewidth=0.4)
        ax.set_title(f"Bootstrap null: {slug}")
        ax.set_xlabel("Statistic")
        ax.set_ylabel("Count")
        fig.tight_layout()
        hp = root / f"bootstrap_viz_{slug}_histogram.png"
        fig.savefig(hp, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(hp)

        if arr.size >= 2:
            fig2, ax2 = plt.subplots(figsize=(5.0, 3.6))
            stats.probplot(arr, dist="norm", plot=ax2)
            ax2.set_title(f"QQ vs normal ({slug})")
            fig2.tight_layout()
            qp = root / f"bootstrap_viz_{slug}_qq.png"
            fig2.savefig(qp, dpi=dpi, bbox_inches="tight")
            plt.close(fig2)
            written.append(qp)

    return written


def write_cost_sensitivity_line_chart_png(
    run_dir: Path | str,
    *,
    summary: Mapping[str, Any] | None = None,
    dpi: int = 100,
) -> Path | None:
    """Plot survival metric versus slippage at ``commission_multiplier == 1``.

    Preference order for survival metric column:
      ``summary.survival_metric`` if provided, otherwise load from ``cost_sensitivity_summary.yaml``.
    """

    root = Path(run_dir)
    csv_path = root / "cost_sensitivity.csv"
    if not csv_path.is_file():
        return None
    try:
        df = pd.read_csv(csv_path)
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return None

    metric: Any = None
    if summary is not None:
        metric = summary.get("survival_metric")
    if metric is None or not str(metric).strip():
        ypath = root / "cost_sensitivity_summary.yaml"
        if ypath.is_file():
            try:
                loaded = yaml.safe_load(ypath.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError, TypeError):
                loaded = None
            if isinstance(loaded, dict):
                metric = loaded.get("survival_metric")
    if metric is None or str(metric) not in df.columns:
        return None
    mcol = str(metric)
    if "slippage_bps" not in df.columns or "commission_multiplier" not in df.columns:
        return None
    mult_np = pd.to_numeric(df["commission_multiplier"], errors="coerce").to_numpy(
        dtype=float, copy=False
    )
    slip_np = pd.to_numeric(df["slippage_bps"], errors="coerce").to_numpy(dtype=float, copy=False)
    y_all = pd.to_numeric(df[mcol], errors="coerce").to_numpy(dtype=float, copy=False)
    row_ok = (
        np.isfinite(mult_np)
        & np.isclose(mult_np, 1.0, rtol=0.0, atol=1e-9)
        & np.isfinite(slip_np)
        & np.isfinite(y_all)
    )
    if not row_ok.any():
        return None
    xv = slip_np[row_ok]
    yv = y_all[row_ok]
    order = np.argsort(xv)
    xv = xv[order]
    yv = yv[order]
    if xv.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.plot(xv, yv, marker="o", color="#59a14f")
    ax.axhline(0.0, color="#999999", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Slippage (bps)")
    ax.set_ylabel(mcol)
    ax.set_title("Cost sensitivity @ commission multiplier 1.0")
    fig.tight_layout()
    outp = root / "cost_sensitivity_survival_curve.png"
    fig.savefig(outp, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return outp
