"""Post-selection transaction cost sweeps for research_validate (pure helpers + IO)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from portfolio_backtester.research.protocol_config import (
    ROBUST_COMPOSITE_METRIC_NAME,
    CostSensitivityConfig,
)

_COMMISSION_MULTIPLIER_KEYS: tuple[str, ...] = (
    "commission_per_share",
    "commission_min_per_order",
    "commission_max_percent_of_trade",
)

_RESERVED_ROW_KEYS: frozenset[str] = frozenset(
    {"slippage_bps", "commission_multiplier", "survives"}
)


def survival_metric_for_selection(selection_metric: str) -> str:
    """Metric used for survival / break-even heuristics."""

    if selection_metric == ROBUST_COMPOSITE_METRIC_NAME:
        return "Total Return"
    return selection_metric


def expand_cost_sensitivity_grid(cfg: CostSensitivityConfig) -> tuple[tuple[float, float], ...]:
    """Cartesian product (slippage, commission_multiplier) in stable order."""

    if not cfg.enabled:
        return ()
    out: list[tuple[float, float]] = []
    for slip in cfg.slippage_bps_grid:
        for mult in cfg.commission_multiplier_grid:
            out.append((float(slip), float(mult)))
    return tuple(out)


def effective_global_config_for_cost_cell(
    base_global: Mapping[str, Any],
    *,
    slippage_bps: float,
    commission_multiplier: float,
) -> dict[str, Any]:
    """Copy globals with slippage override and optional commission scaling."""

    out: dict[str, Any] = dict(base_global)
    out["slippage_bps"] = float(slippage_bps)
    mult = float(commission_multiplier)
    for key in _COMMISSION_MULTIPLIER_KEYS:
        if key not in base_global:
            continue
        raw = base_global[key]
        try:
            out[key] = float(raw) * mult
        except (TypeError, ValueError):
            continue
    return out


def row_survives(metrics: Mapping[str, float], survival_metric: str) -> bool:
    """True if the survival metric is strictly positive (finite)."""

    if survival_metric not in metrics:
        return False
    val = float(metrics[survival_metric])
    if math.isnan(val):
        return False
    return val > 0.0


def breakeven_slippage_bps_at_multiplier(
    rows: Sequence[Mapping[str, Any]],
    *,
    commission_multiplier: float,
    survival_metric: str,
) -> float | None:
    """Largest slippage among surviving rows at the given commission multiplier."""

    _ = survival_metric
    target = float(commission_multiplier)
    candidates: list[float] = []
    for r in rows:
        try:
            m = float(r["commission_multiplier"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isclose(m, target, rel_tol=0.0, abs_tol=1e-9):
            continue
        if not bool(r.get("survives")):
            continue
        try:
            candidates.append(float(r["slippage_bps"]))
        except (KeyError, TypeError, ValueError):
            continue
    return max(candidates) if candidates else None


def build_cost_sensitivity_summary(
    *,
    rows: Sequence[Mapping[str, Any]],
    cost_config: CostSensitivityConfig,
    baseline_global: Mapping[str, Any],
    selection_metric: str,
) -> dict[str, Any]:
    """YAML-ready summary including survival metric and break-even slippage."""

    survival_metric = survival_metric_for_selection(selection_metric)
    chosen: list[str] = [survival_metric]
    metric_keys: set[str] = set()
    for r in rows:
        for k in r:
            if k in _RESERVED_ROW_KEYS:
                continue
            metric_keys.add(str(k))
    chosen.extend(sorted(metric_keys))
    be = breakeven_slippage_bps_at_multiplier(
        rows,
        commission_multiplier=1.0,
        survival_metric=survival_metric,
    )
    return {
        "enabled": cost_config.enabled,
        "run_on": cost_config.run_on.value,
        "survival_metric": survival_metric,
        "chosen_metrics": chosen,
        "baseline_slippage_bps": baseline_global.get("slippage_bps"),
        "breakeven_slippage_bps_at_multiplier_1": be,
        "slippage_bps_grid": list(cost_config.slippage_bps_grid),
        "commission_multiplier_grid": list(cost_config.commission_multiplier_grid),
    }


def write_cost_sensitivity_artifacts(
    run_dir: Path | str,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    """Write ``cost_sensitivity.csv`` and ``cost_sensitivity_summary.yaml``."""

    root = Path(run_dir)
    df = pd.DataFrame(list(rows))
    df.to_csv(root / "cost_sensitivity.csv", index=False)
    text = yaml.safe_dump(dict(summary), sort_keys=False, allow_unicode=True)
    (root / "cost_sensitivity_summary.yaml").write_text(text, encoding="utf-8")
