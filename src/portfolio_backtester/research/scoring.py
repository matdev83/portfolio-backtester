"""Scoring helpers for research protocol ranking (pure, no IO)."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, replace
from typing import Iterable, List, Mapping, Sequence, Tuple

from .results import SelectedProtocol, WFOArchitecture, WFOArchitectureResult

ROBUST_COMPOSITE_METRIC_NAME = "RobustComposite"

_CALCULATE_METRICS_DISPLAY_KEYS: frozenset[str] = frozenset(
    {
        "Sharpe",
        "Calmar",
        "Sortino",
        "Max Drawdown",
        "Total Return",
        "Turnover",
        "Years Positive %",
    }
)

_ALIAS_TO_DISPLAY: dict[str, str] = {
    "sharpe": "Sharpe",
    "calmar": "Calmar",
    "sortino": "Sortino",
    "max_drawdown": "Max Drawdown",
    "total_return": "Total Return",
    "turnover": "Turnover",
    "years_positive_pct": "Years Positive %",
}


@dataclass(frozen=True)
class CompositeRankScoringConfig:
    """Weighted composite of per-metric ranks (higher combined score ranks better)."""

    weights: tuple[tuple[str, float], ...]
    directions: Mapping[str, str]


@dataclass(frozen=True)
class RobustSelectionConfig:
    """Neighbor-smoothed robust score for WFO architecture ranking."""

    enabled: bool
    cell_weight: float
    neighbor_median_weight: float
    neighbor_min_weight: float


def default_composite_rank_scoring() -> CompositeRankScoringConfig:
    """Default RobustComposite weights and directions for research selection."""

    weights = (
        ("Calmar", 0.35),
        ("Sortino", 0.25),
        ("Total Return", 0.20),
        ("Max Drawdown", 0.10),
        ("Turnover", 0.10),
    )
    directions = {
        "Calmar": "higher",
        "Sortino": "higher",
        "Total Return": "higher",
        "Max Drawdown": "higher",
        "Turnover": "lower",
    }
    return CompositeRankScoringConfig(weights=weights, directions=directions)


def is_robust_composite_metric(metric: str) -> bool:
    """Return True if ``metric`` selects RobustComposite selection mode."""

    s = str(metric).strip()
    if not s:
        return True
    if s == ROBUST_COMPOSITE_METRIC_NAME:
        return True
    return s.lower() == "robustcomposite"


class ResearchScoringError(ValueError):
    """Unknown metric or invalid scoring input."""


def canonical_metric_display_key(metric_key: str) -> str:
    """Resolve a user or alias metric name to the ``calculate_metrics`` column name."""

    stripped = metric_key.strip()
    if stripped in _CALCULATE_METRICS_DISPLAY_KEYS:
        return stripped
    alias = _ALIAS_TO_DISPLAY.get(stripped.lower())
    if alias is not None:
        return alias
    msg = f"unsupported metric key: {metric_key!r}"
    raise ResearchScoringError(msg)


def extract_metric_value(metrics: Mapping[str, float], metric_key: str) -> float:
    """Return the raw metric value, raising if the column is absent."""

    display_key = canonical_metric_display_key(metric_key)
    if display_key not in metrics:
        msg = f"metrics missing required key {display_key!r}"
        raise ResearchScoringError(msg)
    return float(metrics[display_key])


class ResearchScoreCalculator:
    """Convert raw metric readings into a consistent higher-is-better score."""

    def __init__(self, *, metric_key: str) -> None:
        self._display_key = canonical_metric_display_key(metric_key)

    @property
    def metric_key(self) -> str:
        return self._display_key

    def score_for_ranking(self, raw_value: float) -> float:
        """Map a metric value to a score where higher ranks better."""

        if math.isnan(raw_value):
            return float("-inf")
        if math.isinf(raw_value):
            return raw_value
        return raw_value


def eligible_grid_neighbor_keys(
    arch: WFOArchitecture,
    eligible: frozenset[WFOArchitecture],
) -> frozenset[WFOArchitecture]:
    """Return eligible architectures adjacent to ``arch`` on the train/test grid."""

    if arch not in eligible:
        return frozenset()
    step = arch.wfo_step_months
    wft = arch.walk_forward_type
    subgroup = tuple(
        a for a in eligible if a.wfo_step_months == step and a.walk_forward_type == wft
    )
    pair_to_arch: dict[tuple[int, int], WFOArchitecture] = {
        (a.train_window_months, a.test_window_months): a for a in subgroup
    }
    trains = sorted({tr for tr, _ in pair_to_arch})
    tests = sorted({te for _, te in pair_to_arch})
    tr, te = arch.train_window_months, arch.test_window_months
    if (tr, te) not in pair_to_arch:
        return frozenset()
    out: list[WFOArchitecture] = []
    ix_tr = trains.index(tr)
    if ix_tr > 0:
        cand = (trains[ix_tr - 1], te)
        if cand in pair_to_arch:
            out.append(pair_to_arch[cand])
    if ix_tr + 1 < len(trains):
        cand = (trains[ix_tr + 1], te)
        if cand in pair_to_arch:
            out.append(pair_to_arch[cand])
    ix_te = tests.index(te)
    if ix_te > 0:
        cand = (tr, tests[ix_te - 1])
        if cand in pair_to_arch:
            out.append(pair_to_arch[cand])
    if ix_te + 1 < len(tests):
        cand = (tr, tests[ix_te + 1])
        if cand in pair_to_arch:
            out.append(pair_to_arch[cand])
    return frozenset(out)


def assign_robust_scores_to_results(
    ordered_results: Sequence[WFOArchitectureResult],
    robust: RobustSelectionConfig,
) -> tuple[WFOArchitectureResult, ...]:
    """Return results in the same order with ``robust_score`` filled."""

    if not robust.enabled:
        return tuple(replace(r, robust_score=r.score) for r in ordered_results)

    eligible_arch = frozenset(r.architecture for r in ordered_results if r.constraint_passed)
    by_arch = {r.architecture: r for r in ordered_results}
    total_w = robust.cell_weight + robust.neighbor_median_weight + robust.neighbor_min_weight
    wc = robust.cell_weight / total_w
    wm = robust.neighbor_median_weight / total_w
    wn = robust.neighbor_min_weight / total_w

    merged: list[WFOArchitectureResult] = []
    for r in ordered_results:
        if not r.constraint_passed:
            merged.append(replace(r, robust_score=None))
            continue
        nbr_arch = eligible_grid_neighbor_keys(r.architecture, eligible_arch)
        if not nbr_arch:
            merged.append(replace(r, robust_score=float(r.score)))
            continue
        vals = [float(by_arch[a].score) for a in nbr_arch]
        med = float(statistics.median(vals))
        mn = min(vals)
        rs = wc * float(r.score) + wm * med + wn * mn
        merged.append(replace(r, robust_score=rs))
    return tuple(merged)


def select_top_protocols_by_score(
    grid: Sequence[WFOArchitecture],
    scores: Sequence[float],
    *,
    top_n: int,
) -> List[WFOArchitecture]:
    """Pick the top ``top_n`` architectures by descending score.

    Ties break stably by earlier grid position.
    """

    if len(grid) != len(scores):
        msg = "grid and scores length mismatch"
        raise ValueError(msg)
    if top_n < 0:
        msg = "top_n must be non-negative"
        raise ValueError(msg)
    indexed: Iterable[Tuple[int, WFOArchitecture, float]] = (
        (i, grid[i], scores[i]) for i in range(len(grid))
    )
    sorted_rows = sorted(indexed, key=lambda t: (-t[2], t[0]))
    return [row[1] for row in sorted_rows[:top_n]]


def select_top_selected_protocols(
    results: Sequence[WFOArchitectureResult],
    *,
    metric_key: str,
    top_n: int,
    rank_by_robust: bool = False,
) -> tuple[SelectedProtocol, ...]:
    """Score results by metric, return top ``SelectedProtocol`` rows stable on ties."""

    if top_n < 0:
        msg = "top_n must be non-negative"
        raise ValueError(msg)
    calc = ResearchScoreCalculator(metric_key=metric_key)
    scored_rows: List[tuple[int, WFOArchitectureResult, float]] = []
    for i, res in enumerate(results):
        if rank_by_robust:
            rs = res.robust_score
            sort_val = float(rs) if rs is not None else float(res.score)
        else:
            raw = extract_metric_value(res.metrics, metric_key)
            sort_val = calc.score_for_ranking(raw)
        scored_rows.append((i, res, sort_val))

    scored_rows.sort(key=lambda t: (-t[2], t[0]))
    selected: List[SelectedProtocol] = []
    for rank, (_, res, sort_val) in enumerate(scored_rows[:top_n], start=1):
        primary = res.score if rank_by_robust else sort_val
        selected.append(
            SelectedProtocol(
                rank=rank,
                architecture=res.architecture,
                metrics=res.metrics,
                score=primary,
                robust_score=res.robust_score,
                selected_parameters=res.best_parameters,
                constraint_passed=res.constraint_passed,
                constraint_failures=res.constraint_failures,
            )
        )
    return tuple(selected)


def _normalize_weight_pairs(pairs: Sequence[tuple[str, float]]) -> tuple[tuple[str, float], ...]:
    total = sum(w for _, w in pairs)
    if total <= 0:
        msg = "composite weights must sum to a positive total"
        raise ResearchScoringError(msg)
    return tuple((k, w / total) for k, w in pairs)


def _worst_to_best_indices(values: Sequence[float], *, higher_is_better: bool) -> list[int]:
    """Indices from worst (NaN first, stable by input index) to best."""

    def sort_key(i: int) -> tuple[float, float, float]:
        v = float(values[i])
        if math.isnan(v):
            return (0.0, 0.0, float(-i))
        if higher_is_better:
            return (1.0, v, float(-i))
        return (1.0, -v, float(-i))

    return sorted(range(len(values)), key=sort_key)


def compute_composite_rank_scores_for_results(
    results: Sequence[WFOArchitectureResult],
    composite: CompositeRankScoringConfig,
) -> list[float]:
    """Return higher-is-better composite scores from weighted average of ranks (1=worst)."""

    n = len(results)
    if n == 0:
        return []
    weights = _normalize_weight_pairs(composite.weights)
    scores = [0.0] * n
    for display_key, w in weights:
        direction = composite.directions.get(display_key, "higher")
        higher = direction == "higher"
        raw_vals = [extract_metric_value(res.metrics, display_key) for res in results]
        order = _worst_to_best_indices(raw_vals, higher_is_better=higher)
        for pos, idx in enumerate(order):
            scores[idx] += w * float(pos + 1)
    return scores


def select_top_selected_protocols_robust_composite(
    results: Sequence[WFOArchitectureResult],
    *,
    composite: CompositeRankScoringConfig,
    top_n: int,
    rank_by_robust: bool = False,
) -> tuple[SelectedProtocol, ...]:
    """Pick top protocols by composite rank score, stable ties by original order."""

    if top_n < 0:
        msg = "top_n must be non-negative"
        raise ValueError(msg)
    comp_scores = compute_composite_rank_scores_for_results(results, composite)
    scored_rows: List[tuple[int, WFOArchitectureResult, float]] = []
    for i, res in enumerate(results):
        if rank_by_robust:
            rs = res.robust_score
            sort_val = float(rs) if rs is not None else comp_scores[i]
        else:
            sort_val = comp_scores[i]
        scored_rows.append((i, res, sort_val))
    scored_rows.sort(key=lambda t: (-t[2], t[0]))
    selected: List[SelectedProtocol] = []
    for rank, (i, res, _) in enumerate(scored_rows[:top_n], start=1):
        selected.append(
            SelectedProtocol(
                rank=rank,
                architecture=res.architecture,
                metrics=res.metrics,
                score=comp_scores[i],
                robust_score=res.robust_score,
                selected_parameters=res.best_parameters,
                constraint_passed=res.constraint_passed,
                constraint_failures=res.constraint_failures,
            )
        )
    return tuple(selected)
