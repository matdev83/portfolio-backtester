"""Tests for neighbor-smoothed robust WFO selection scores."""

from __future__ import annotations

import pytest

from portfolio_backtester.research.results import WFOArchitecture, WFOArchitectureResult
from portfolio_backtester.research.scoring import (
    RobustSelectionConfig,
    assign_robust_scores_to_results,
    eligible_grid_neighbor_keys,
)


def _res(
    tr: int,
    te: int,
    st: int,
    wft: str,
    score: float,
    *,
    passed: bool = True,
) -> WFOArchitectureResult:
    arch = WFOArchitecture(tr, te, st, wft)
    return WFOArchitectureResult(
        architecture=arch,
        metrics={},
        score=score,
        robust_score=None,
        best_parameters={},
        n_evaluations=1,
        constraint_passed=passed,
    )


def test_neighbor_keys_same_subgroup_and_axis_adjacent_only() -> None:
    a = WFOArchitecture(12, 6, 3, "rolling")
    b = WFOArchitecture(18, 6, 3, "rolling")
    c = WFOArchitecture(12, 12, 3, "rolling")
    d = WFOArchitecture(18, 12, 3, "rolling")
    e = WFOArchitecture(12, 6, 6, "rolling")
    eligible = frozenset({a, b, c, d, e})
    n_a = eligible_grid_neighbor_keys(a, eligible)
    assert b in n_a
    assert c in n_a
    assert d not in n_a
    assert e not in n_a


def test_isolated_peak_penalized_vs_plateau_rewarded() -> None:
    rows = (
        _res(10, 6, 3, "rolling", 100.0),
        _res(11, 6, 3, "rolling", 40.0),
        _res(12, 6, 3, "rolling", 85.0),
        _res(13, 6, 3, "rolling", 85.0),
    )
    cfg = RobustSelectionConfig(
        enabled=True,
        cell_weight=0.5,
        neighbor_median_weight=0.3,
        neighbor_min_weight=0.2,
    )
    out = assign_robust_scores_to_results(rows, cfg)
    assert out[0].score == 100.0
    assert out[3].robust_score is not None and out[3].robust_score == pytest.approx(85.0)
    assert out[0].robust_score is not None
    assert out[0].robust_score < out[3].robust_score


def test_edge_cell_uses_only_existing_neighbors() -> None:
    rows = (
        _res(12, 6, 3, "rolling", 10.0),
        _res(18, 6, 3, "rolling", 50.0),
    )
    cfg = RobustSelectionConfig(True, 0.5, 0.3, 0.2)
    out = assign_robust_scores_to_results(rows, cfg)
    assert out[0].robust_score == pytest.approx(0.5 * 10 + 0.3 * 50 + 0.2 * 50)
    assert out[1].robust_score == pytest.approx(0.5 * 50 + 0.3 * 10 + 0.2 * 10)


def test_no_eligible_neighbors_falls_back_to_score() -> None:
    rows = (_res(12, 6, 3, "rolling", 77.0),)
    cfg = RobustSelectionConfig(True, 0.5, 0.3, 0.2)
    out = assign_robust_scores_to_results(rows, cfg)
    assert out[0].robust_score == pytest.approx(77.0)


def test_median_two_neighbors_is_average() -> None:
    rows = (
        _res(10, 6, 3, "rolling", 10.0),
        _res(11, 6, 3, "rolling", 99.0),
        _res(12, 6, 3, "rolling", 20.0),
    )
    cfg = RobustSelectionConfig(True, 0.0, 1.0, 0.0)
    out = assign_robust_scores_to_results(rows, cfg)
    assert out[1].robust_score == pytest.approx(15.0)


def test_disabled_sets_robust_equal_to_score() -> None:
    rows = (
        _res(12, 6, 3, "rolling", 5.0),
        _res(18, 6, 3, "rolling", 9.0),
    )
    cfg = RobustSelectionConfig(False, 0.5, 0.3, 0.2)
    out = assign_robust_scores_to_results(rows, cfg)
    assert out[0].robust_score == 5.0
    assert out[1].robust_score == 9.0


def test_ineligible_row_gets_no_robust_when_enabled() -> None:
    rows = (
        _res(12, 6, 3, "rolling", 99.0, passed=False),
        _res(18, 6, 3, "rolling", 10.0),
    )
    cfg = RobustSelectionConfig(True, 0.5, 0.3, 0.2)
    out = assign_robust_scores_to_results(rows, cfg)
    assert out[0].robust_score is None
    assert out[1].robust_score is not None


def test_neighbor_pool_only_eligible() -> None:
    rows = (
        _res(12, 6, 3, "rolling", 50.0, passed=False),
        _res(18, 6, 3, "rolling", 100.0),
    )
    cfg = RobustSelectionConfig(True, 0.5, 0.3, 0.2)
    out = assign_robust_scores_to_results(rows, cfg)
    assert out[1].robust_score == pytest.approx(100.0)


def test_deterministic_ordering_preserved_in_output() -> None:
    rows = (
        _res(20, 6, 3, "rolling", 1.0),
        _res(10, 6, 3, "rolling", 2.0),
    )
    out = assign_robust_scores_to_results(rows, RobustSelectionConfig(True, 1.0, 0.0, 0.0))
    assert [r.architecture.train_window_months for r in out] == [20, 10]
