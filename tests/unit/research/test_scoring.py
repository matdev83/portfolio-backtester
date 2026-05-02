"""Tests for research scoring helpers."""

from __future__ import annotations

import pytest

from portfolio_backtester.research.results import WFOArchitecture, WFOArchitectureResult
from portfolio_backtester.research.scoring import (
    CompositeRankScoringConfig,
    ResearchScoreCalculator,
    ResearchScoringError,
    compute_composite_rank_scores_for_results,
    extract_metric_value,
    select_top_protocols_by_score,
    select_top_selected_protocols,
    select_top_selected_protocols_robust_composite,
)


def test_calculator_common_metrics_maximize_direction() -> None:
    sharpe = ResearchScoreCalculator(metric_key="Sharpe")
    assert sharpe.score_for_ranking(2.0) > sharpe.score_for_ranking(1.0)
    calcar = ResearchScoreCalculator(metric_key="Calmar")
    assert calcar.score_for_ranking(2.0) > calcar.score_for_ranking(1.0)
    sortino = ResearchScoreCalculator(metric_key="sortino")
    assert sortino.score_for_ranking(1.5) > sortino.score_for_ranking(0.5)
    tr = ResearchScoreCalculator(metric_key="total_return")
    assert tr.score_for_ranking(0.1) > tr.score_for_ranking(0.05)
    dd = ResearchScoreCalculator(metric_key="Max Drawdown")
    assert dd.score_for_ranking(-0.10) > dd.score_for_ranking(-0.30)


def test_extractor_display_and_aliases() -> None:
    metrics = {"Calmar": 1.5, "Max Drawdown": -0.2}
    assert extract_metric_value(metrics, "calmar") == 1.5
    assert extract_metric_value(metrics, "Max Drawdown") == -0.2


def test_missing_registered_metric_raises() -> None:
    with pytest.raises(ResearchScoringError):
        ResearchScoreCalculator(metric_key="not_a_registered_metric")


def test_extract_missing_metric_raises() -> None:
    with pytest.raises(ResearchScoringError):
        extract_metric_value({"Sharpe": 1.0}, "Calmar")


def test_nan_positive_inf_negative_inf_rules() -> None:
    calc = ResearchScoreCalculator(metric_key="Sharpe")
    assert calc.score_for_ranking(float("nan")) == float("-inf")
    assert calc.score_for_ranking(float("inf")) == float("inf")
    assert calc.score_for_ranking(float("-inf")) == float("-inf")


def test_select_top_architectures_stable_tie_breaker_grid_order() -> None:
    grid = [
        WFOArchitecture(
            train_window_months=12,
            test_window_months=6,
            wfo_step_months=3,
            walk_forward_type="rolling",
        ),
        WFOArchitecture(
            train_window_months=24,
            test_window_months=6,
            wfo_step_months=3,
            walk_forward_type="rolling",
        ),
        WFOArchitecture(
            train_window_months=36,
            test_window_months=6,
            wfo_step_months=3,
            walk_forward_type="rolling",
        ),
    ]
    scores = [1.0, 1.0, 0.5]
    picked = select_top_protocols_by_score(grid, scores, top_n=2)
    assert picked == [grid[0], grid[1]]


def mk_result(
    *,
    train: int,
    calmar: float,
) -> WFOArchitectureResult:
    arch = WFOArchitecture(
        train_window_months=train,
        test_window_months=6,
        wfo_step_months=3,
        walk_forward_type="rolling",
    )
    return WFOArchitectureResult(
        architecture=arch,
        metrics={"Calmar": calmar, "Sharpe": 0.5},
        score=0.0,
        robust_score=None,
        best_parameters={"p": train},
        n_evaluations=1,
        stitched_returns=None,
    )


def test_select_top_selected_protocols_metric_and_ties() -> None:
    r0 = mk_result(train=12, calmar=1.0)
    r1 = mk_result(train=24, calmar=1.0)
    r2 = mk_result(train=36, calmar=0.5)
    out = select_top_selected_protocols(
        (r2, r0, r1),
        metric_key="Calmar",
        top_n=2,
    )
    assert len(out) == 2
    assert out[0].rank == 1 and out[1].rank == 2
    assert out[0].architecture.train_window_months == 12
    assert out[1].architecture.train_window_months == 24
    assert out[0].selected_parameters == {"p": 12}
    assert out[0].score == 1.0


def _res(train: int, metrics: dict) -> WFOArchitectureResult:
    arch = WFOArchitecture(
        train_window_months=train,
        test_window_months=6,
        wfo_step_months=3,
        walk_forward_type="rolling",
    )
    return WFOArchitectureResult(
        architecture=arch,
        metrics=metrics,
        score=0.0,
        robust_score=None,
        best_parameters={"t": train},
        n_evaluations=1,
        stitched_returns=None,
    )


def test_composite_ranks_use_order_not_raw_scale() -> None:
    cfg = CompositeRankScoringConfig(
        weights=(("Calmar", 1.0),),
        directions={"Calmar": "higher"},
    )
    r0 = _res(0, {"Calmar": 1.0})
    r1 = _res(1, {"Calmar": 100.0})
    scores = compute_composite_rank_scores_for_results((r0, r1), cfg)
    assert scores[0] < scores[1]


def test_composite_lower_is_better_turnover() -> None:
    cfg = CompositeRankScoringConfig(
        weights=(("Turnover", 1.0),),
        directions={"Turnover": "lower"},
    )
    hi = _res(0, {"Turnover": 0.9})
    lo = _res(1, {"Turnover": 0.1})
    scores = compute_composite_rank_scores_for_results((hi, lo), cfg)
    assert scores[1] > scores[0]


def test_composite_max_drawdown_higher_raw_is_better() -> None:
    cfg = CompositeRankScoringConfig(
        weights=(("Max Drawdown", 1.0),),
        directions={"Max Drawdown": "higher"},
    )
    a = _res(0, {"Max Drawdown": -0.30})
    b = _res(1, {"Max Drawdown": -0.10})
    scores = compute_composite_rank_scores_for_results((a, b), cfg)
    assert scores[1] > scores[0]


def test_composite_nan_is_worst() -> None:
    cfg = CompositeRankScoringConfig(
        weights=(("Calmar", 1.0),),
        directions={"Calmar": "higher"},
    )
    nan_r = _res(0, {"Calmar": float("nan")})
    ok = _res(1, {"Calmar": -1e9})
    scores = compute_composite_rank_scores_for_results((nan_r, ok), cfg)
    assert scores[1] > scores[0]


def test_composite_ties_break_by_earlier_grid_index() -> None:
    cfg = CompositeRankScoringConfig(
        weights=(("Calmar", 1.0),),
        directions={"Calmar": "higher"},
    )
    first = _res(0, {"Calmar": 2.0})
    second = _res(1, {"Calmar": 2.0})
    picked = select_top_selected_protocols_robust_composite(
        (first, second),
        composite=cfg,
        top_n=1,
    )
    assert picked[0].architecture.train_window_months == 0


def test_composite_multi_metric_weighted_ranks() -> None:
    cfg = CompositeRankScoringConfig(
        weights=(
            ("Calmar", 0.5),
            ("Turnover", 0.5),
        ),
        directions={"Calmar": "higher", "Turnover": "lower"},
    )
    a = _res(0, {"Calmar": 1.0, "Turnover": 0.5})
    b = _res(1, {"Calmar": 2.0, "Turnover": 0.9})
    c = _res(2, {"Calmar": 1.5, "Turnover": 0.1})
    scores = compute_composite_rank_scores_for_results((a, b, c), cfg)
    order = sorted(range(3), key=lambda i: (-scores[i], i))
    assert order[0] == 2


def test_select_top_eligible_subset_skips_constraint_failed_rows() -> None:
    hi = mk_result(train=12, calmar=3.0)
    fail_arch = WFOArchitecture(
        train_window_months=18,
        test_window_months=6,
        wfo_step_months=3,
        walk_forward_type="rolling",
    )
    hi_fail = WFOArchitectureResult(
        architecture=fail_arch,
        metrics={"Calmar": 100.0},
        score=100.0,
        robust_score=None,
        best_parameters={"p": "bad"},
        n_evaluations=1,
        constraint_passed=False,
        constraint_failures=("Turnover: too high",),
    )
    lo = mk_result(train=36, calmar=1.0)
    eligible = [r for r in (hi_fail, hi, lo) if r.constraint_passed]
    out = select_top_selected_protocols(eligible, metric_key="Calmar", top_n=1)
    assert out[0].architecture.train_window_months == 12
    assert out[0].constraint_passed is True


def test_robust_composite_selection_on_eligible_only_order() -> None:
    cfg = CompositeRankScoringConfig(
        weights=(("Calmar", 1.0),),
        directions={"Calmar": "higher"},
    )
    fail = WFOArchitectureResult(
        architecture=WFOArchitecture(48, 6, 3, "rolling"),
        metrics={"Calmar": 1e9},
        score=0.0,
        robust_score=None,
        best_parameters={},
        n_evaluations=1,
        constraint_passed=False,
        constraint_failures=("x",),
    )
    mid = _res(0, {"Calmar": 1.5})
    low = _res(1, {"Calmar": 1.0})
    eligible = [fail, mid, low]
    eligible_pass = [r for r in eligible if r.constraint_passed]
    picked = select_top_selected_protocols_robust_composite(
        eligible_pass,
        composite=cfg,
        top_n=1,
    )
    assert picked[0].architecture.train_window_months == 0
