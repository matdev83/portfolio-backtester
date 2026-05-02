"""Tests for cross-validation fold aggregation."""

from __future__ import annotations

import pytest

from portfolio_backtester.research.cross_validation_aggregate import (
    aggregate_blocked_fold_architecture_rows,
)
from portfolio_backtester.research.results import WFOArchitecture, WFOArchitectureResult


def _row(
    arch: WFOArchitecture,
    sharpe: float,
    *,
    passed: bool = True,
    fails: tuple[str, ...] = (),
    score: float | None = None,
    best_parameters: dict | None = None,
) -> WFOArchitectureResult:
    bp = {"k_param": sharpe} if best_parameters is None else dict(best_parameters)
    scr = sharpe if score is None else float(score)
    return WFOArchitectureResult(
        architecture=arch,
        metrics={"Sharpe": sharpe},
        score=scr,
        robust_score=None,
        best_parameters=bp,
        n_evaluations=10,
        constraint_passed=passed,
        constraint_failures=fails,
    )


def test_empty_fold_list_returns_empty_tuple() -> None:
    assert aggregate_blocked_fold_architecture_rows(()) == ()


def test_two_folds_average_metrics() -> None:
    arch = WFOArchitecture(24, 6, 3, "rolling")
    folds = (
        (_row(arch, 4.0),),
        (_row(arch, 2.0),),
    )
    out = aggregate_blocked_fold_architecture_rows(folds)
    assert len(out) == 1
    assert out[0].architecture == arch
    assert out[0].metrics["Sharpe"] == pytest.approx(3.0)
    assert out[0].constraint_passed is True


def test_fold_length_mismatch_raises() -> None:
    arch = WFOArchitecture(24, 6, 3, "rolling")
    folds = ((_row(arch, 1.0), _row(arch, 2.0)), (_row(arch, 3.0),))
    with pytest.raises(ValueError, match="fold 1"):
        aggregate_blocked_fold_architecture_rows(folds)


def test_architecture_mismatch_in_slot_raises() -> None:
    a1 = WFOArchitecture(24, 6, 3, "rolling")
    a2 = WFOArchitecture(12, 6, 3, "rolling")
    folds = ((_row(a1, 1.0),), (_row(a2, 2.0),))
    with pytest.raises(ValueError, match="architecture mismatch"):
        aggregate_blocked_fold_architecture_rows(folds)


def test_any_fold_constraint_failure_marks_slot_failed() -> None:
    arch = WFOArchitecture(24, 6, 3, "rolling")
    folds = (
        (
            _row(
                arch,
                4.0,
                passed=True,
                score=100.0,
                best_parameters={"beta": 0.25},
            ),
        ),
        (
            _row(
                arch,
                2.0,
                passed=False,
                fails=("Max Drawdown",),
                score=1.0,
                best_parameters={"beta": -0.5},
            ),
        ),
    )
    out = aggregate_blocked_fold_architecture_rows(folds)
    assert len(out) == 1
    assert out[0].constraint_passed is False
    assert "Max Drawdown" in out[0].constraint_failures
    assert out[0].metrics["Sharpe"] == pytest.approx(3.0)
    assert out[0].score == pytest.approx(50.5)
    assert out[0].best_parameters["beta"] == pytest.approx(-0.125)


def test_failed_slot_keeps_average_of_overlapping_metrics_only() -> None:
    arch = WFOArchitecture(24, 6, 3, "rolling")
    folds = (
        (_row(arch, 4.0, passed=True),),
        (
            WFOArchitectureResult(
                architecture=arch,
                metrics={"Sharpe": 2.0, "Turnover": 10.0},
                score=2.0,
                robust_score=None,
                best_parameters={"k_param": 2.0},
                n_evaluations=10,
                constraint_passed=False,
                constraint_failures=("Turnover",),
            ),
        ),
    )
    out = aggregate_blocked_fold_architecture_rows(folds)
    assert set(out[0].metrics.keys()) == {"Sharpe"}
    assert out[0].metrics["Sharpe"] == pytest.approx(3.0)
