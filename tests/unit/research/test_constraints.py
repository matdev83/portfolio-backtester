"""Tests for research protocol metric constraints."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from portfolio_backtester.research.constraints import (
    ConstraintEvaluator,
    MetricConstraint,
    ResearchConstraintError,
    evaluate_architecture_constraints,
)
from portfolio_backtester.research.protocol_config import (
    ResearchProtocolConfigError,
    parse_double_oos_wfo_protocol,
)


def test_evaluator_single_metric_passes_within_bounds() -> None:
    rules = (MetricConstraint(display_key="Turnover", min_value=None, max_value=60.0),)
    ev = ConstraintEvaluator(rules)
    ok, fails = ev.evaluate({"Turnover": 50.0})
    assert ok is True
    assert fails == ()


def test_evaluator_min_bound_fails() -> None:
    rules = (MetricConstraint(display_key="Max Drawdown", min_value=-0.35, max_value=None),)
    ev = ConstraintEvaluator(rules)
    ok, fails = ev.evaluate({"Max Drawdown": -0.50})
    assert ok is False
    assert len(fails) == 1
    assert "Max Drawdown" in fails[0]


def test_evaluator_max_bound_fails() -> None:
    rules = (MetricConstraint(display_key="Turnover", min_value=None, max_value=60.0),)
    ev = ConstraintEvaluator(rules)
    ok, fails = ev.evaluate({"Turnover": 70.0})
    assert ok is False
    assert "Turnover" in fails[0]


def test_evaluator_both_bounds_pass_when_min_le_max_value_in_range() -> None:
    rules = (MetricConstraint(display_key="Calmar", min_value=0.5, max_value=2.0),)
    ev = ConstraintEvaluator(rules)
    ok, fails = ev.evaluate({"Calmar": 1.0})
    assert ok is True
    assert fails == ()


def test_evaluator_missing_metric_key_in_metrics_fails() -> None:
    rules = (MetricConstraint(display_key="Sharpe", min_value=0.0, max_value=None),)
    ev = ConstraintEvaluator(rules)
    ok, fails = ev.evaluate({"Calmar": 1.0})
    assert ok is False
    assert any("Sharpe" in f for f in fails)


def test_evaluator_nan_value_fails() -> None:
    rules = (MetricConstraint(display_key="Sharpe", min_value=0.0, max_value=None),)
    ev = ConstraintEvaluator(rules)
    ok, fails = ev.evaluate({"Sharpe": float("nan")})
    assert ok is False
    assert fails


def test_empty_rules_always_pass() -> None:
    ev = ConstraintEvaluator(())
    ok, fails = ev.evaluate({})
    assert ok is True
    assert fails == ()


def test_evaluate_architecture_constraints_helper() -> None:
    rules = (MetricConstraint(display_key="Turnover", min_value=None, max_value=10.0),)
    ok, fails = evaluate_architecture_constraints({"Turnover": 5.0}, rules)
    assert ok is True
    assert fails == ()


def test_parse_constraints_canonicalizes_metric_alias() -> None:
    inner = parse_double_oos_wfo_protocol(
        {
            "research_protocol": {
                "type": "double_oos_wfo",
                "enabled": True,
                "global_train_period": {"start_date": "2020-01-01", "end_date": "2021-12-31"},
                "unseen_test_period": {"start_date": "2022-01-01", "end_date": "2023-12-31"},
                "wfo_window_grid": {
                    "train_window_months": [12],
                    "test_window_months": [6],
                    "wfo_step_months": [3],
                    "walk_forward_type": ["rolling"],
                },
                "selection": {"top_n": 1, "metric": "Calmar"},
                "constraints": [{"metric": "max_drawdown", "min_value": -0.4}],
            }
        }
    )
    assert len(inner.constraints) == 1
    assert inner.constraints[0].display_key == "Max Drawdown"
    assert inner.constraints[0].min_value == pytest.approx(-0.4)
    assert inner.constraints[0].max_value is None


def test_parse_constraints_rejects_no_bounds() -> None:
    raw = {
        "research_protocol": {
            "type": "double_oos_wfo",
            "enabled": True,
            "global_train_period": {"start_date": "2020-01-01", "end_date": "2021-12-31"},
            "unseen_test_period": {"start_date": "2022-01-01", "end_date": "2023-12-31"},
            "wfo_window_grid": {
                "train_window_months": [12],
                "test_window_months": [6],
                "wfo_step_months": [3],
                "walk_forward_type": ["rolling"],
            },
            "selection": {"top_n": 1, "metric": "Calmar"},
            "constraints": [{"metric": "Sharpe"}],
        }
    }
    with pytest.raises(ResearchProtocolConfigError, match="min_value"):
        parse_double_oos_wfo_protocol(raw)


def test_parse_constraints_rejects_min_greater_than_max() -> None:
    raw = {
        "research_protocol": {
            "type": "double_oos_wfo",
            "enabled": True,
            "global_train_period": {"start_date": "2020-01-01", "end_date": "2021-12-31"},
            "unseen_test_period": {"start_date": "2022-01-01", "end_date": "2023-12-31"},
            "wfo_window_grid": {
                "train_window_months": [12],
                "test_window_months": [6],
                "wfo_step_months": [3],
                "walk_forward_type": ["rolling"],
            },
            "selection": {"top_n": 1, "metric": "Calmar"},
            "constraints": [{"metric": "Calmar", "min_value": 2.0, "max_value": 1.0}],
        }
    }
    with pytest.raises(ResearchProtocolConfigError, match="min"):
        parse_double_oos_wfo_protocol(raw)


def test_parse_constraints_rejects_bad_metric_key() -> None:
    raw = {
        "research_protocol": {
            "type": "double_oos_wfo",
            "enabled": True,
            "global_train_period": {"start_date": "2020-01-01", "end_date": "2021-12-31"},
            "unseen_test_period": {"start_date": "2022-01-01", "end_date": "2023-12-31"},
            "wfo_window_grid": {
                "train_window_months": [12],
                "test_window_months": [6],
                "wfo_step_months": [3],
                "walk_forward_type": ["rolling"],
            },
            "selection": {"top_n": 1, "metric": "Calmar"},
            "constraints": [{"metric": "not_a_metric_xyz", "max_value": 1.0}],
        }
    }
    with pytest.raises(ResearchProtocolConfigError):
        parse_double_oos_wfo_protocol(raw)


def test_research_constraint_error_is_clear() -> None:
    exc = ResearchConstraintError("no architectures satisfied constraints")
    assert "constraint" in str(exc).lower() or "no " in str(exc).lower()


def test_metric_constraint_dataclass_frozen() -> None:
    c = MetricConstraint(display_key="Sharpe", min_value=0.0, max_value=None)
    with pytest.raises(FrozenInstanceError):
        c.display_key = "x"  # type: ignore[misc]
