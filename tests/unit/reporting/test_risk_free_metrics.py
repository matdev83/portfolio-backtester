"""Tests for Treasury yield → rf returns and excess Sharpe (TDD)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.reporting.fast_objective_metrics import calculate_optimizer_metrics_fast
from portfolio_backtester.reporting.performance_metrics import calculate_metrics
from portfolio_backtester.reporting.risk_free import (
    build_risk_free_return_series,
    excess_returns,
    extract_yield_levels,
    resolve_risk_free_yield_ticker,
    yield_levels_to_implied_rf_returns,
)


def test_yield_levels_to_implied_rf_returns_simple_split() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    levels = pd.Series([5.0, 10.0, 2.5], index=idx)
    rf = yield_levels_to_implied_rf_returns(levels, steps_per_year=252)
    assert np.allclose(rf.iloc[0], 5.0 / 100.0 / 252.0)
    assert np.allclose(rf.iloc[1], 10.0 / 100.0 / 252.0)
    assert np.allclose(rf.iloc[2], 2.5 / 100.0 / 252.0)


def test_excess_returns_mean_identity() -> None:
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.02, size=len(idx)), index=idx)
    rf = pd.Series(rng.normal(0.0001, 0.00001, size=len(idx)), index=idx)
    e = excess_returns(r, rf)
    assert len(e) == len(r)
    assert np.allclose(e.mean(), r.mean() - rf.mean(), rtol=1e-10)


def test_traditional_sharpe_excess_matches_closed_form() -> None:
    steps = 252
    idx = pd.date_range("2020-01-01", periods=steps, freq="B")
    rng = np.random.default_rng(42)
    r = pd.Series(rng.normal(0.0008, 0.015, size=len(idx)), index=idx)
    rf = pd.Series(np.full(len(idx), 2.0 / 100.0 / steps), index=idx)
    e = r - rf
    expected = float(np.sqrt(steps) * e.mean() / e.std(ddof=1))
    bench = pd.Series(0.0, index=idx)
    out = calculate_metrics(r, bench, "BENCH", risk_free_rets=rf)
    assert out["Sharpe"] == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_calculate_metrics_fast_objective_parity_with_risk_free() -> None:
    idx = pd.date_range("2019-06-01", periods=300, freq="B")
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.0005, 0.012, size=len(idx)), index=idx)
    rf = pd.Series((1.5 + 0.5 * np.sin(np.linspace(0, 3, len(idx)))) / 100.0 / 252.0, index=idx)
    bench = pd.Series(rng.normal(0.0004, 0.011, size=len(idx)), index=idx)
    slow = calculate_metrics(r, bench, "SPY", risk_free_rets=rf)
    fast = calculate_optimizer_metrics_fast(r, bench, "SPY", risk_free_rets=rf)
    assert slow["Sharpe"] == pytest.approx(float(fast["Sharpe"]), rel=1e-10, abs=1e-10)
    assert slow["Sortino"] == pytest.approx(float(fast["Sortino"]), rel=1e-10, abs=1e-10)


def test_legacy_sharpe_unchanged_when_risk_free_none() -> None:
    idx = pd.date_range("2018-01-01", periods=120, freq="B")
    r = pd.Series(0.001, index=idx)
    bench = pd.Series(0.0005, index=idx)
    out = calculate_metrics(r, bench, "SPY", risk_free_rets=None)
    out2 = calculate_metrics(r, bench, "SPY")
    assert out["Sharpe"] == out2["Sharpe"]


def test_extract_yield_levels_multiindex() -> None:
    idx = pd.date_range("2020-01-01", periods=4, freq="B")
    cols = pd.MultiIndex.from_tuples(
        [
            ("^IRX", "Open"),
            ("^IRX", "High"),
            ("^IRX", "Low"),
            ("^IRX", "Close"),
        ],
        names=["Ticker", "Field"],
    )
    mat = np.array(
        [[4.0, 4.0, 4.0, 4.0], [4.1, 4.1, 4.1, 4.1], [4.2, 4.2, 4.2, 4.2], [4.3, 4.3, 4.3, 4.3]]
    )
    df = pd.DataFrame(mat, index=idx, columns=cols)
    levels = extract_yield_levels(df, "^IRX", idx)
    assert np.allclose(levels.values, [4.0, 4.1, 4.2, 4.3])


def test_build_risk_free_return_series_end_to_end() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    cols = pd.MultiIndex.from_tuples([("^IRX", "Close")], names=["Ticker", "Field"])
    df = pd.DataFrame([[5.0], [5.0], [5.0]], index=idx, columns=cols)
    rf = build_risk_free_return_series(df, "^IRX", idx, steps_per_year=252)
    assert np.allclose(rf.iloc[0], 5.0 / 100.0 / 252.0)


def test_resolve_risk_free_yield_ticker_global_only() -> None:
    g: dict[str, object] = {"risk_free_yield_ticker": "^IRX"}
    assert resolve_risk_free_yield_ticker(g) == "^IRX"
    assert resolve_risk_free_yield_ticker({}) is None


def test_resolve_scenario_extras_null_opt_out_over_global() -> None:
    g: dict[str, object] = {"risk_free_yield_ticker": "^IRX"}
    scen = {"extras": {"risk_free_yield_ticker": None}}
    assert resolve_risk_free_yield_ticker(g, scen) is None


def test_resolve_scenario_sentinel_legacy_opt_out() -> None:
    g: dict[str, object] = {"risk_free_yield_ticker": "^IRX"}
    scen = {"extras": {"risk_free_yield_ticker": "legacy"}}
    assert resolve_risk_free_yield_ticker(g, scen) is None


def test_resolve_global_metrics_disabled() -> None:
    g: dict[str, object] = {
        "risk_free_metrics_enabled": False,
        "risk_free_yield_ticker": "^IRX",
    }
    assert resolve_risk_free_yield_ticker(g) is None


def test_resolve_scenario_metrics_enabled_overrides_global_off() -> None:
    g: dict[str, object] = {
        "risk_free_metrics_enabled": False,
        "risk_free_yield_ticker": "^IRX",
    }
    scen = {"extras": {"risk_free_metrics_enabled": True}}
    assert resolve_risk_free_yield_ticker(g, scen) == "^IRX"


def test_calculate_metrics_fast_parity_all_nan_rf_falls_back_to_legacy() -> None:
    idx = pd.date_range("2019-06-01", periods=120, freq="B")
    rng = np.random.default_rng(11)
    r = pd.Series(rng.normal(0.0005, 0.012, size=len(idx)), index=idx)
    bench = pd.Series(rng.normal(0.0004, 0.011, size=len(idx)), index=idx)
    rf_nan = pd.Series(np.nan, index=idx)
    slow_none = calculate_metrics(r, bench, "SPY", risk_free_rets=None)
    slow_rf = calculate_metrics(r, bench, "SPY", risk_free_rets=rf_nan)
    fast_rf = calculate_optimizer_metrics_fast(r, bench, "SPY", risk_free_rets=rf_nan)
    assert slow_rf["Sharpe"] == pytest.approx(float(slow_none["Sharpe"]), rel=1e-12, abs=1e-12)
    assert fast_rf["Sharpe"] == pytest.approx(float(slow_none["Sharpe"]), rel=1e-12, abs=1e-12)


def test_resolve_risk_free_yield_ticker_scenario_override() -> None:
    from frozendict import frozendict

    from portfolio_backtester.canonical_config import CanonicalScenarioConfig

    g = {"risk_free_yield_ticker": "^IRX"}
    cfg = CanonicalScenarioConfig(
        name="n",
        strategy="s",
        extras=frozendict({"risk_free_yield_ticker": "^TNX"}),
    )
    assert resolve_risk_free_yield_ticker(g, cfg) == "^TNX"
