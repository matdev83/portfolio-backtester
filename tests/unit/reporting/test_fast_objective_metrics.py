"""Parity: fast optimizer metrics vs calculate_metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.reporting.fast_objective_metrics import calculate_optimizer_metrics_fast
from portfolio_backtester.reporting.performance_metrics import calculate_metrics

_OBJECTIVE = (
    "Total Return",
    "Ann. Return",
    "Ann. Vol",
    "Sharpe",
    "Sortino",
    "Calmar",
    "Max Drawdown",
)


def _bench(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(0.0, index=idx)


def _assert_objective_parity(
    rets: pd.Series, bench: pd.Series, rtol: float = 1e-9, atol: float = 1e-9
) -> None:
    slow = calculate_metrics(rets, bench, "BENCH")
    fast = calculate_optimizer_metrics_fast(rets, bench, "BENCH")
    for k in _OBJECTIVE:
        a, b = float(slow.get(k, np.nan)), float(fast.get(k, np.nan))
        if np.isnan(a) and np.isnan(b):
            continue
        if np.isinf(a) and np.isinf(b) and np.sign(a) == np.sign(b):
            continue
        assert a == pytest.approx(b, rel=rtol, abs=atol), f"{k} slow={a} fast={b}"


def test_normal_returns_daily() -> None:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2018-01-01", periods=120, freq="B")
    rets = pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)
    _assert_objective_parity(rets, _bench(idx))


def test_empty_returns() -> None:
    rets = pd.Series(dtype=float)
    bench = pd.Series(dtype=float)
    slow = calculate_metrics(rets, bench, "BENCH")
    fast = calculate_optimizer_metrics_fast(rets, bench, "BENCH")
    assert slow["Sharpe"] == fast["Sharpe"]
    assert slow["Total Return"] == fast["Total Return"]


def test_all_zero_returns() -> None:
    idx = pd.date_range("2018-01-01", periods=30, freq="B")
    rets = pd.Series(0.0, index=idx)
    bench = _bench(idx)
    slow = calculate_metrics(rets, bench, "BENCH")
    fast = calculate_optimizer_metrics_fast(rets, bench, "BENCH")
    for k in _OBJECTIVE:
        assert float(slow[k]) == float(fast[k])


def test_all_negative_active_returns() -> None:
    idx = pd.date_range("2018-01-01", periods=80, freq="B")
    rets = pd.Series(-0.001, index=idx)
    _assert_objective_parity(rets, _bench(idx))


def test_nans_in_returns() -> None:
    idx = pd.date_range("2018-01-01", periods=40, freq="B")
    raw = np.linspace(0.001, -0.002, len(idx))
    raw[10:15] = np.nan
    rets = pd.Series(raw, index=idx)
    _assert_objective_parity(rets, _bench(idx))


def test_stitched_duplicate_dates_like_evaluator() -> None:
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2020-03-02"),
            pd.Timestamp("2020-03-03"),
            pd.Timestamp("2020-03-03"),
            pd.Timestamp("2020-03-04"),
        ]
    )
    s = pd.Series([0.01, 0.02, 0.99, 0.01], index=idx)
    combined = s.sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    bench = _bench(pd.DatetimeIndex(combined.index))
    _assert_objective_parity(combined, bench)


def test_nonzero_benchmark_regression() -> None:
    idx = pd.date_range("2019-01-01", periods=60, freq="B")
    rng = np.random.default_rng(7)
    rets = pd.Series(rng.normal(0.0004, 0.008, len(idx)), index=idx)
    bench = pd.Series(rng.normal(0.0002, 0.007, len(idx)), index=idx)
    _assert_objective_parity(rets, bench, rtol=1e-8, atol=1e-8)
