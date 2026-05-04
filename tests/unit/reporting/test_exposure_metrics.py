"""Exposure / utilization metrics from realized weights (not returns-inferred)."""

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.reporting.performance_metrics import (
    EXPOSURE_CANONICAL_KEYS,
    calculate_metrics,
)


def _bench_same(rets: pd.Series, values) -> pd.Series:
    return pd.Series(values, index=rets.index)


def test_exposure_metrics_keys_stable_when_exposure_none():
    idx = pd.bdate_range("2023-01-03", periods=3)
    rets = pd.Series([0.05, 0.06, 0.01], index=idx)
    bench = _bench_same(rets, [0.01, 0.01, 0.01])
    m = calculate_metrics(rets, bench, "SPY", exposure=None)
    for k in EXPOSURE_CANONICAL_KEYS:
        assert k in m.index
        assert np.isnan(m[k]), k
    assert not np.isnan(m["Total Return"])


def test_no_returns_based_exposure_inference():
    idx = pd.bdate_range("2023-01-03", periods=3)
    rets = pd.Series([0.10, -0.05, 0.20], index=idx)
    bench = _bench_same(rets, [0.0, 0.0, 0.0])
    m = calculate_metrics(rets, bench, "SPY", exposure=None)
    assert np.isnan(m["Avg Gross Exposure"])
    assert np.isnan(m["Time in Market %"])


def test_long_only_exposure_expected():
    idx = pd.bdate_range("2023-01-03", periods=4)
    rets = pd.Series([0.01, 0.0, -0.005, 0.002], index=idx)
    bench = _bench_same(rets, [0.001, 0.001, 0.001, 0.001])
    w = pd.DataFrame({"X": [1.0, 1.0, 1.0, 1.0]}, index=idx)
    m = calculate_metrics(rets, bench, "SPY", exposure=w)
    assert m["Time in Market %"] == pytest.approx(1.0)
    assert m["Avg Gross Exposure"] == pytest.approx(1.0)
    assert m["Max Gross Exposure"] == pytest.approx(1.0)
    assert m["Avg Net Exposure"] == pytest.approx(1.0)
    assert m["Avg Long Exposure"] == pytest.approx(1.0)
    assert m["Avg Short Exposure"] == pytest.approx(0.0)
    assert not np.isnan(m["Return / Avg Gross Exposure"])
    assert not np.isnan(m["Ann. Return / Avg Gross Exposure"])


def test_cash_only_exposure_and_guarded_ratios():
    idx = pd.bdate_range("2023-01-03", periods=3)
    rets = pd.Series([0.0, 0.0, 0.0], index=idx)
    bench = _bench_same(rets, [0.01, 0.02, 0.01])
    w = pd.DataFrame({"X": [0.0, 0.0, 0.0]}, index=idx)
    m = calculate_metrics(rets, bench, "SPY", exposure=w)
    assert m["Time in Market %"] == pytest.approx(0.0)
    assert m["Avg Gross Exposure"] == pytest.approx(0.0)
    assert m["Max Gross Exposure"] == pytest.approx(0.0)
    assert m["Avg Net Exposure"] == pytest.approx(0.0)
    assert m["Avg Long Exposure"] == pytest.approx(0.0)
    assert m["Avg Short Exposure"] == pytest.approx(0.0)
    assert np.isnan(m["Return / Avg Gross Exposure"])
    assert np.isnan(m["Ann. Return / Avg Gross Exposure"])


def test_market_neutral_long_short_unit_weights():
    idx = pd.bdate_range("2023-01-03", periods=3)
    rets = pd.Series([0.01, -0.01, 0.0], index=idx)
    bench = _bench_same(rets, [0.001, 0.001, 0.001])
    w = pd.DataFrame({"L": [1.0, 1.0, 1.0], "S": [-1.0, -1.0, -1.0]}, index=idx)
    m = calculate_metrics(rets, bench, "SPY", exposure=w)
    assert m["Avg Gross Exposure"] == pytest.approx(2.0)
    assert m["Max Gross Exposure"] == pytest.approx(2.0)
    assert m["Avg Net Exposure"] == pytest.approx(0.0)
    assert m["Avg Long Exposure"] == pytest.approx(1.0)
    assert m["Avg Short Exposure"] == pytest.approx(1.0)
    assert m["Time in Market %"] == pytest.approx(1.0)


def test_exposure_reindex_to_returns_drops_extra_dates():
    idx_r = pd.bdate_range("2023-01-05", periods=3)
    idx_w = pd.bdate_range("2023-01-02", periods=8)
    rets = pd.Series([0.01, 0.02, 0.03], index=idx_r)
    bench = pd.Series([0.001, 0.001, 0.001], index=idx_r)
    w = pd.DataFrame({"A": np.linspace(1.0, 0.5, len(idx_w))}, index=idx_w)
    m = calculate_metrics(rets, bench, "SPY", exposure=w)
    assert m["Avg Gross Exposure"] == pytest.approx(float(w.loc[idx_r, "A"].mean()))
    assert m["Max Gross Exposure"] == pytest.approx(float(w.loc[idx_r, "A"].max()))


def test_gross_series_partial_metrics():
    idx = pd.bdate_range("2023-01-03", periods=4)
    rets = pd.Series([0.01, 0.02, -0.01, 0.0], index=idx)
    bench = _bench_same(rets, [0.001, 0.001, 0.001, 0.001])
    gross = pd.Series([0.5, 0.5, 0.5, 0.0], index=idx)
    m = calculate_metrics(rets, bench, "SPY", exposure=gross)
    assert m["Avg Gross Exposure"] == pytest.approx(0.375)
    assert m["Time in Market %"] == pytest.approx(0.75)
    assert np.isnan(m["Avg Net Exposure"])
    assert np.isnan(m["Avg Long Exposure"])
    assert np.isnan(m["Avg Short Exposure"])


def test_exposure_rows_all_nan_excluded_from_averages():
    idx = pd.bdate_range("2023-01-03", periods=5)
    rets = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=idx)
    bench = _bench_same(rets, [0.001] * 5)
    w = pd.DataFrame(
        {
            "X": [1.0, np.nan, np.nan, np.nan, 1.0],
        },
        index=idx,
    )
    m = calculate_metrics(rets, bench, "SPY", exposure=w)
    assert m["Avg Gross Exposure"] == pytest.approx(1.0)
    assert m["Time in Market %"] == pytest.approx(1.0)


def test_exposure_gross_series_nan_observations_excluded():
    idx = pd.bdate_range("2023-01-03", periods=3)
    rets = pd.Series([0.01, 0.02, 0.03], index=idx)
    bench = _bench_same(rets, [0.001, 0.001, 0.001])
    gross = pd.Series([1.0, np.nan, 0.5], index=idx)
    m = calculate_metrics(rets, bench, "SPY", exposure=gross)
    assert m["Avg Gross Exposure"] == pytest.approx(0.75)
    assert m["Time in Market %"] == pytest.approx(1.0)


def test_exposure_partial_nan_within_row_uses_skipna_sums():
    idx = pd.bdate_range("2023-01-03", periods=2)
    rets = pd.Series([0.01, 0.02], index=idx)
    bench = _bench_same(rets, [0.001, 0.001])
    w = pd.DataFrame({"A": [1.0, 0.5], "B": [np.nan, -0.5]}, index=idx)
    m = calculate_metrics(rets, bench, "SPY", exposure=w)
    assert m["Avg Gross Exposure"] == pytest.approx(1.0)
    assert m["Avg Net Exposure"] == pytest.approx(0.5)
