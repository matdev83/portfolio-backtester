"""Parity and cache semantics for framework :class:`FeatureStore`."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.numba_optimized import atr_fast_fixed
from portfolio_backtester.optimization.feature_store import (
    FeatureStore,
    market_data_panel_fingerprint,
)
from portfolio_backtester.optimization.market_data_panel import MarketDataPanel
from portfolio_backtester.risk_management.atr_service import calculate_atr_fast
from portfolio_backtester.strategies.builtins.signal.mmm_qs_swing_nasdaq_signal_strategy import (
    _compute_adx_series,
    drawdown_from_ath_pct,
)
from tests.unit.optimization.test_market_data_panel import _make_multiindex_ohlc


def _panel_from_mi(daily: pd.DataFrame, rets: pd.DataFrame | None = None) -> MarketDataPanel:
    if rets is None:
        rets = pd.DataFrame(index=daily.index, columns=[], dtype=float)
    return MarketDataPanel.from_daily_ohlc_and_returns(daily, rets)


def test_market_data_panel_fingerprint_stable_for_same_panel() -> None:
    idx = pd.date_range("2024-01-02", periods=20, freq="B")
    tickers = ["A", "B"]
    daily = _make_multiindex_ohlc(idx, tickers)
    p1 = _panel_from_mi(daily)
    p2 = _panel_from_mi(daily)
    assert market_data_panel_fingerprint(p1) == market_data_panel_fingerprint(p2)


def test_feature_store_returns_matches_panel_returns_np() -> None:
    idx = pd.date_range("2024-01-02", periods=10, freq="B")
    tickers = ["X", "Y"]
    daily = _make_multiindex_ohlc(idx, tickers)
    rng = np.random.default_rng(1)
    rets = pd.DataFrame(rng.standard_normal((len(idx), 2)) * 0.01, index=idx, columns=tickers)
    panel = _panel_from_mi(daily, rets)
    fs = FeatureStore(panel)
    j = panel.ticker_to_column["Y"]
    got = fs.get_returns("Y")
    exp = np.asarray(panel.returns_np[:, j], dtype=np.float64)
    np.testing.assert_allclose(got, exp, rtol=1e-9)


def test_feature_store_atr_matches_atr_fast_fixed_series() -> None:
    idx = pd.date_range("2024-01-02", periods=60, freq="B")
    tickers = ["AAA"]
    daily = _make_multiindex_ohlc(idx, tickers)
    panel = _panel_from_mi(daily)
    assert panel.high_np is not None and panel.low_np is not None
    fs = FeatureStore(panel)
    period = 14
    got = fs.get_atr("AAA", period=period)
    j = 0
    high = np.asarray(panel.high_np[:, j], dtype=np.float64)
    low = np.asarray(panel.low_np[:, j], dtype=np.float64)
    close = np.asarray(panel.daily_close_np[:, j], dtype=np.float64)
    exp = atr_fast_fixed(high, low, close, period).astype(np.float64)
    np.testing.assert_allclose(got, exp, rtol=1e-9, equal_nan=True)


def test_feature_store_atr_matches_calculate_atr_fast_on_last_bar() -> None:
    idx = pd.date_range("2024-03-01", periods=40, freq="B")
    tickers = ["T1"]
    daily = _make_multiindex_ohlc(idx, tickers)
    panel = _panel_from_mi(daily)
    fs = FeatureStore(panel)
    period = 10
    atr_series = fs.get_atr("T1", period=period)
    last = idx[-1]
    legacy = calculate_atr_fast(daily, last, atr_length=period)
    assert len(legacy) == 1
    np.testing.assert_allclose(float(atr_series[-1]), float(legacy.iloc[0]), rtol=1e-9)


def test_feature_store_adx_matches_compute_adx_series() -> None:
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    tickers = ["Q"]
    daily = _make_multiindex_ohlc(idx, tickers)
    panel = _panel_from_mi(daily)
    assert panel.high_np is not None and panel.low_np is not None
    fs = FeatureStore(panel)
    di_l, adx_s = 7, 14
    got = fs.get_adx("Q", di_length=di_l, adx_smoothing=adx_s)
    j = 0
    high = pd.Series(np.asarray(panel.high_np[:, j], dtype=float), index=idx)
    low = pd.Series(np.asarray(panel.low_np[:, j], dtype=float), index=idx)
    close = pd.Series(np.asarray(panel.daily_close_np[:, j], dtype=float), index=idx)
    exp = _compute_adx_series(high, low, close, di_l, adx_s).to_numpy(dtype=np.float64)
    np.testing.assert_allclose(got, exp, rtol=1e-9, equal_nan=True)


def test_feature_store_rolling_mean_trailing_min_periods() -> None:
    idx = pd.date_range("2024-06-03", periods=15, freq="B")
    tickers = ["Z"]
    daily = _make_multiindex_ohlc(idx, tickers)
    panel = _panel_from_mi(daily)
    fs = FeatureStore(panel)
    window = 5
    got = fs.get_rolling_mean("Z", window=window, source="close")
    close = pd.Series(np.asarray(panel.daily_close_np[:, 0], dtype=float), index=idx)
    exp = close.rolling(window=window, min_periods=window).mean().to_numpy(dtype=np.float64)
    np.testing.assert_allclose(got, exp, rtol=1e-9, equal_nan=True)


def test_feature_store_ath_drawdown_matches_point_evals() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    tickers = ["S"]
    daily = _make_multiindex_ohlc(idx, tickers)
    high_manual = pd.Series([100.0, 110.0, 120.0, 100.0, 130.0], index=idx)
    close_manual = pd.Series([100.0, 105.0, 115.0, 100.0, 117.0], index=idx)
    loc_h = daily.columns.get_loc(("S", "High"))
    loc_c = daily.columns.get_loc(("S", "Close"))
    assert isinstance(loc_h, int) and isinstance(loc_c, int)
    field_ix = loc_h
    daily.iloc[:, field_ix] = high_manual.to_numpy()
    field_ix_c = loc_c
    daily.iloc[:, field_ix_c] = close_manual.to_numpy()
    panel = _panel_from_mi(daily)
    fs = FeatureStore(panel)
    dd_line = fs.get_ath_drawdown_pct("S")
    for t in idx:
        exp = drawdown_from_ath_pct(high_manual, close_manual, t)
        ix = cast(int, idx.get_loc(t))
        np.testing.assert_allclose(dd_line[ix], exp, rtol=1e-12)


def test_feature_store_cache_hit_and_param_miss() -> None:
    idx = pd.date_range("2024-01-02", periods=12, freq="B")
    daily = _make_multiindex_ohlc(idx, ["K"])
    panel = _panel_from_mi(daily)
    fs = FeatureStore(panel)
    a1 = fs.get_atr("K", period=5)
    a2 = fs.get_atr("K", period=5)
    assert a1 is not a2
    np.testing.assert_array_equal(a1, a2)
    b1 = fs.get_atr("K", period=6)
    assert not np.array_equal(a1, b1)


def test_feature_store_mutation_of_returned_array_does_not_corrupt_cache() -> None:
    idx = pd.date_range("2024-01-02", periods=8, freq="B")
    daily = _make_multiindex_ohlc(idx, ["M"])
    panel = _panel_from_mi(daily)
    fs = FeatureStore(panel)
    r1 = fs.get_returns("M")
    r1[:] = 999.0
    r2 = fs.get_returns("M")
    assert not np.allclose(r2, 999.0)


def test_feature_store_atr_requires_ohlc() -> None:
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    closes = pd.DataFrame({"X": np.linspace(10.0, 11.0, len(idx))}, index=idx)
    panel = _panel_from_mi(closes)
    fs = FeatureStore(panel)
    with pytest.raises(ValueError, match="OHLC"):
        fs.get_atr("X", period=14)
