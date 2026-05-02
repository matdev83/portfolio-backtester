"""Construction and alignment semantics for typed market data panels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.optimization.market_data_panel import (
    MarketDataPanel,
    evaluation_naive_datetimes_like_evaluator,
)


def _make_multiindex_ohlc(
    index: pd.DatetimeIndex,
    tickers: list[str],
) -> pd.DataFrame:
    tuples: list[tuple[str, str]] = []
    for t in tickers:
        tuples.extend([(t, "Open"), (t, "High"), (t, "Low"), (t, "Close")])
    columns = pd.MultiIndex.from_tuples(tuples, names=["Ticker", "Field"])
    n = len(index)
    m = len(tickers)
    base = np.linspace(50.0, 60.0, n * m, dtype=float).reshape(n, m)
    off = {"Open": -0.1, "High": 1.5, "Low": -1.6, "Close": 0.0}
    blocks: list[np.ndarray] = []
    for j in range(m):
        for field in ["Open", "High", "Low", "Close"]:
            blocks.append((base[:, j] + off[field]).astype(float))
    mat = np.column_stack(blocks)
    return pd.DataFrame(mat, index=index.copy(), columns=columns)


def test_multiindex_preserves_date_index_and_ticker_order() -> None:
    tz_idx = pd.date_range("2024-06-03", periods=6, freq="B", tz="UTC", name="date")
    idx = tz_idx.tz_convert("Europe/Warsaw")
    tickers = ["BBB", "AAA"]
    mi = _make_multiindex_ohlc(idx, tickers)

    rnd_times = pd.date_range("2024-06-03", periods=6, freq="B")
    rnd_vals = np.random.default_rng(0).standard_normal((len(rnd_times), 2))
    rets_seed = pd.DataFrame(rnd_vals, index=rnd_times, columns=tickers[::-1])
    panel = MarketDataPanel.from_daily_ohlc_and_returns(mi, rets_seed)

    assert list(panel.tickers) == tickers

    expected = evaluation_naive_datetimes_like_evaluator(
        mi.xs("Close", axis=1, level="Field").index
    )
    pd.testing.assert_index_equal(panel.daily_index_naive, expected)


def test_orchestrator_parity_arrays_multiindex_returns_fill() -> None:
    naive_idx = pd.date_range("2024-06-03", periods=8, freq="B")
    tickers = ["ZZ", "QQ"]
    daily = _make_multiindex_ohlc(naive_idx, tickers)

    legacy_close_maybe = daily.xs("Close", level="Field", axis=1)
    daily_close_df = (
        legacy_close_maybe
        if isinstance(legacy_close_maybe, pd.DataFrame)
        else pd.DataFrame(legacy_close_maybe)
    )
    tls = list(daily_close_df.columns)
    daily_np_legacy = np.ascontiguousarray(daily_close_df.to_numpy(dtype=np.float32))

    partial_index = naive_idx[list(range(0, len(naive_idx), 2))]
    rets_wide = pd.DataFrame(
        {"ZZ": 0.0, "QQ": np.linspace(0.001, 0.005, len(partial_index))},
        index=partial_index,
    )

    panel = MarketDataPanel.from_daily_ohlc_and_returns(daily, rets_wide)
    np.testing.assert_allclose(panel.daily_close_np, daily_np_legacy, rtol=1e-6)

    rets_full_df = pd.DataFrame(rets_wide)
    legacy_aligned = rets_full_df.reindex(daily_close_df.index).reindex(columns=tls).fillna(0.0)
    legacy_np = np.ascontiguousarray(legacy_aligned.to_numpy(dtype=np.float32))
    np.testing.assert_allclose(panel.returns_np, legacy_np)


def test_ohlc_field_arrays_ordered_and_optional() -> None:
    naive_idx = pd.date_range("2025-01-02", periods=5, freq="B")
    tickers = ["C", "D"]
    daily = _make_multiindex_ohlc(naive_idx, tickers)
    closes_df = daily.xs("Close", level="Field", axis=1)
    assert isinstance(closes_df, pd.DataFrame)
    rets_df = closes_df.pct_change().fillna(0.0)
    panel = MarketDataPanel.from_daily_ohlc_and_returns(daily, rets_df)

    assert panel.open_np is not None and panel.high_np is not None and panel.low_np is not None
    np.testing.assert_allclose(
        panel.daily_close_np, closes_df.to_numpy(dtype=np.float32), rtol=1e-5, atol=1e-6
    )
    assert panel.ticker_to_column == {"C": 0, "D": 1}
    for field, extractor in ("Open", "open_np"), ("High", "high_np"), ("Low", "low_np"):
        raw = getattr(panel, extractor)
        assert raw is not None
        want = daily.xs(field, level="Field", axis=1).to_numpy(dtype=np.float32)
        np.testing.assert_allclose(raw, want, rtol=1e-6)


def test_close_only_simple_columns() -> None:
    naive_idx = pd.date_range("2026-03-02", periods=4, freq="B")
    close_df = pd.DataFrame(
        [[1.0, 2.0], [1.5, 2.5], [1.55, 2.52], [1.6, 2.61]], columns=["X", "Y"], index=naive_idx
    )

    skew_rets_index = naive_idx[:-1][1:]  # sub index
    rets_partial = pd.DataFrame(
        np.ones((len(skew_rets_index), 2)) * 0.01, index=skew_rets_index, columns=["Y", "X"]
    )

    panel = MarketDataPanel.from_daily_ohlc_and_returns(close_df, rets_partial)
    pd.testing.assert_frame_equal(
        close_df.astype(np.float64), panel.to_close_dataframe().astype(np.float64)
    )
    tls = ["X", "Y"]
    legacy_aligned = rets_partial.reindex(close_df.index).reindex(columns=tls).fillna(0.0)
    legacy_np = np.ascontiguousarray(legacy_aligned.to_numpy(dtype=np.float32))
    np.testing.assert_allclose(panel.returns_np, legacy_np)


def test_multiindex_raises_without_close_field_metadata() -> None:
    naive_idx = pd.date_range("2026-03-02", periods=3, freq="B")
    bad = pd.DataFrame(
        [[1.0]],
        index=naive_idx,
        columns=pd.MultiIndex.from_tuples([("QQ", "NotClose")], names=["Ticker", "Field"]),
    )
    with pytest.raises(ValueError, match="Close"):
        MarketDataPanel.from_daily_ohlc_and_returns(bad, pd.DataFrame())
