"""SeasonalStrategy: ``generate_target_weights`` matches expanding ``generate_signals`` (first-anchor)."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from portfolio_backtester.strategies._core.target_generation import StrategyContext
from portfolio_backtester.strategies.builtins.signal.seasonal_signal_strategy import (
    SeasonalSignalStrategy,
)


def _close_panel_to_ohlc_multiindex(close: pd.DataFrame) -> pd.DataFrame:
    pieces: dict[tuple[str, str], pd.Series] = {}
    for ticker in close.columns:
        col = str(ticker)
        c = pd.to_numeric(close[ticker], errors="coerce").astype(float)
        pieces[(col, "Open")] = c.shift(1).fillna(c)
        pieces[(col, "High")] = c + 1.0
        pieces[(col, "Low")] = c - 1.0
        pieces[(col, "Close")] = c
    out = pd.DataFrame(pieces, index=close.index)
    out.columns.names = ["Ticker", "Field"]
    return out


def test_seasonal_target_weights_matches_expanding_signals_cross_month() -> None:
    idx = pd.bdate_range("2024-01-02", periods=52, freq="B")
    universe_tickers = ["AAA", "BBB"]
    asset_df = pd.DataFrame({"AAA": 1.0, "BBB": 1.1}, index=idx)
    benchmark_df = asset_df[["AAA"]].copy()
    strat = SeasonalSignalStrategy(
        {
            "strategy_params": {
                "entry_day": 2,
                "hold_days": 18,
                "month_local_seasonal_windows": False,
                "entry_day_by_month": {2: 1},
                "hold_days_by_month": {1: 4, 2: 10},
                "trade_month_12": False,
                "trade_month_1": True,
                "trade_month_2": True,
            }
        }
    )

    rd = pd.DatetimeIndex(idx[5:43:3])
    ctx = StrategyContext.from_standard_inputs(
        asset_data=asset_df,
        benchmark_data=benchmark_df,
        non_universe_data=pd.DataFrame(),
        rebalance_dates=rd,
        universe_tickers=list(universe_tickers),
        benchmark_ticker="AAA",
        wfo_start_date=None,
        wfo_end_date=None,
        use_sparse_nan_for_inactive_rows=False,
    )
    dense = strat.generate_target_weights(ctx)
    assert dense is not None

    for d in rd:
        end = cast(int, asset_df.index.searchsorted(pd.Timestamp(d), side="right"))
        exp = strat.generate_signals(
            asset_df.iloc[:end],
            benchmark_df.iloc[:end],
            non_universe_historical_data=None,
            current_date=pd.Timestamp(d),
        ).iloc[0]
        got = cast(pd.Series, dense.loc[pd.Timestamp(d)])
        pd.testing.assert_series_equal(
            got.reindex(universe_tickers).fillna(0.0),
            exp.reindex(universe_tickers).fillna(0.0),
            rtol=0.0,
            atol=1e-12,
        )


def test_seasonal_target_weights_simple_high_low_exits_returns_dataframe_matches_legacy() -> None:
    idx = pd.bdate_range("2024-01-02", periods=45, freq="B")
    universe_tickers = ["AAA", "BBB"]
    close_vals = pd.DataFrame(
        {
            "AAA": [100.0] * len(idx),
            "BBB": [200.0] * len(idx),
        },
        index=idx,
        dtype=float,
    )
    breach_pos = 12
    close_vals.iloc[breach_pos, 0] = 98.0
    asset_df = _close_panel_to_ohlc_multiindex(close_vals)
    benchmark_df = _close_panel_to_ohlc_multiindex(close_vals[["AAA"]])
    strat = SeasonalSignalStrategy(
        {
            "strategy_params": {
                "entry_day": 1,
                "hold_days": 15,
                "month_local_seasonal_windows": True,
                "simple_high_low_stop_loss": True,
            }
        }
    )

    rd = pd.DatetimeIndex(idx[5:25:2])
    ctx = StrategyContext.from_standard_inputs(
        asset_data=asset_df,
        benchmark_data=benchmark_df,
        non_universe_data=pd.DataFrame(),
        rebalance_dates=rd,
        universe_tickers=list(universe_tickers),
        benchmark_ticker="AAA",
        wfo_start_date=None,
        wfo_end_date=None,
        use_sparse_nan_for_inactive_rows=False,
    )
    dense = strat.generate_target_weights(ctx)
    assert dense is not None
    assert isinstance(dense, pd.DataFrame)

    for d in rd:
        end = cast(int, asset_df.index.searchsorted(pd.Timestamp(d), side="right"))
        exp = strat.generate_signals(
            asset_df.iloc[:end],
            benchmark_df.iloc[:end],
            non_universe_historical_data=None,
            current_date=pd.Timestamp(d),
        ).iloc[0]
        got = cast(pd.Series, dense.loc[pd.Timestamp(d)])
        pd.testing.assert_series_equal(
            got.reindex(universe_tickers).fillna(0.0),
            exp.reindex(universe_tickers).fillna(0.0),
            rtol=0.0,
            atol=1e-12,
        )


def test_seasonal_target_weights_atr_exits_returns_dataframe_matches_legacy() -> None:
    idx = pd.bdate_range("2024-06-03", periods=80, freq="B")
    universe_tickers = ["AAA", "BBB"]
    base = pd.Series(
        np.linspace(100.0, 120.0, len(idx)),
        index=idx,
        dtype=float,
    )
    close_vals = pd.DataFrame({"AAA": base, "BBB": base * 1.02 + 5.0}, index=idx)
    asset_df = _close_panel_to_ohlc_multiindex(close_vals)
    benchmark_df = _close_panel_to_ohlc_multiindex(close_vals[["AAA"]])
    strat = SeasonalSignalStrategy(
        {
            "strategy_params": {
                "entry_day": 1,
                "hold_days": 25,
                "month_local_seasonal_windows": True,
                "stop_loss_atr_multiple": 2.0,
                "take_profit_atr_multiple": 0.0,
            }
        }
    )

    rd = pd.DatetimeIndex(idx[30:65:4])
    ctx = StrategyContext.from_standard_inputs(
        asset_data=asset_df,
        benchmark_data=benchmark_df,
        non_universe_data=pd.DataFrame(),
        rebalance_dates=rd,
        universe_tickers=list(universe_tickers),
        benchmark_ticker="AAA",
        wfo_start_date=None,
        wfo_end_date=None,
        use_sparse_nan_for_inactive_rows=False,
    )
    dense = strat.generate_target_weights(ctx)
    assert dense is not None
    assert isinstance(dense, pd.DataFrame)

    for d in rd:
        end = cast(int, asset_df.index.searchsorted(pd.Timestamp(d), side="right"))
        exp = strat.generate_signals(
            asset_df.iloc[:end],
            benchmark_df.iloc[:end],
            non_universe_historical_data=None,
            current_date=pd.Timestamp(d),
        ).iloc[0]
        got = cast(pd.Series, dense.loc[pd.Timestamp(d)])
        pd.testing.assert_series_equal(
            got.reindex(universe_tickers).fillna(0.0),
            exp.reindex(universe_tickers).fillna(0.0),
            rtol=0.0,
            atol=1e-12,
        )
