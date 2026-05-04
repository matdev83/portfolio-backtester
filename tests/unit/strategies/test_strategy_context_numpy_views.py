from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_backtester.strategies._core.target_generation import StrategyContext


def _ctx() -> StrategyContext:
    dates = pd.date_range("2023-01-01", periods=3, freq="D")
    asset = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [10.0, 20.0, 30.0]}, index=dates)
    bench = pd.DataFrame({"^SPX": [100.0, 101.0, 102.0]}, index=dates)
    rb = pd.DatetimeIndex([dates[0], dates[2]])
    return StrategyContext.from_standard_inputs(
        asset_data=asset,
        benchmark_data=bench,
        non_universe_data=None,
        rebalance_dates=rb,
        universe_tickers=["B", "A"],
        benchmark_ticker="^SPX",
        wfo_start_date=None,
        wfo_end_date=None,
        use_sparse_nan_for_inactive_rows=False,
    )


def test_universe_close_np_column_order_matches_universe_tickers() -> None:
    ctx = _ctx()
    m = ctx.universe_close_np
    assert m.shape == (3, 2)
    assert np.allclose(m[:, 0], [10.0, 20.0, 30.0])
    assert np.allclose(m[:, 1], [1.0, 2.0, 3.0])


def test_rebalance_session_mask_np_true_on_scheduled_rows() -> None:
    ctx = _ctx()
    mask = ctx.rebalance_session_mask_np
    assert mask.dtype == np.bool_
    assert mask[0]
    assert not mask[1]
    assert mask[2]


def test_universe_close_np_inserts_nan_for_missing_ticker_column() -> None:
    dates = pd.date_range("2023-01-02", periods=2, freq="D")
    asset = pd.DataFrame({"A": [5.0, 6.0]}, index=dates)
    bench = pd.DataFrame({"^SPX": [1.0, 1.0]}, index=dates)
    ctx = StrategyContext.from_standard_inputs(
        asset_data=asset,
        benchmark_data=bench,
        non_universe_data=None,
        rebalance_dates=dates[:1],
        universe_tickers=["A", "Z"],
        benchmark_ticker="^SPX",
        wfo_start_date=None,
        wfo_end_date=None,
        use_sparse_nan_for_inactive_rows=False,
    )
    m = ctx.universe_close_np
    assert m.shape == (2, 2)
    assert not np.isnan(m[:, 0]).any()
    assert np.isnan(m[:, 1]).all()
