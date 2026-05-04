"""Conservative timing smoke: seasonal target weights + canonical sim after Numba warmup."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.backtester_logic.portfolio_logic import calculate_portfolio_returns
from portfolio_backtester.strategies._core.target_generation import StrategyContext
from portfolio_backtester.strategies.builtins.signal.seasonal_signal_strategy import (
    SeasonalSignalStrategy,
)


@pytest.mark.perf
def test_seasonal_targets_and_canonical_portfolio_sim_complete_quickly_after_warmup() -> None:
    idx = pd.bdate_range("2024-01-02", periods=120, freq="B")
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
    rd = pd.DatetimeIndex(idx[5:100:3])
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
    daily = pd.DataFrame(
        {"AAA": np.linspace(100.0, 105.0, len(idx)), "BBB": np.linspace(50.0, 51.0, len(idx))},
        index=idx,
    )
    rets = daily.pct_change(fill_method=None).fillna(0.0)
    sized_template = pd.DataFrame(0.5, index=idx, columns=["AAA", "BBB"])
    scenario = {
        "timing_config": {"rebalance_frequency": "D"},
        "costs_config": {"transaction_costs_bps": 5.0},
    }
    g = {"portfolio_value": 50_000.0}

    warm_tw = strat.generate_target_weights(ctx)
    assert warm_tw is not None
    _, _ = calculate_portfolio_returns(
        sized_template, scenario, daily, rets, universe_tickers, g, track_trades=False
    )

    t0 = time.perf_counter()
    for _ in range(3):
        tw = strat.generate_target_weights(ctx)
        assert tw is not None
        _, _ = calculate_portfolio_returns(
            sized_template, scenario, daily, rets, universe_tickers, g, track_trades=False
        )
    assert (time.perf_counter() - t0) < 10.0
