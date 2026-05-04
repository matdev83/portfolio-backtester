"""Conservative timing smoke: Numba JIT warmup, then warm-path regression bounds.

Measures **mean time per call** after explicit warmup so gradual regressions in
``generate_target_weights`` or ``calculate_portfolio_returns`` are more visible
than a single loose wall-clock ceiling. Total-time caps remain as a hang guard.
"""

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

_WARM_SIM_ITERATIONS = 25
_WARM_TARGET_ITERATIONS = 8
# Per-call warm limits (generous for CI VMs); tighten when dedicated benchmarks exist.
_MAX_MEAN_SECONDS_PER_PORTFOLIO_SIM = 2.0
_MAX_MEAN_SECONDS_PER_TARGET_GEN = 5.0
_MAX_TOTAL_SIM_SECONDS = 30.0
_MAX_TOTAL_TARGET_SECONDS = 40.0


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

    tw_warm = strat.generate_target_weights(ctx)
    assert tw_warm is not None
    _, _ = calculate_portfolio_returns(
        sized_template, scenario, daily, rets, universe_tickers, g, track_trades=False
    )

    t_targets = time.perf_counter()
    for _ in range(_WARM_TARGET_ITERATIONS):
        tw_run = strat.generate_target_weights(ctx)
        assert tw_run is not None
    target_gen_seconds = time.perf_counter() - t_targets

    t_sim = time.perf_counter()
    for _ in range(_WARM_SIM_ITERATIONS):
        _, _ = calculate_portfolio_returns(
            sized_template, scenario, daily, rets, universe_tickers, g, track_trades=False
        )
    sim_seconds = time.perf_counter() - t_sim

    mean_target = target_gen_seconds / float(_WARM_TARGET_ITERATIONS)
    mean_sim = sim_seconds / float(_WARM_SIM_ITERATIONS)
    assert mean_target <= _MAX_MEAN_SECONDS_PER_TARGET_GEN
    assert mean_sim <= _MAX_MEAN_SECONDS_PER_PORTFOLIO_SIM
    assert target_gen_seconds <= _MAX_TOTAL_TARGET_SECONDS
    assert sim_seconds <= _MAX_TOTAL_SIM_SECONDS
