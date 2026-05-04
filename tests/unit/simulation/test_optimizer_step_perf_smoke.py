"""Additional perf smoke for canonical portfolio simulation workloads.

Complements ``test_seasonal_sim_perf_smoke`` with multi-asset schedules closer to
optimizer-step shapes (fixed-weight monthly grid, wider universe with sparse events).

Marked ``@pytest.mark.perf`` so CI can exclude them when needed; caps are hang guards,
not tight regressions—see seasonal perf test for JIT warmup pattern.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.backtester_logic.portfolio_logic import calculate_portfolio_returns

_WARM_ITERATIONS = 5
_MEASURE_ITERATIONS = 15
_MAX_MEAN_SECONDS_FIXED_WEIGHT_MONTHLY = 4.0
_MAX_MEAN_SECONDS_WIDE_UNIVERSE_SPARSE = 6.0
_MAX_TOTAL_SECONDS = 45.0


@pytest.mark.perf
def test_fixed_weight_ten_asset_monthly_rebalance_canonical_sim_warms_quickly() -> None:
    """~2y daily OHLC, 10 names, identical targets on month-end sparse rows (time-based ME)."""
    idx = pd.bdate_range("2022-01-03", periods=520, freq="B")
    rng = np.random.default_rng(42)
    prices = 80.0 + np.cumsum(rng.normal(0, 0.35, (len(idx), 10)), axis=0)
    tickers = [f"EW{i}" for i in range(10)]
    daily = pd.DataFrame(prices, index=idx, columns=tickers).clip(lower=1.0)
    rets = daily.pct_change(fill_method=None).fillna(0.0)

    month_end = idx.to_series().groupby(idx.to_period("M")).max().values
    month_end_ix = pd.DatetimeIndex(month_end)
    n_months = len(month_end_ix)
    sized = pd.DataFrame(np.full((n_months, 10), 0.1), index=month_end_ix, columns=tickers)

    scenario = {
        "timing_config": {"mode": "time_based", "rebalance_frequency": "ME"},
        "costs_config": {"transaction_costs_bps": 8.0},
        "allocation_mode": "reinvestment",
    }
    g = {
        "portfolio_value": 250_000.0,
        "commission_per_share": 0.0,
        "commission_min_per_order": 0.0,
        "commission_max_percent_of_trade": 0.0,
        "slippage_bps": 0.0,
    }

    for _ in range(_WARM_ITERATIONS):
        _, _ = calculate_portfolio_returns(
            sized, scenario, daily, rets, tickers, g, track_trades=False
        )

    t0 = time.perf_counter()
    for _ in range(_MEASURE_ITERATIONS):
        _, _ = calculate_portfolio_returns(
            sized, scenario, daily, rets, tickers, g, track_trades=False
        )
    elapsed = time.perf_counter() - t0
    mean_s = elapsed / float(_MEASURE_ITERATIONS)
    assert mean_s <= _MAX_MEAN_SECONDS_FIXED_WEIGHT_MONTHLY
    assert elapsed <= _MAX_TOTAL_SECONDS


@pytest.mark.perf
def test_wide_universe_sparse_signal_events_canonical_sim_warms_quickly() -> None:
    """50 names, sparse rebalance rows every ~15 sessions (signal_based-style grid)."""
    idx = pd.bdate_range("2021-06-01", periods=320, freq="B")
    rng = np.random.default_rng(7)
    prices = 50.0 * np.cumprod(1.0 + rng.normal(0.0, 0.008, (len(idx), 50)), axis=0)
    tickers = [f"U{i:02d}" for i in range(50)]
    daily = pd.DataFrame(prices, index=idx, columns=tickers)
    rets = daily.pct_change(fill_method=None).fillna(0.0)

    sized = pd.DataFrame(np.nan, index=idx, columns=tickers, dtype=float)
    for i in range(0, len(idx), 15):
        w = rng.random(50)
        sized.iloc[i, :] = w / w.sum()

    scenario = {
        "timing_config": {
            "mode": "signal_based",
            "rebalance_frequency": "D",
            "trade_execution_timing": "bar_close",
        },
        "costs_config": {"transaction_costs_bps": 5.0},
        "allocation_mode": "reinvestment",
    }
    g = {
        "portfolio_value": 500_000.0,
        "commission_per_share": 0.0,
        "commission_min_per_order": 0.0,
        "commission_max_percent_of_trade": 0.0,
        "slippage_bps": 0.0,
    }

    for _ in range(_WARM_ITERATIONS):
        _, _ = calculate_portfolio_returns(
            sized, scenario, daily, rets, tickers, g, track_trades=False
        )

    t0 = time.perf_counter()
    for _ in range(_MEASURE_ITERATIONS):
        _, _ = calculate_portfolio_returns(
            sized, scenario, daily, rets, tickers, g, track_trades=False
        )
    elapsed = time.perf_counter() - t0
    mean_s = elapsed / float(_MEASURE_ITERATIONS)
    assert mean_s <= _MAX_MEAN_SECONDS_WIDE_UNIVERSE_SPARSE
    assert elapsed <= _MAX_TOTAL_SECONDS
