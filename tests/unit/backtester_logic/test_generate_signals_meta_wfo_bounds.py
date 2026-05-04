"""Integration-style tests: meta ``generate_signals`` respects scenario/WFO date windows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_backtester.backtester_logic.strategy_logic import (
    _resolve_signal_scan_window,
    generate_signals,
)
from portfolio_backtester.scenario_normalizer import ScenarioNormalizer
from portfolio_backtester.strategies.builtins.meta.simple_meta_strategy import SimpleMetaStrategy


def _daily_ohlc_2023() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    assets = ["AAPL", "MSFT", "GOOGL", "SPY"]
    columns = pd.MultiIndex.from_product(
        [assets, ["Open", "High", "Low", "Close", "Volume"]], names=["Ticker", "Field"]
    )
    rng = np.random.default_rng(42)
    data = rng.standard_normal((len(dates), len(columns))) * 0.02 + 1.0
    data = np.cumprod(data, axis=0) * 100.0
    df = pd.DataFrame(data, index=dates, columns=columns)
    for t in assets:
        df[(t, "High")] = df[[(t, "Open"), (t, "Close")]].max(axis=1) * (
            1.0 + rng.random(len(dates)) * 0.01
        )
        df[(t, "Low")] = df[[(t, "Open"), (t, "Close")]].min(axis=1) * (
            1.0 - rng.random(len(dates)) * 0.01
        )
        df[(t, "Volume")] = rng.integers(1_000_000, 10_000_000, len(dates))
    return df


def test_generate_signals_meta_respects_wfo_window_narrower_than_ohlc() -> None:
    """Meta branch must clip rebalance schedule to WFO/scenario bounds like standard strategies."""

    price_data_daily_ohlc = _daily_ohlc_2023()
    meta_cfg = {
        "initial_capital": 1_000_000,
        "allocations": [
            {
                "strategy_id": "momentum",
                "strategy_class": "CalmarMomentumPortfolioStrategy",
                "strategy_params": {
                    "rolling_window": 3,
                    "num_holdings": 2,
                    "price_column_asset": "Close",
                    "price_column_benchmark": "Close",
                    "timing_config": {"mode": "time_based", "rebalance_frequency": "M"},
                },
                "weight": 0.7,
            },
            {
                "strategy_id": "seasonal",
                "strategy_class": "SeasonalSignalStrategy",
                "strategy_params": {
                    "direction": "long",
                    "entry_day": 5,
                    "hold_days": 5,
                    "price_column_asset": "Close",
                    "trade_longs": True,
                    "trade_shorts": False,
                    "timing_config": {"mode": "signal_based"},
                },
                "weight": 0.3,
            },
        ],
    }
    scenario_raw = {
        "name": "meta_wfo_bounds",
        "strategy": "SimpleMetaStrategy",
        "strategy_params": meta_cfg,
        "timing_config": {"mode": "time_based", "rebalance_frequency": "M"},
        "wfo_start_date": "2023-06-01",
        "wfo_end_date": "2023-08-31",
    }
    normalizer = ScenarioNormalizer()
    canonical = normalizer.normalize(scenario=scenario_raw, global_config={})
    exp_start, exp_end, *_ = _resolve_signal_scan_window(canonical, price_data_daily_ohlc.index)

    meta = SimpleMetaStrategy(meta_cfg)
    signals = generate_signals(
        strategy=meta,
        scenario_config=scenario_raw,
        price_data_daily_ohlc=price_data_daily_ohlc,
        universe_tickers=["AAPL", "MSFT", "GOOGL"],
        benchmark_ticker="SPY",
        has_timed_out=lambda: False,
    )

    assert not signals.empty
    assert signals.index.min() >= exp_start
    assert signals.index.max() <= exp_end
    assert bool(signals.index.isin(price_data_daily_ohlc.index).all())
    assert signals.index.min() >= pd.Timestamp("2023-06-01")
    assert signals.index.max() <= pd.Timestamp("2023-08-31")
