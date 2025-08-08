import numpy as np
import pandas as pd
from unittest.mock import Mock
import pytest

from portfolio_backtester.core import Backtester


@pytest.mark.integration
@pytest.mark.fast
def test_dummy_meta_strategy_end_to_end():
    """Meta-strategy that wraps a real registered sub-strategy should work and produce trades."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-01", periods=150, freq="D")
    prices = 50 * np.cumprod(1 + rng.normal(0.0003, 0.012, len(dates)))
    volumes = rng.integers(800, 9000, len(dates))

    columns = pd.MultiIndex.from_tuples(
        [
            ("SPY", "Close"),
            ("SPY", "Volume"),
        ],
        names=["Ticker", "Field"],
    )
    daily_data = pd.DataFrame(np.column_stack([prices, volumes]), index=dates, columns=columns)
    benchmark_data = pd.DataFrame({"Close": prices}, index=dates)

    config = {
        "GLOBAL_CONFIG": {
            "benchmark": "SPY",
            "data_source": "memory",
            "start_date": "2020-01-01",
            "end_date": "2020-05-30",
            "output_dir": "test_output",
            "universe": ["SPY"],
            "data_source_config": {
                "data_frames": {
                    "daily_data": daily_data,
                    "benchmark_data": benchmark_data,
                }
            },
        },
        "BACKTEST_SCENARIOS": [
            {
                "name": "meta_dummy",
                "strategy": "simple_meta",  # alias for SimpleMetaStrategy
                "strategy_params": {
                    "allocations": [
                        {
                            "strategy_id": "dummy1",
                            "strategy_class": "EmaCrossoverSignalStrategy",
                            "strategy_params": {
                                "fast_ema_days": 12,
                                "slow_ema_days": 26,
                                "leverage": 1.0,
                            },
                            "weight": 1.0,
                        }
                    ]
                },
                "position_sizer": "direct",
                "timing_config": {"mode": "time_based", "rebalance_frequency": "D"},
                "universe_config": {"type": "fixed", "tickers": ["SPY"]},
            }
        ],
    }

    backtester = Backtester(
        global_config=config["GLOBAL_CONFIG"],
        scenarios=config["BACKTEST_SCENARIOS"],
        args=Mock(scenario_name="meta_dummy"),
    )
    backtester.run()

    results = backtester.results["meta_dummy"]
    returns = results["returns"]
    assert not returns.empty
    assert (1 + returns).prod() - 1 != 0
    trade_stats = results["trade_stats"] or {}
    assert trade_stats.get("all_num_trades", 0) > 0
    max_dd = results["performance_stats"].get("max_drawdown", 0)
    assert max_dd != 0
