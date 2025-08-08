import numpy as np
import pandas as pd
from unittest.mock import Mock
import pytest

from portfolio_backtester.backtester_logic.backtester_facade import BacktesterFacade as Backtester


@pytest.mark.integration
@pytest.mark.fast
def test_dummy_strategy_end_to_end():
    """End-to-end check that a real registered strategy generates trades, returns and commissions."""
    # -----------------------------
    # Synthetic price data
    # -----------------------------
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    prices = 100 * np.cumprod(1 + rng.normal(0.0002, 0.01, len(dates)))
    volumes = rng.integers(1_000, 10_000, len(dates))

    # MultiIndex columns as expected by data manager utilities
    columns = pd.MultiIndex.from_tuples(
        [
            ("SPY", "Close"),
            ("SPY", "Volume"),
        ],
        names=["Ticker", "Field"],
    )
    data_matrix = np.column_stack([prices, volumes])
    daily_data = pd.DataFrame(data_matrix, index=dates, columns=columns)

    benchmark_data = pd.DataFrame({"Close": prices}, index=dates)

    # -----------------------------
    # Backtest configuration
    # -----------------------------
    config = {
        "GLOBAL_CONFIG": {
            "benchmark": "SPY",
            "data_source": "memory",
            "start_date": "2020-01-01",
            "end_date": "2020-07-19",
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
                "name": "dummy_scenario",
                "strategy": "EmaCrossoverSignalStrategy",
                "strategy_params": {
                    "fast_ema_days": 12,
                    "slow_ema_days": 26,
                    "leverage": 1.0,
                },
                "position_sizer": "direct",  # use raw signal weights
                "timing_config": {
                    "mode": "time_based",
                    "rebalance_frequency": "D",  # daily rebalance for more trades
                },
                "universe_config": {
                    "type": "fixed",
                    "tickers": ["SPY"],
                },
            }
        ],
    }

    # -----------------------------
    # Run backtester
    # -----------------------------
    backtester = Backtester(
        global_config=config["GLOBAL_CONFIG"],
        scenarios=config["BACKTEST_SCENARIOS"],
        args=Mock(scenario_name="dummy_scenario"),
    )
    backtester.run()

    results = backtester.results["dummy_scenario"]

    # -----------------------------
    # Assertions
    # -----------------------------
    returns = results["returns"]
    assert not returns.empty, "Returns series should not be empty"
    total_ret = (1 + returns).prod() - 1
    assert not np.isclose(total_ret, 0.0), "Total return should be non-zero"

    trade_stats = results["trade_stats"] or {}
    assert trade_stats.get("all_num_trades", 0) > 0, "Should record at least one trade"
    assert trade_stats.get("all_total_commissions_paid", 0) > 0, "Commissions paid should be > 0"

    # Drawdown sanity
    max_dd = results["performance_stats"].get("max_drawdown", 0)
    assert max_dd != 0, "Max drawdown should be non-zero"
