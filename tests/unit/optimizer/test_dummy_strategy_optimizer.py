import numpy as np
import pandas as pd
from unittest.mock import Mock
import pytest

from src.portfolio_backtester.core import Backtester

@pytest.mark.fast
def test_dummy_strategy_single_trial_optimizer():
    """Run optimisation with a single trial to ensure objective is non-zero and no crash."""
    rng = np.random.default_rng(11)
    dates = pd.date_range("2020-01-01", periods=120, freq="D")
    prices = 100 * np.cumprod(1 + rng.normal(0.00025, 0.011, len(dates)))
    volumes = rng.integers(1000, 8000, len(dates))

    columns = pd.MultiIndex.from_tuples([("SPY", "Close"), ("SPY", "Volume")], names=["Ticker", "Field"])
    daily_data = pd.DataFrame(np.column_stack([prices, volumes]), index=dates, columns=columns)
    benchmark_data = pd.DataFrame({"Close": prices}, index=dates)

    config = {
        "GLOBAL_CONFIG": {
            "benchmark": "SPY",
            "data_source": "memory",
            "start_date": "2020-01-01",
            "end_date": "2020-04-30",
            "output_dir": "test_output",
            "universe": ["SPY"],
            "data_source_config": {
                "data_frames": {"daily_data": daily_data, "benchmark_data": benchmark_data},
            },
        },
        "BACKTEST_SCENARIOS": [
            {
                "name": "dummy_opt",
                "strategy": "dummy",
                "strategy_params": {"symbol": "SPY", "seed": 77},
                "position_sizer": "direct",
                "timing_config": {"mode": "time_based", "rebalance_frequency": "D"},
                "universe_config": {"type": "fixed", "tickers": ["SPY"]},
                "optimization": {
                    "enabled": True,
                    "metric": "total_return",
                    "n_trials": 1,
                    "timeout": 30,
                    "parameter_space": {
                        "open_long_prob": {"type": "float", "low": 0.05, "high": 0.5},
                        "close_long_prob": {"type": "float", "low": 0.01, "high": 0.2},
                    },
                },
            }
        ],
    }

    args = Mock(scenario_name="dummy_opt", optimize_mode=True, timeout=60, n_jobs=1, early_stop_patience=3)

    backtester = Backtester(global_config=config["GLOBAL_CONFIG"], scenarios=config["BACKTEST_SCENARIOS"], args=args)
    backtester.run()

    result_series = backtester.results["dummy_opt"]["returns"]
    assert not result_series.empty
    objective = (1 + result_series).prod() - 1
    assert objective != 0
