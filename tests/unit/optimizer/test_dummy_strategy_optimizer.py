import numpy as np
import pandas as pd
from unittest.mock import Mock
import pytest

from portfolio_backtester.backtester_logic.backtester_facade import BacktesterFacade as Backtester


@pytest.mark.fast
def test_dummy_strategy_single_trial_optimizer():
    """Run optimisation with a single trial to ensure objective is non-zero and no crash."""
    rng = np.random.default_rng(11)
    dates = pd.date_range("2015-01-01", periods=2500, freq="D")  # ~7 years of data for WFO
    prices = 100 * np.cumprod(1 + rng.normal(0.00025, 0.011, len(dates)))
    volumes = rng.integers(1000, 8000, len(dates))

    columns = pd.MultiIndex.from_tuples(
        [("SPY", "Close"), ("SPY", "Volume")], names=["Ticker", "Field"]
    )
    daily_data = pd.DataFrame(np.column_stack([prices, volumes]), index=dates, columns=columns)
    benchmark_data = pd.DataFrame({"Close": prices}, index=dates)

    config = {
        "GLOBAL_CONFIG": {
            "benchmark": "SPY",
            "data_source": "memory",
            "start_date": "2015-01-01",
            "end_date": "2021-12-31",
            "output_dir": "test_output",
            "universe": ["SPY"],
            "data_source_config": {
                "data_frames": {"daily_data": daily_data, "benchmark_data": benchmark_data},
            },
        },
        "BACKTEST_SCENARIOS": [
            {
                "name": "dummy_opt",
                "strategy": "SimpleMetaStrategy",
                "strategy_params": {
                    "initial_capital": 1000000.0,
                    "min_allocation": 0.05,
                    "rebalance_threshold": 0.05,
                    "allocations": [
                        {
                            "strategy_id": "momentum",
                            "strategy_class": "SimpleMomentumPortfolioStrategy",
                            "strategy_params": {"lookback_period": 12},
                            "weight": 1.0,
                        }
                    ],
                },
                "position_sizer": "direct",
                "timing_config": {"mode": "time_based", "rebalance_frequency": "D"},
                "universe_config": {"type": "fixed", "tickers": ["SPY"]},
                "train_window_months": 24,
                "test_window_months": 6,
                "optimization": {
                    "enabled": True,
                    "metric": "total_return",
                    "n_trials": 1,
                    "timeout": 30,
                    "parameter_space": {
                        "min_allocation": {"type": "float", "low": 0.01, "high": 0.2},
                        "rebalance_threshold": {"type": "float", "low": 0.01, "high": 0.1},
                    },
                },
            }
        ],
    }

    args = Mock(
        scenario_name="dummy_opt",
        mode="optimize",
        timeout=60,
        n_jobs=1,
        early_stop_patience=3,
        optimizer="optuna",
        study_name=None,
        optuna_trials=1,
        pruning_enabled=False,
    )

    backtester = Backtester(
        global_config=config["GLOBAL_CONFIG"], scenarios=config["BACKTEST_SCENARIOS"], args=args
    )
    backtester.run()

    result_series = backtester.results["dummy_opt"]["returns"]
    assert not result_series.empty
    objective = (1 + result_series).prod() - 1
    assert objective != 0
