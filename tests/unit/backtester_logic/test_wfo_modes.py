import argparse
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from portfolio_backtester.backtester_logic.optimization_orchestrator import (
    OptimizationOrchestrator,
)
from portfolio_backtester.optimization.results import OptimizationResult


def _sample_data():
    dates = pd.date_range("2020-01-01", "2021-12-31", freq="B")
    daily_data = pd.DataFrame({"SPY": np.linspace(100, 120, len(dates))}, index=dates)
    monthly_data = daily_data.resample("ME").last()
    rets_full = daily_data.pct_change(fill_method=None).fillna(0.0)
    return monthly_data, daily_data, rets_full


def test_run_optimization_uses_reoptimize_mode():
    global_config = {"benchmark": "SPY"}
    orchestrator = OptimizationOrchestrator(
        global_config=global_config,
        data_source=Mock(),
        backtest_runner=Mock(),
        evaluation_engine=Mock(),
        rng=np.random.default_rng(42),
    )

    scenario_config = {
        "name": "test_scenario",
        "strategy": "DummyStrategyForTestingSignalStrategy",
        "strategy_params": {},
        "optimize": [
            {
                "parameter": "open_long_prob",
                "type": "float",
                "min_value": 0.05,
                "max_value": 0.15,
            }
        ],
        "train_window_months": 6,
        "test_window_months": 3,
        "wfo_mode": "reoptimize",
    }

    monthly_data, daily_data, rets_full = _sample_data()
    args = argparse.Namespace(optimizer="optuna", optuna_trials=2)

    stub_result = OptimizationResult(
        best_parameters={},
        best_value=-1e9,
        n_evaluations=0,
        optimization_history=[],
        stitched_returns=pd.Series(dtype=float),
    )

    with patch.object(
        OptimizationOrchestrator, "_run_reoptimized_wfo", return_value=stub_result
    ) as mock_reopt:
        result = orchestrator.run_optimization(
            scenario_config, monthly_data, daily_data, rets_full, args
        )

    assert result is stub_result
    mock_reopt.assert_called_once()
