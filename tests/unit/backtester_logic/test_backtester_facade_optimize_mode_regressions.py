from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch

import pandas as pd

from portfolio_backtester.backtester_logic.backtester_facade import Backtester
from portfolio_backtester.canonical_config import CanonicalScenarioConfig


def test_run_optimize_mode_stitched_results_store_only_optimized_key() -> None:
    facade = SimpleNamespace()
    facade.global_config = {"benchmark": "SPY"}
    facade.data_source = Mock()
    facade.args = SimpleNamespace()
    facade.results = {}

    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    daily_data = pd.DataFrame({"SPY": [100.0, 101.0, 102.0]}, index=idx)
    monthly_data = pd.DataFrame()
    rets_full = pd.DataFrame()

    optimization_result = SimpleNamespace(
        stitched_returns=pd.Series([0.01, -0.005, 0.02], index=idx),
        best_parameters={"num_holdings": 12},
        n_evaluations=10,
        best_trial=None,
        wfo_mode="reoptimize",
        wfo_window_params=[{"num_holdings": 12}],
        wfo_window_results=[],
    )

    optimization_orchestrator = Mock()
    optimization_orchestrator.run_optimization.return_value = optimization_result

    scenario_config = SimpleNamespace(name="simple_momentum_strategy", extras={})

    with (
        patch(
            "portfolio_backtester.reporting.performance_metrics.calculate_metrics",
            return_value=pd.Series({"Sharpe": 1.0}),
        ),
        patch(
            "portfolio_backtester.backtesting.strategy_backtester.StrategyBacktester"
        ) as mock_strategy_backtester,
    ):
        mock_strategy_backtester.return_value._create_performance_stats.return_value = {}
        mock_strategy_backtester.return_value._create_charts_data.return_value = {}

        Backtester._run_optimize_mode(
            cast(Backtester, facade),
            optimization_orchestrator,
            cast(CanonicalScenarioConfig, scenario_config),
            monthly_data,
            daily_data,
            rets_full,
        )

    assert "simple_momentum_strategy_Optimized" in facade.results
    assert "simple_momentum_strategy" not in facade.results
