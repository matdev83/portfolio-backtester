import argparse
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd

from portfolio_backtester.backtester_logic.optimization_orchestrator import (
    OptimizationOrchestrator,
)
from portfolio_backtester.optimization.results import OptimizationResult
from portfolio_backtester.optimization.wfo_window import WFOWindow
from portfolio_backtester.scenario_normalizer import ScenarioNormalizer


def _sample_data():
    dates = pd.date_range("2020-01-01", "2021-12-31", freq="B")
    daily_data = pd.DataFrame({"SPY": np.linspace(100, 120, len(dates))}, index=dates)
    monthly_data = daily_data.resample("ME").last()
    rets_full = daily_data.pct_change(fill_method=None).fillna(0.0)
    return monthly_data, daily_data, rets_full


def _minimal_optimize_specs():
    return [
        {
            "parameter": "open_long_prob",
            "type": "float",
            "min_value": 0.05,
            "max_value": 0.15,
        }
    ]


def test_get_wfo_mode_defaults_cv_without_global_or_scenario_override():
    orchestrator = OptimizationOrchestrator(
        global_config={},
        data_source=Mock(),
        backtest_runner=Mock(),
        evaluation_engine=Mock(),
        rng=np.random.default_rng(42),
    )
    raw = {
        "name": "wfo_cv_default",
        "strategy": "DummyStrategyForTestingSignalStrategy",
        "strategy_params": {},
        "optimize": _minimal_optimize_specs(),
        "train_window_months": 6,
        "test_window_months": 3,
    }
    canon = ScenarioNormalizer().normalize(scenario=raw, global_config={})
    assert orchestrator._get_wfo_mode(canon) == "cv"


def test_get_wfo_mode_preserves_explicit_global_reoptimize():
    orchestrator = OptimizationOrchestrator(
        global_config={"wfo_mode": "reoptimize"},
        data_source=Mock(),
        backtest_runner=Mock(),
        evaluation_engine=Mock(),
        rng=np.random.default_rng(42),
    )
    raw = {
        "name": "wfo_global_reopt",
        "strategy": "DummyStrategyForTestingSignalStrategy",
        "strategy_params": {},
        "optimize": _minimal_optimize_specs(),
        "train_window_months": 6,
        "test_window_months": 3,
    }
    canon = ScenarioNormalizer().normalize(scenario=raw, global_config={})
    assert orchestrator._get_wfo_mode(canon) == "reoptimize"


def test_get_wfo_mode_unknown_token_defaults_cv():
    orchestrator = OptimizationOrchestrator(
        global_config={"wfo_mode": "not_a_real_mode"},
        data_source=Mock(),
        backtest_runner=Mock(),
        evaluation_engine=Mock(),
        rng=np.random.default_rng(42),
    )
    raw = {
        "name": "wfo_bad_mode",
        "strategy": "DummyStrategyForTestingSignalStrategy",
        "strategy_params": {},
        "optimize": _minimal_optimize_specs(),
        "train_window_months": 6,
        "test_window_months": 3,
    }
    canon = ScenarioNormalizer().normalize(scenario=raw, global_config={})
    assert orchestrator._get_wfo_mode(canon) == "cv"


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
        "optimize": _minimal_optimize_specs(),
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


def test_run_optimization_without_wfo_config_skips_reoptimize_path():
    global_config = {"benchmark": "SPY"}
    orchestrator = OptimizationOrchestrator(
        global_config=global_config,
        data_source=Mock(),
        backtest_runner=Mock(),
        evaluation_engine=Mock(),
        rng=np.random.default_rng(42),
    )

    scenario_config = {
        "name": "no_wfo_keys",
        "strategy": "DummyStrategyForTestingSignalStrategy",
        "strategy_params": {},
        "optimize": _minimal_optimize_specs(),
        "train_window_months": 6,
        "test_window_months": 3,
    }

    monthly_data, daily_data, rets_full = _sample_data()
    args = argparse.Namespace(optimizer="optuna", optuna_trials=2)

    stub_result = OptimizationResult(
        best_parameters={"open_long_prob": 0.1},
        best_value=1.0,
        n_evaluations=3,
        optimization_history=[],
        stitched_returns=pd.Series(dtype=float),
    )

    ts = pd.Timestamp("2020-06-01")
    windows = [
        WFOWindow(
            train_start=ts,
            train_end=ts + pd.DateOffset(months=6),
            test_start=ts + pd.DateOffset(months=6),
            test_end=ts + pd.DateOffset(months=9),
        ),
        WFOWindow(
            train_start=ts + pd.DateOffset(months=3),
            train_end=ts + pd.DateOffset(months=9),
            test_start=ts + pd.DateOffset(months=9),
            test_end=ts + pd.DateOffset(months=12),
        ),
    ]

    mock_inner = MagicMock()
    mock_inner.optimize.return_value = stub_result

    with (
        patch.object(OptimizationOrchestrator, "_run_reoptimized_wfo") as mock_reopt,
        patch(
            "portfolio_backtester.optimization.orchestrator_factory.create_orchestrator",
            return_value=mock_inner,
        ),
        patch(
            "portfolio_backtester.optimization.factory.create_parameter_generator"
        ),
        patch(
            "portfolio_backtester.backtesting.strategy_backtester.StrategyBacktester"
        ),
        patch(
            "portfolio_backtester.utils.generate_enhanced_wfo_windows",
            return_value=windows,
        ),
    ):
        result = orchestrator.run_optimization(
            scenario_config, monthly_data, daily_data, rets_full, args
        )

    mock_reopt.assert_not_called()
    mock_inner.optimize.assert_called_once()
    assert result.wfo_mode == "cv"


def test_run_optimization_explicit_is_wfo_false_forces_cv_despite_global_reoptimize():
    global_config = {"benchmark": "SPY", "wfo_mode": "reoptimize"}
    orchestrator = OptimizationOrchestrator(
        global_config=global_config,
        data_source=Mock(),
        backtest_runner=Mock(),
        evaluation_engine=Mock(),
        rng=np.random.default_rng(42),
    )

    scenario_config = {
        "name": "disable_wfo",
        "strategy": "DummyStrategyForTestingSignalStrategy",
        "strategy_params": {},
        "optimize": _minimal_optimize_specs(),
        "train_window_months": 6,
        "test_window_months": 3,
        "is_wfo": False,
    }

    monthly_data, daily_data, rets_full = _sample_data()
    args = argparse.Namespace(optimizer="optuna", optuna_trials=2)

    stub_result = OptimizationResult(
        best_parameters={"open_long_prob": 0.1},
        best_value=1.0,
        n_evaluations=2,
        optimization_history=[],
        stitched_returns=pd.Series(dtype=float),
    )

    ts = pd.Timestamp("2020-06-01")
    windows = [
        WFOWindow(
            train_start=ts,
            train_end=ts + pd.DateOffset(months=6),
            test_start=ts + pd.DateOffset(months=6),
            test_end=ts + pd.DateOffset(months=9),
        ),
        WFOWindow(
            train_start=ts + pd.DateOffset(months=3),
            train_end=ts + pd.DateOffset(months=9),
            test_start=ts + pd.DateOffset(months=9),
            test_end=ts + pd.DateOffset(months=12),
        ),
    ]

    mock_inner = MagicMock()
    mock_inner.optimize.return_value = stub_result

    with (
        patch.object(OptimizationOrchestrator, "_run_reoptimized_wfo") as mock_reopt,
        patch(
            "portfolio_backtester.optimization.orchestrator_factory.create_orchestrator",
            return_value=mock_inner,
        ),
        patch(
            "portfolio_backtester.optimization.factory.create_parameter_generator"
        ),
        patch(
            "portfolio_backtester.backtesting.strategy_backtester.StrategyBacktester"
        ),
        patch(
            "portfolio_backtester.utils.generate_enhanced_wfo_windows",
            return_value=windows,
        ),
    ):
        result = orchestrator.run_optimization(
            scenario_config, monthly_data, daily_data, rets_full, args
        )

    mock_reopt.assert_not_called()
    assert result.wfo_mode == "cv"
