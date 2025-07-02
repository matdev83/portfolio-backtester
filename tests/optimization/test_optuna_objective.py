import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.portfolio_backtester.optimization.optuna_objective import build_objective

# Common setup for tests
@pytest.fixture
def common_mocks():
    g_cfg = {"benchmark": "SPY"}
    train_data_monthly = MagicMock()
    train_data_daily = MagicMock()
    train_rets_daily = MagicMock()
    bench_series_daily = MagicMock()
    features_slice = MagicMock()
    trial = MagicMock()
    return g_cfg, train_data_monthly, train_data_daily, train_rets_daily, bench_series_daily, features_slice, trial

def test_build_objective_single_metric(common_mocks):
    g_cfg, train_data_monthly, train_data_daily, train_rets_daily, bench_series_daily, features_slice, trial = common_mocks

    base_scen_cfg = {
        "strategy_params": {"max_lookback": 50, "leverage": 1.0},
        "optimization_metric": "Sharpe",
        "optimize": [
            {"parameter": "max_lookback"},
            {"parameter": "leverage"}
        ]
    }

    objective_fn = build_objective(
        g_cfg, base_scen_cfg, train_data_monthly, train_data_daily,
        train_rets_daily, bench_series_daily, features_slice
    )

    mock_metrics_result = {"Sharpe": 1.5, "Calmar": 0.8}
    mock_calculate_metrics = MagicMock(return_value=mock_metrics_result)

    with patch('src.portfolio_backtester.optimization.optuna_objective._run_scenario_static', return_value=MagicMock()), \
         patch('src.portfolio_backtester.optimization.optuna_objective.calculate_metrics', mock_calculate_metrics):
        result = objective_fn(trial)

    trial.suggest_int.assert_called_with("max_lookback", 20, 252, step=10)
    trial.suggest_float.assert_called_with("leverage", 0.5, 2.0, step=0.1, log=False)
    mock_calculate_metrics.assert_called_once()
    assert result == 1.5

def test_build_objective_multi_metric(common_mocks):
    g_cfg, train_data_monthly, train_data_daily, train_rets_daily, bench_series_daily, features_slice, trial = common_mocks

    base_scen_cfg = {
        "strategy_params": {"param1": 10},
        "optimization_targets": [
            {"name": "Total Return", "direction": "maximize"},
            {"name": "Max Drawdown", "direction": "minimize"}
        ],
        "optimize": [{"parameter": "param1"}] # Assuming param1 is in OPTIMIZER_PARAMETER_DEFAULTS
    }
    # Mock OPTIMIZER_PARAMETER_DEFAULTS for "param1" if not already present
    with patch('src.portfolio_backtester.optimization.optuna_objective.OPTIMIZER_PARAMETER_DEFAULTS', {"param1": {"type": "int", "low": 1, "high": 20, "step": 1}}):
        objective_fn = build_objective(
            g_cfg, base_scen_cfg, train_data_monthly, train_data_daily,
            train_rets_daily, bench_series_daily, features_slice
        )

    mock_metrics_result = {"Total Return": 0.25, "Max Drawdown": -0.1, "Sharpe": 1.2}
    mock_calculate_metrics = MagicMock(return_value=mock_metrics_result)

    with patch('src.portfolio_backtester.optimization.optuna_objective._run_scenario_static', return_value=MagicMock()), \
         patch('src.portfolio_backtester.optimization.optuna_objective.calculate_metrics', mock_calculate_metrics):
        result = objective_fn(trial)

    trial.suggest_int.assert_called_with("param1", 1, 20, step=1)
    mock_calculate_metrics.assert_called_once()
    assert isinstance(result, tuple)
    assert result == (0.25, -0.1)

def test_build_objective_default_metric_when_none_specified(common_mocks):
    g_cfg, train_data_monthly, train_data_daily, train_rets_daily, bench_series_daily, features_slice, trial = common_mocks

    base_scen_cfg = { # No optimization_metric or optimization_targets
        "strategy_params": {"leverage": 1.0},
        "optimize": [{"parameter": "leverage"}]
    }

    objective_fn = build_objective(
        g_cfg, base_scen_cfg, train_data_monthly, train_data_daily,
        train_rets_daily, bench_series_daily, features_slice
    )

    mock_metrics_result = {"Calmar": 0.7, "Sharpe": 1.1} # Must contain default "Calmar"
    mock_calculate_metrics = MagicMock(return_value=mock_metrics_result)

    with patch('src.portfolio_backtester.optimization.optuna_objective._run_scenario_static', return_value=MagicMock()), \
         patch('src.portfolio_backtester.optimization.optuna_objective.calculate_metrics', mock_calculate_metrics):
        result = objective_fn(trial)

    mock_calculate_metrics.assert_called_once()
    assert result == 0.7 # Should default to Calmar

def test_build_objective_single_metric_invalid_value(common_mocks):
    g_cfg, train_data_monthly, train_data_daily, train_rets_daily, bench_series_daily, features_slice, trial = common_mocks

    base_scen_cfg = {"optimization_metric": "Sharpe", "strategy_params": {}, "optimize": []}
    objective_fn = build_objective(
        g_cfg, base_scen_cfg, train_data_monthly, train_data_daily,
        train_rets_daily, bench_series_daily, features_slice
    )

    mock_metrics_result = {"Sharpe": np.nan}
    mock_calculate_metrics = MagicMock(return_value=mock_metrics_result)

    with patch('src.portfolio_backtester.optimization.optuna_objective._run_scenario_static', return_value=MagicMock()), \
         patch('src.portfolio_backtester.optimization.optuna_objective.calculate_metrics', mock_calculate_metrics):
        result = objective_fn(trial)
    assert result == float("-inf")

    mock_metrics_result_inf = {"Sharpe": np.inf}
    mock_calculate_metrics_inf = MagicMock(return_value=mock_metrics_result_inf)
    with patch('src.portfolio_backtester.optimization.optuna_objective._run_scenario_static', return_value=MagicMock()), \
         patch('src.portfolio_backtester.optimization.optuna_objective.calculate_metrics', mock_calculate_metrics_inf):
        result_inf = objective_fn(trial)
    assert result_inf == float("-inf")


def test_build_objective_multi_metric_invalid_values(common_mocks):
    g_cfg, train_data_monthly, train_data_daily, train_rets_daily, bench_series_daily, features_slice, trial = common_mocks

    base_scen_cfg = {
        "optimization_targets": [
            {"name": "Total Return", "direction": "maximize"},
            {"name": "Max Drawdown", "direction": "minimize"},
            {"name": "Sharpe", "direction": "maximize"}
        ],
        "strategy_params": {}, "optimize": []
    }
    objective_fn = build_objective(
        g_cfg, base_scen_cfg, train_data_monthly, train_data_daily,
        train_rets_daily, bench_series_daily, features_slice
    )

    mock_metrics_result = {"Total Return": np.nan, "Max Drawdown": 0.1, "Sharpe": np.inf}
    mock_calculate_metrics = MagicMock(return_value=mock_metrics_result)

    with patch('src.portfolio_backtester.optimization.optuna_objective._run_scenario_static', return_value=MagicMock()), \
         patch('src.portfolio_backtester.optimization.optuna_objective.calculate_metrics', mock_calculate_metrics):
        result = objective_fn(trial)

    assert isinstance(result, tuple)
    assert len(result) == 3
    assert np.isnan(result[0]) # Total Return was np.nan
    assert result[1] == 0.1     # Max Drawdown was valid
    assert np.isnan(result[2]) # Sharpe was np.inf, should be nan for multi-obj
