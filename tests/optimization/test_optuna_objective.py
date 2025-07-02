import pytest
from unittest.mock import MagicMock, patch
from src.portfolio_backtester.optimization.optuna_objective import build_objective

def test_build_objective_with_config():
    g_cfg = {"benchmark": "SPY"}
    base_scen_cfg = {
        "strategy_params": {"max_lookback": 50, "leverage": 1.0},
        "optimization_metric": "Sharpe",  # New scenario-level metric
        "optimize": [
            {"parameter": "max_lookback"},
            {"parameter": "leverage"}
        ]
    }
    train_data_monthly = MagicMock()
    train_data_daily = MagicMock()
    train_rets_daily = MagicMock()
    bench_series_daily = MagicMock()
    features_slice = MagicMock()

    objective_fn = build_objective(
        g_cfg,
        base_scen_cfg,
        train_data_monthly,
        train_data_daily,
        train_rets_daily,
        bench_series_daily,
        features_slice
        # metric argument removed
    )

    trial = MagicMock()
    mock_calculate_metrics = MagicMock(return_value={"Sharpe": 1.0, "Calmar": 0.5}) # Ensure it contains the target metric

    with patch('src.portfolio_backtester.optimization.optuna_objective._run_scenario_static', return_value=MagicMock()), \
         patch('src.portfolio_backtester.optimization.optuna_objective.calculate_metrics', mock_calculate_metrics):
        
        result = objective_fn(trial)

    trial.suggest_int.assert_called_with("max_lookback", 20, 252, step=10) # These defaults come from OPTIMIZER_PARAMETER_DEFAULTS
    trial.suggest_float.assert_called_with("leverage", 0.5, 2.0, step=0.1, log=False) # These defaults come from OPTIMIZER_PARAMETER_DEFAULTS

    # Assert that calculate_metrics was called and the value for "Sharpe" (from base_scen_cfg) was returned
    mock_calculate_metrics.assert_called_once()
    assert result == 1.0

    # Test with a different metric to be sure
    base_scen_cfg["optimization_metric"] = "Calmar"
    objective_fn_calmar = build_objective(
        g_cfg,
        base_scen_cfg,
        train_data_monthly,
        train_data_daily,
        train_rets_daily,
        bench_series_daily,
        features_slice
    )
    with patch('src.portfolio_backtester.optimization.optuna_objective._run_scenario_static', return_value=MagicMock()), \
         patch('src.portfolio_backtester.optimization.optuna_objective.calculate_metrics', mock_calculate_metrics):
        mock_calculate_metrics.reset_mock() # Reset mock for the second call
        result_calmar = objective_fn_calmar(trial)

    mock_calculate_metrics.assert_called_once()
    assert result_calmar == 0.5
