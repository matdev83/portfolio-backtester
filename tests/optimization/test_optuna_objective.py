import pytest
from unittest.mock import MagicMock, patch
from src.portfolio_backtester.optimization.optuna_objective import build_objective

def test_build_objective_with_config():
    g_cfg = {"benchmark": "SPY"}
    base_scen_cfg = {"strategy_params": {"max_lookback": 50, "leverage": 1.0},
                     "optimize": [
                         {"parameter": "max_lookback"},
                         {"parameter": "leverage"}
                     ]}
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
        features_slice,
    )

    trial = MagicMock()
    with patch('src.portfolio_backtester.optimization.optuna_objective._run_scenario_static', return_value=MagicMock()), \
         patch('src.portfolio_backtester.optimization.optuna_objective.calculate_metrics', return_value={"Calmar": 1.0}):
        
        objective_fn(trial)

    trial.suggest_int.assert_called_with("max_lookback", 20, 252, step=10)
    trial.suggest_float.assert_called_with("leverage", 0.5, 2.0, step=0.1, log=False)
