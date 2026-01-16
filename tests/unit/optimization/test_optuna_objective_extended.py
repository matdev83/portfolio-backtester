import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import optuna

from portfolio_backtester.optimization.optuna_objective import build_objective

@pytest.fixture
def mock_data():
    dates = pd.date_range("2023-01-01", periods=20, freq="B")
    monthly_dates = pd.date_range("2023-01-01", periods=2, freq="ME")
    
    daily_data = pd.DataFrame(
        100.0 + np.random.randn(len(dates), 2),
        index=dates,
        columns=["A", "B"]
    )
    monthly_data = pd.DataFrame(
        100.0 + np.random.randn(len(monthly_dates), 2),
        index=monthly_dates,
        columns=["A", "B"]
    )
    daily_rets = daily_data.pct_change().fillna(0.0)
    
    return {
        "monthly": monthly_data,
        "daily": daily_data,
        "daily_rets": daily_rets,
        "bench": daily_data["A"] # Self benchmark
    }

def test_build_objective_parameter_suggestion(mock_data):
    # Setup config
    g_cfg = {"benchmark": "A", "portfolio_value": 100000.0}
    scen_cfg = {
        "strategy": "TestStrategy",
        "strategy_params": {"p1": 1},
        "optimize": [
            {"parameter": "p1", "type": "int", "min_value": 1, "max_value": 10},
            {"parameter": "p2", "type": "float", "min_value": 0.1, "max_value": 1.0},
            {"parameter": "p3", "type": "categorical", "values": ["x", "y"]}
        ],
        "optimization_metric": "Sharpe",
        "rebalance_frequency": "M"
    }
    
    # Mock dependencies
    with patch("portfolio_backtester.optimization.optuna_objective._resolve_strategy") as mock_resolve, \
         patch("portfolio_backtester.optimization.optuna_objective._run_scenario_static") as mock_run:
        
        mock_cls = MagicMock()
        mock_cls.tunable_parameters.return_value = {"p1", "p2", "p3"}
        mock_resolve.return_value = mock_cls
        
        # Mock run to return positive sharpe
        rets = pd.Series(0.01, index=mock_data["daily"].index)
        mock_run.return_value = rets
        
        # Build objective
        objective = build_objective(
            g_cfg, scen_cfg,
            mock_data["monthly"],
            mock_data["daily"],
            mock_data["daily_rets"],
            mock_data["bench"],
            None
        )
        
        # Run with mock trial
        trial = MagicMock(spec=optuna.trial.Trial)
        trial.suggest_int.return_value = 5
        trial.suggest_float.return_value = 0.5
        trial.suggest_categorical.return_value = "x"
        
        result = objective(trial)
        
        # Verify calls
        trial.suggest_int.assert_called_with("p1", 1, 10, step=1)
        trial.suggest_float.assert_called_with("p2", 0.1, 1.0, step=None, log=False)
        trial.suggest_categorical.assert_called_with("p3", ["x", "y"])
        
        # Result should be float (Sharpe)
        assert isinstance(result, float)

def test_build_objective_constraints(mock_data):
    g_cfg = {"benchmark": "A"}
    scen_cfg = {
        "strategy": "TestStrategy",
        "strategy_params": {},
        "optimize": [],
        "optimization_metric": "Sharpe",
        "optimization_constraints": [
            {"metric": "Max Drawdown", "operator": "LT", "value": 0.2} # Constraint: MDD < 0.2
        ],
        "rebalance_frequency": "M"
    }
    
    with patch("portfolio_backtester.optimization.optuna_objective._resolve_strategy"), \
         patch("portfolio_backtester.optimization.optuna_objective._run_scenario_static") as mock_run, \
         patch("portfolio_backtester.optimization.optuna_objective.calculate_metrics") as mock_calc:
        
        mock_run.return_value = pd.Series(0.0, index=mock_data["daily"].index)
        
        # Case 1: Constraint violated (MDD = 0.3)
        mock_calc.return_value = {"Sharpe": 1.0, "Max Drawdown": 0.3}
        
        objective = build_objective(
            g_cfg, scen_cfg,
            mock_data["monthly"],
            mock_data["daily"],
            mock_data["daily_rets"],
            mock_data["bench"],
            None
        )
        
        trial = MagicMock()
        result = objective(trial)
        assert result == float("-inf") # Penalty
        
        # Case 2: Constraint met (MDD = 0.1)
        mock_calc.return_value = {"Sharpe": 1.0, "Max Drawdown": 0.1}
        result = objective(trial)
        assert result == 1.0

def test_build_objective_multi_objective(mock_data):
    scen_cfg = {
        "strategy": "TestStrategy",
        "strategy_params": {},
        "optimization_targets": [
            {"name": "Sharpe", "direction": "maximize"},
            {"name": "Max Drawdown", "direction": "minimize"}
        ],
        "rebalance_frequency": "M"
    }
    g_cfg = {"benchmark": "A"}

    with patch("portfolio_backtester.optimization.optuna_objective._resolve_strategy"), \
         patch("portfolio_backtester.optimization.optuna_objective._run_scenario_static"), \
         patch("portfolio_backtester.optimization.optuna_objective.calculate_metrics") as mock_calc:
        
        mock_calc.return_value = {"Sharpe": 1.5, "Max Drawdown": 0.1}
        
        objective = build_objective(
            g_cfg, scen_cfg,
            mock_data["monthly"],
            mock_data["daily"],
            mock_data["daily_rets"],
            mock_data["bench"],
            None
        )
        
        trial = MagicMock()
        result = objective(trial)
        
        assert isinstance(result, tuple)
        assert result == (1.5, 0.1)

def test_build_objective_invalid_param_type(mock_data):
    scen_cfg = {
        "strategy": "TestStrategy",
        "strategy_params": {},
        "optimize": [{"parameter": "p1", "type": "unknown"}],
        "rebalance_frequency": "M"
    }
    g_cfg = {"benchmark": "A"}
    
    with patch("portfolio_backtester.optimization.optuna_objective._resolve_strategy") as mock_res:
         mock_res.return_value.tunable_parameters.return_value = {"p1"}
         
         objective = build_objective(
            g_cfg, scen_cfg,
            mock_data["monthly"],
            mock_data["daily"],
            mock_data["daily_rets"],
            mock_data["bench"],
            None
        )
         
         trial = MagicMock()
         # Should raise ValueError
         with pytest.raises(ValueError, match="Unsupported parameter type"):
             objective(trial)
