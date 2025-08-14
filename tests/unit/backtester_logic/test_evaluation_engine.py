"""
Unit tests for the EvaluationEngine class.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from portfolio_backtester.backtester_logic.evaluation_engine import EvaluationEngine


class TestEvaluationEngine:
    """Test suite for EvaluationEngine class."""

    def setup_method(self):
        """Set up the test environment."""
        self.global_config = {
            "data_source": {"type": "memory", "data": {}},
            "benchmark": "SPY",
        }
        self.data_source = Mock()
        self.strategy_manager = Mock()
        self.evaluation_engine = EvaluationEngine(
            self.global_config, self.data_source, self.strategy_manager
        )
        self.evaluation_engine.metrics_to_optimize = ["sharpe"]

    def test_initialization(self):
        """Test that the EvaluationEngine is initialized correctly."""
        assert self.evaluation_engine.global_config == self.global_config
        assert self.evaluation_engine.data_source == self.data_source
        assert self.evaluation_engine.strategy_manager == self.strategy_manager

    @patch("portfolio_backtester.backtesting.strategy_backtester.StrategyBacktester")
    @patch("portfolio_backtester.optimization.evaluator.BacktestEvaluator")
    def test_evaluate_walk_forward_fast_single_objective(
        self, mock_evaluator, mock_backtester
    ):
        """Test evaluate_walk_forward_fast with a single objective."""
        mock_evaluator.return_value.evaluate_parameters.return_value.objective_value = 0.5
        trial = Mock()
        trial.params = {}
        result = self.evaluation_engine.evaluate_walk_forward_fast(
            trial, {}, [], None, None, None, ["sharpe"], False, None, Mock()
        )
        assert result == 0.5

    @patch("portfolio_backtester.backtesting.strategy_backtester.StrategyBacktester")
    @patch("portfolio_backtester.optimization.evaluator.BacktestEvaluator")
    def test_evaluate_walk_forward_fast_multi_objective(
        self, mock_evaluator, mock_backtester
    ):
        """Test evaluate_walk_forward_fast with multiple objectives."""
        mock_evaluator.return_value.evaluate_parameters.return_value.objective_value = [
            0.5,
            0.6,
        ]
        trial = Mock()
        trial.params = {}
        result = self.evaluation_engine.evaluate_walk_forward_fast(
            trial, {}, [], None, None, None, ["sharpe", "calmar"], True, None, Mock()
        )
        assert result == (0.5, 0.6)

    @patch.dict("os.environ", {"ENABLE_NUMBA_WALKFORWARD": "1"})
    @patch(
        "portfolio_backtester.backtester_logic.evaluation_engine.EvaluationEngine.evaluate_walk_forward_fast"
    )
    def test_evaluate_fast_use_fast_path(self, mock_evaluate_walk_forward_fast):
        """Test evaluate_fast uses the fast path when enabled."""
        mock_evaluate_walk_forward_fast.return_value = 0.5
        trial = Mock()
        trial.params = {}
        result, _ = self.evaluation_engine.evaluate_fast(
            trial,
            {"strategy": "MomentumSignalStrategy", "strategy_params": {}},
            [],
            pd.DataFrame(),
            pd.DataFrame(columns=["Close"]),
            pd.DataFrame(),
            ["sharpe"],
            False,
        )
        assert result == 0.5
        mock_evaluate_walk_forward_fast.assert_called_once()

    @patch.dict("os.environ", {"DISABLE_NUMBA_WALKFORWARD": "1"})
    @patch("portfolio_backtester.optimization.evaluator.BacktestEvaluator")
    def test_evaluate_fast_use_new_architecture_path(self, mock_evaluator):
        """Test evaluate_fast uses the new architecture path when fast path is disabled."""
        mock_evaluator.return_value.evaluate_parameters.return_value.objective_value = 0.5
        trial = Mock()
        trial.params = {}
        trial.user_attrs = {}
        result, _ = self.evaluation_engine.evaluate_fast(
            trial, {}, [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), ["sharpe"], False
        )
        assert result == 0.5

    def test_evaluate_trial_parameters_returns_none(self):
        """Test evaluate_trial_parameters when returns is None."""
        run_scenario_func = Mock(return_value=None)
        result = self.evaluation_engine.evaluate_trial_parameters(
            {}, {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), run_scenario_func
        )
        assert result == {"sharpe": 0.0}

    def test_evaluate_trial_parameters_returns_empty(self):
        """Test evaluate_trial_parameters when returns is empty."""
        run_scenario_func = Mock(return_value=pd.Series(dtype=float))
        result = self.evaluation_engine.evaluate_trial_parameters(
            {}, {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), run_scenario_func
        )
        assert result == {"sharpe": 0.0}
