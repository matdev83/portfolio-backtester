"""
Unit tests for the evaluation_strategy module.
"""
import pytest
from unittest.mock import Mock, patch
import pandas as pd

from portfolio_backtester.interfaces.evaluation_strategy import (
    StandardEvaluationStrategy,
    MultiObjectiveEvaluationStrategy,
    EvaluationStrategyFactory,
    IEvaluationStrategy,
)


class TestStandardEvaluationStrategy:
    """Test suite for StandardEvaluationStrategy."""

    def setup_method(self):
        """Set up the test environment."""
        self.strategy = StandardEvaluationStrategy()

    @patch("portfolio_backtester.interfaces.evaluation_strategy.calculate_metrics")
    def test_evaluate_performance_series(self, mock_calculate_metrics):
        """Test evaluate_performance with a pandas Series."""
        mock_calculate_metrics.return_value = {"sharpe_ratio": 2.0}
        returns = pd.Series([0.1, 0.2, 0.3])
        result = self.strategy.evaluate_performance(returns)
        assert result == {"sharpe_ratio": 2.0}
        mock_calculate_metrics.assert_called_once()

    @patch("portfolio_backtester.interfaces.evaluation_strategy.calculate_metrics")
    def test_evaluate_performance_dataframe(self, mock_calculate_metrics):
        """Test evaluate_performance with a pandas DataFrame."""
        mock_calculate_metrics.return_value = {"sharpe_ratio": 2.0}
        returns = pd.DataFrame({"returns": [0.1, 0.2, 0.3]})
        result = self.strategy.evaluate_performance(returns)
        assert result == {"sharpe_ratio": 2.0}
        mock_calculate_metrics.assert_called_once()

    def test_evaluate_performance_invalid_data(self):
        """Test evaluate_performance with invalid data."""
        with pytest.raises(ValueError):
            self.strategy.evaluate_performance("not a series or dataframe")

    def test_evaluate_parameters(self):
        """Test evaluate_parameters."""
        parameters = {"param1": 1}
        scenario_config = {"optimization": {"metrics": ["sharpe_ratio"]}}
        result = self.strategy.evaluate_parameters(parameters, scenario_config, None)
        assert result == {"sharpe_ratio": 0.0}


class TestMultiObjectiveEvaluationStrategy:
    """Test suite for MultiObjectiveEvaluationStrategy."""

    def setup_method(self):
        """Set up the test environment."""
        self.strategy = MultiObjectiveEvaluationStrategy(objectives=["sharpe_ratio", "calmar_ratio"])

    @patch("portfolio_backtester.interfaces.evaluation_strategy.StandardEvaluationStrategy.evaluate_performance")
    def test_evaluate_performance(self, mock_evaluate_performance):
        """Test that evaluate_performance delegates to the standard strategy."""
        mock_evaluate_performance.return_value = {"sharpe_ratio": 2.0, "calmar_ratio": 3.0}
        returns = pd.Series([0.1, 0.2, 0.3])
        result = self.strategy.evaluate_performance(returns)
        assert result == {"sharpe_ratio": 2.0, "calmar_ratio": 3.0}
        mock_evaluate_performance.assert_called_once()

    @patch("portfolio_backtester.interfaces.evaluation_strategy.StandardEvaluationStrategy.evaluate_parameters")
    def test_evaluate_parameters(self, mock_evaluate_parameters):
        """Test that evaluate_parameters delegates to the standard strategy."""
        mock_evaluate_parameters.return_value = {"sharpe_ratio": 0.0, "calmar_ratio": 0.0}
        parameters = {"param1": 1}
        scenario_config = {"optimization": {"metrics": ["sharpe_ratio", "calmar_ratio"]}}
        result = self.strategy.evaluate_parameters(parameters, scenario_config, None)
        assert result == {"sharpe_ratio": 0.0, "calmar_ratio": 0.0}
        mock_evaluate_parameters.assert_called_once()


class TestEvaluationStrategyFactory:
    """Test suite for EvaluationStrategyFactory."""

    def test_create_strategy_single_objective(self):
        """Test that the factory creates a StandardEvaluationStrategy for a single objective."""
        scenario_config = {"optimization": {"metrics": ["sharpe_ratio"]}}
        strategy = EvaluationStrategyFactory.create_strategy(scenario_config)
        assert isinstance(strategy, StandardEvaluationStrategy)

    def test_create_strategy_multi_objective(self):
        """Test that the factory creates a MultiObjectiveEvaluationStrategy for multiple objectives."""
        scenario_config = {"optimization": {"metrics": ["sharpe_ratio", "calmar_ratio"]}}
        strategy = EvaluationStrategyFactory.create_strategy(scenario_config)
        assert isinstance(strategy, MultiObjectiveEvaluationStrategy)

    def test_create_strategy_default_objective(self):
        """Test that the factory creates a StandardEvaluationStrategy by default."""
        scenario_config = {}
        strategy = EvaluationStrategyFactory.create_strategy(scenario_config)
        assert isinstance(strategy, StandardEvaluationStrategy)
