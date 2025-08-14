"""
Unit tests for the optimization module.
"""
import pytest
from unittest.mock import Mock, patch

from portfolio_backtester.backtester_logic import optimization


class TestOptimization:
    """Test suite for optimization module."""

    def setup_method(self):
        """Set up the test environment."""
        self.backtester = Mock()
        self.backtester.logger = Mock()
        self.backtester.global_config = {}
        self.backtester.random_state = 1

    @patch("portfolio_backtester.backtester_logic.optimization.GeneticOptimizer")
    def test_get_optimizer_genetic(self, mock_genetic_optimizer):
        """Test that get_optimizer returns a GeneticOptimizer instance."""
        optimizer = optimization.get_optimizer(
            "genetic", {}, self.backtester, {}, None, None, None, None
        )
        assert optimizer == mock_genetic_optimizer.return_value

    @patch("portfolio_backtester.backtester_logic.optimization.OptunaOptimizer")
    def test_get_optimizer_optuna_available(self, mock_optuna_optimizer):
        """Test that get_optimizer returns an OptunaOptimizer instance when available."""
        optimization.OptunaOptimizer = mock_optuna_optimizer
        optimizer = optimization.get_optimizer(
            "optuna", {}, self.backtester, {}, None, None, None, None
        )
        assert optimizer == mock_optuna_optimizer.return_value

    def test_get_optimizer_optuna_not_available(self):
        """Test that get_optimizer raises an ImportError when Optuna is not available."""
        optimization.OptunaOptimizer = None
        with pytest.raises(ImportError):
            optimization.get_optimizer(
                "optuna", {}, self.backtester, {}, None, None, None, None
            )

    def test_get_optimizer_unknown(self):
        """Test that get_optimizer raises a ValueError for an unknown optimizer type."""
        with pytest.raises(ValueError):
            optimization.get_optimizer(
                "unknown", {}, self.backtester, {}, None, None, None, None
            )

    @patch("portfolio_backtester.backtester_logic.optimization.get_optimizer")
    def test_run_optimization_success(self, mock_get_optimizer):
        """Test a successful run of the optimization process."""
        mock_optimizer = Mock()
        mock_optimizer.optimize.return_value = (1, 2)
        mock_get_optimizer.return_value = mock_optimizer

        self.backtester.global_config = {"optimizer_config": {"optimizer_type": "optuna"}}

        result = optimization.run_optimization(self.backtester, {"name": "test"}, None, None, None)
        assert result == (1, 2)
        mock_get_optimizer.assert_called_once()
        mock_optimizer.optimize.assert_called_once()

    @patch("portfolio_backtester.backtester_logic.optimization.get_optimizer")
    def test_run_optimization_unexpected_return(self, mock_get_optimizer):
        """Test that run_optimization raises a ValueError for an unexpected return value."""
        mock_optimizer = Mock()
        mock_optimizer.optimize.return_value = (1,)  # Return a single value
        mock_get_optimizer.return_value = mock_optimizer

        self.backtester.global_config = {"optimizer_config": {"optimizer_type": "optuna"}}

        with pytest.raises(ValueError):
            optimization.run_optimization(self.backtester, {"name": "test"}, None, None, None)
