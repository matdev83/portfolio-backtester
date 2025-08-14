"""
Unit tests for the constraint_logic module.
"""
from unittest.mock import Mock, patch
import pandas as pd

from portfolio_backtester.backtester_logic.constraint_logic import handle_constraints


class TestHandleConstraints:
    """Test suite for the handle_constraints function."""

    def setup_method(self):
        """Set up the test environment."""
        self.backtester = Mock()
        self.backtester.global_config = {"benchmark": "SPY"}
        self.backtester.rets_full = pd.DataFrame({"SPY": [0.1, 0.2, 0.3]})
        self.scenario_config = {"name": "test_scenario"}
        self.optimal_params = {"param1": 1}
        self.full_rets = pd.Series([0.1, 0.2, 0.3])
        self.monthly_data = None
        self.daily_data = None
        self.rets_full = None

    def test_no_constraints(self):
        """Test handle_constraints with no constraints defined."""
        result = handle_constraints(
            self.backtester,
            self.scenario_config,
            self.optimal_params,
            self.full_rets,
            self.monthly_data,
            self.daily_data,
            self.rets_full,
        )
        assert result[3] == "OK"
        assert result[2] == self.optimal_params

    @patch("portfolio_backtester.backtester_logic.constraint_logic.ConstraintHandler")
    def test_constraints_satisfied(self, mock_constraint_handler):
        """Test handle_constraints when constraints are satisfied."""
        mock_handler_instance = mock_constraint_handler.return_value
        adjusted_params = {"param1": 2}
        adjusted_rets = pd.Series([0.4, 0.5, 0.6])
        mock_handler_instance.find_constraint_satisfying_params.return_value = (
            adjusted_params,
            adjusted_rets,
            True,
        )
        scenario_config = {
            "name": "test_scenario",
            "optimization_constraints": ["some_constraint"],
        }
        result = handle_constraints(
            self.backtester,
            scenario_config,
            self.optimal_params,
            self.full_rets,
            self.monthly_data,
            self.daily_data,
            self.rets_full,
        )
        assert result[3] == "ADJUSTED"
        assert result[2] == adjusted_params
        assert result[1].equals(adjusted_rets)

    @patch("portfolio_backtester.backtester_logic.constraint_logic.ConstraintHandler")
    def test_constraints_violated(self, mock_constraint_handler):
        """Test handle_constraints when constraints are violated."""
        mock_handler_instance = mock_constraint_handler.return_value
        mock_handler_instance.find_constraint_satisfying_params.return_value = (
            self.optimal_params,
            self.full_rets,
            False,
        )
        scenario_config = {
            "name": "test_scenario",
            "optimization_constraints": ["some_constraint"],
        }
        result = handle_constraints(
            self.backtester,
            scenario_config,
            self.optimal_params,
            self.full_rets,
            self.monthly_data,
            self.daily_data,
            self.rets_full,
        )
        assert result[3] == "VIOLATED"

    @patch("portfolio_backtester.backtester_logic.constraint_logic.ConstraintHandler")
    def test_constraint_handling_error(self, mock_constraint_handler):
        """Test handle_constraints when an error occurs."""
        mock_handler_instance = mock_constraint_handler.return_value
        mock_handler_instance.find_constraint_satisfying_params.side_effect = Exception("Test error")
        scenario_config = {
            "name": "test_scenario",
            "optimization_constraints": ["some_constraint"],
        }
        result = handle_constraints(
            self.backtester,
            scenario_config,
            self.optimal_params,
            self.full_rets,
            self.monthly_data,
            self.daily_data,
            self.rets_full,
        )
        assert result[3] == "ERROR"
        assert result[4] == "Test error"
