"""
Unit tests for the execution module.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch

from portfolio_backtester.backtester_logic import execution


class TestExecution:
    """Test suite for execution module."""

    def setup_method(self):
        """Set up the test environment."""
        self.backtester = Mock()
        self.backtester.logger = Mock()
        self.backtester.args = Mock()
        self.backtester.global_config = {}
        self.backtester.results = {}
        self.backtester.data_source = Mock()
        if hasattr(self.backtester, "_deferred_report_data"):
            del self.backtester._deferred_report_data

    def test_run_backtest_mode_no_study(self):
        """Test run_backtest_mode without a study."""
        scenario_config = {"name": "test", "strategy_params": {}}
        with patch(
            "portfolio_backtester.backtesting.strategy_backtester.StrategyBacktester"
        ) as mock_strategy_backtester:
            mock_strategy_backtester.return_value.backtest_strategy.return_value = Mock(
                returns=pd.Series(),
                trade_stats={},
                trade_history=[],
                performance_stats={},
                charts_data={},
            )
            execution.run_backtest_mode(
                self.backtester, scenario_config, None, None, None
            )
        assert "test" in self.backtester.results
        assert "returns" in self.backtester.results["test"]

    @patch("optuna.load_study")
    def test_run_backtest_mode_with_study(self, mock_load_study):
        """Test run_backtest_mode with a study."""
        mock_load_study.return_value.best_params = {"param1": 1}
        self.backtester.args.study_name = "test_study"
        scenario_config = {"name": "test", "strategy_params": {}}
        with patch(
            "portfolio_backtester.backtesting.strategy_backtester.StrategyBacktester"
        ) as mock_strategy_backtester:
            mock_strategy_backtester.return_value.backtest_strategy.return_value = Mock(
                returns=pd.Series(),
                trade_stats={},
                trade_history=[],
                performance_stats={},
                charts_data={},
            )
            execution.run_backtest_mode(
                self.backtester, scenario_config, None, None, None
            )
        assert "test" in self.backtester.results
        assert "returns" in self.backtester.results["test"]
        assert scenario_config["strategy_params"]["param1"] == 1

    @patch("optuna.load_study", side_effect=KeyError)
    def test_run_backtest_mode_study_not_found(self, mock_load_study):
        """Test run_backtest_mode when the study is not found."""
        self.backtester.args.study_name = "test_study"
        scenario_config = {"name": "test", "strategy_params": {}}
        with patch(
            "portfolio_backtester.backtesting.strategy_backtester.StrategyBacktester"
        ) as mock_strategy_backtester:
            mock_strategy_backtester.return_value.backtest_strategy.return_value = Mock(
                returns=pd.Series(),
                trade_stats={},
                trade_history=[],
                performance_stats={},
                charts_data={},
            )
            execution.run_backtest_mode(
                self.backtester, scenario_config, None, None, None
            )
        assert "test" in self.backtester.results
        assert "returns" in self.backtester.results["test"]

    def test_run_optimize_mode_interrupted(self):
        """Test run_optimize_mode when optimization is interrupted."""
        self.backtester.run_optimization.return_value = (None, 0, None)
        with patch("portfolio_backtester.backtester_logic.execution.CENTRAL_INTERRUPTED_FLAG", True):
            execution.run_optimize_mode(self.backtester, {"name": "test"}, None, None, None)
        assert "test (Optimization Interrupted)" in self.backtester.results

    def test_run_optimize_mode_fails(self):
        """Test run_optimize_mode when optimization fails."""
        self.backtester.run_optimization.return_value = (None, 0, None)
        execution.run_optimize_mode(self.backtester, {"name": "test"}, None, None, None)
        assert "test" not in self.backtester.results

    def test_run_optimize_mode_succeeds(self):
        """Test run_optimize_mode when optimization succeeds."""
        self.backtester.run_optimization.return_value = ({}, 0, None)
        with patch("portfolio_backtester.backtester_logic.execution.handle_constraints") as mock_handle_constraints:
            mock_handle_constraints.return_value = ("test", pd.Series(), {}, "pass", "", [], {})
            execution.run_optimize_mode(self.backtester, {"name": "test"}, None, None, None)
        assert "test" in self.backtester.results

    def test_run_optimize_mode_reporting_enabled(self):
        """Test run_optimize_mode with reporting enabled."""
        self.backtester.run_optimization.return_value = ({}, 0, None)
        self.backtester.global_config = {"advanced_reporting_config": {"enable_optimization_reports": True}}
        with patch("portfolio_backtester.backtester_logic.execution.handle_constraints") as mock_handle_constraints, \
             patch("portfolio_backtester.backtester_logic.reporting_logic.generate_optimization_report") as mock_generate_report:
            mock_handle_constraints.return_value = ("test", pd.Series(), {}, "pass", "", [], {})
            execution.run_optimize_mode(self.backtester, {"name": "test"}, None, None, None)
        assert "test" in self.backtester.results
        assert hasattr(self.backtester, "_deferred_report_data")

    def test_run_optimize_mode_reporting_disabled(self):
        """Test run_optimize_mode with reporting disabled."""
        self.backtester.run_optimization.return_value = ({}, 0, None)
        self.backtester.global_config = {"advanced_reporting_config": {"enable_optimization_reports": False}}
        with patch("portfolio_backtester.backtester_logic.execution.handle_constraints") as mock_handle_constraints:
            mock_handle_constraints.return_value = ("test", pd.Series(), {}, "pass", "", [], {})
            execution.run_optimize_mode(self.backtester, {"name": "test"}, None, None, None)
        assert "test" in self.backtester.results
        assert not hasattr(self.backtester, "_deferred_report_data")

    def test_generate_deferred_report(self):
        """Test generate_deferred_report."""
        self.backtester._deferred_report_data = {"test": "test"}
        with patch("portfolio_backtester.backtester_logic.reporting_logic.generate_optimization_report") as mock_generate_report:
            execution.generate_deferred_report(self.backtester)
        mock_generate_report.assert_called_once_with(self.backtester, test="test")
        assert not hasattr(self.backtester, "_deferred_report_data")
