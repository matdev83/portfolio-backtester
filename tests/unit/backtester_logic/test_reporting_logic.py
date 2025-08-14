"""
Unit tests for the reporting_logic module.
"""
import pytest
from unittest.mock import Mock, patch
import pandas as pd
import sys

from portfolio_backtester.backtester_logic.reporting_logic import (
    _get_benchmark_returns,
    _build_optimization_metadata,
    _extract_trials_data,
    _build_additional_info,
    generate_optimization_report,
)


class TestReportingLogic:
    """Test suite for reporting_logic module."""

    def setup_method(self):
        """Set up the test environment."""
        self.backtester = Mock()
        self.backtester.global_config = {"benchmark": "SPY"}
        self.full_rets = pd.Series([0.1, 0.2, 0.3], index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))
        self.scenario_config = {"name": "test_scenario"}
        self.optimal_params = {"param1": 1}

    def test_get_benchmark_returns_multiindex(self):
        """Test _get_benchmark_returns with MultiIndex columns."""
        self.backtester.daily_data_ohlc = pd.DataFrame({
            ('SPY', 'Close'): [100, 101, 102],
            ('AAPL', 'Close'): [150, 151, 152]
        }, index=self.full_rets.index)
        self.backtester.daily_data_ohlc.columns = pd.MultiIndex.from_tuples(
            self.backtester.daily_data_ohlc.columns, names=['Ticker', 'Field']
        )
        returns = _get_benchmark_returns(self.backtester, self.full_rets)
        assert isinstance(returns, pd.Series)
        assert not returns.empty

    def test_get_benchmark_returns_singleindex(self):
        """Test _get_benchmark_returns with a single index."""
        self.backtester.daily_data_ohlc = pd.DataFrame({
            'SPY': [100, 101, 102],
            'AAPL': [150, 151, 152]
        }, index=self.full_rets.index)
        returns = _get_benchmark_returns(self.backtester, self.full_rets)
        assert isinstance(returns, pd.Series)
        assert not returns.empty

    def test_get_benchmark_returns_no_data(self):
        """Test _get_benchmark_returns with no daily data."""
        self.backtester.daily_data_ohlc = None
        returns = _get_benchmark_returns(self.backtester, self.full_rets)
        assert returns.equals(pd.Series(0.0, index=self.full_rets.index))

    def test_build_optimization_metadata(self):
        """Test _build_optimization_metadata."""
        metadata = _build_optimization_metadata(self.backtester, 10, True, True)
        assert metadata["num_trials"] == 10
        assert metadata["defer_expensive_plots"] is True
        assert metadata["defer_parameter_analysis"] is True

    def test_extract_trials_data(self):
        """Test _extract_trials_data."""
        mock_optuna = Mock()
        mock_optuna.importance = Mock()
        mock_optuna.importance.get_param_importances = Mock(return_value={"param": 0.9})

        with patch.dict("sys.modules", {"optuna": mock_optuna, "optuna.importance": mock_optuna.importance}):
            mock_study = Mock()
            mock_trial = Mock()
            mock_trial.number = 1
            mock_trial.value = 0.5
            mock_trial.params = {"param": "value"}
            mock_trial.state.name = "COMPLETE"
            mock_trial.user_attrs = {}
            mock_study.trials = [mock_trial] * 11
            mock_best_trial = Mock()
            mock_best_trial.study = mock_study
            mock_best_trial.number = 1

            data = _extract_trials_data(mock_best_trial)
            assert "trials_data" in data
            assert "parameter_importance" in data
            assert len(data["trials_data"]) == 11

    def test_build_additional_info(self):
        """Test _build_additional_info."""
        result_data = {
            "constraint_status": "OK",
            "constraint_message": "All good",
            "constraint_violations": [],
            "constraints_config": {},
        }
        info = _build_additional_info(self.backtester, result_data, 10, None, True, True)
        assert info["num_trials"] == 10
        assert info["constraint_info"]["status"] == "OK"

    @patch("portfolio_backtester.backtester_logic.reporting_logic.create_optimization_report")
    @patch("portfolio_backtester.backtester_logic.reporting_logic.calculate_metrics")
    def test_generate_optimization_report(self, mock_calculate_metrics, mock_create_report):
        """Test the main generate_optimization_report function."""
        mock_calculate_metrics.return_value = {}
        mock_create_report.return_value = "report_path"
        self.backtester.results = {"test_scenario": {}}
        self.backtester.daily_data_ohlc = pd.DataFrame({
            'SPY': [100, 101, 102]
        }, index=self.full_rets.index)

        generate_optimization_report(
            self.backtester,
            self.scenario_config,
            self.optimal_params,
            self.full_rets,
            None,
            10,
        )
        mock_create_report.assert_called_once()
