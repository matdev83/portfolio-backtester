"""
Unit tests for the parameter_analysis module.
"""
import pytest
from unittest.mock import Mock, patch
import pandas as pd

from portfolio_backtester.reporting.parameter_analysis import _plot_parameter_impact_analysis


class TestParameterAnalysis:
    """Test suite for the parameter_analysis module."""

    def setup_method(self):
        """Set up the test environment."""
        self.backtester = Mock()
        self.backtester.logger = Mock()
        self.scenario_name = "test_scenario"
        self.timestamp = "20230101"

    def test_plot_parameter_impact_analysis_no_study(self):
        """Test _plot_parameter_impact_analysis with no study object."""
        best_trial_obj = Mock()
        del best_trial_obj.study
        _plot_parameter_impact_analysis(self.backtester, self.scenario_name, best_trial_obj, self.timestamp)
        self.backtester.logger.warning.assert_called_with(
            "No study object found. Cannot create parameter impact analysis."
        )

    def test_plot_parameter_impact_analysis_insufficient_trials(self):
        """Test _plot_parameter_impact_analysis with insufficient trials."""
        mock_study = Mock()
        mock_trial = Mock()
        mock_trial.state.name = "COMPLETE"
        mock_study.trials = [mock_trial] * 5  # Less than 10
        best_trial_obj = Mock()
        best_trial_obj.study = mock_study
        _plot_parameter_impact_analysis(self.backtester, self.scenario_name, best_trial_obj, self.timestamp)
        self.backtester.logger.warning.assert_called_with(
            "Need ≥10 completed trials for meaningful parameter analysis."
        )



