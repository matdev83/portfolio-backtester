"""
Unit tests for the monte_carlo_analyzer module.
"""
import pytest
from unittest.mock import Mock, patch
import pandas as pd

from portfolio_backtester.reporting.monte_carlo_analyzer import plot_stability_measures


class TestMonteCarloAnalyzer:
    """Test suite for the monte_carlo_analyzer module."""

    def setup_method(self):
        """Set up the test environment."""
        self.backtester = Mock()
        self.backtester.logger = Mock()
        self.backtester.global_config = {}
        self.scenario_name = "test_scenario"
        self.optimization_result = Mock()
        self.optimal_returns = pd.Series([0.1, 0.2, 0.3])

    def test_plot_stability_measures_no_history(self):
        """Test plot_stability_measures with no optimization history."""
        self.optimization_result.optimization_history = []
        plot_stability_measures(
            self.backtester,
            self.scenario_name,
            self.optimization_result,
            self.optimal_returns,
        )
        self.backtester.logger.info.assert_called_with(
            "Skipping trial P&L visualization: insufficient optimization history (need >=2)."
        )

    def test_plot_stability_measures_insufficient_history(self):
        """Test plot_stability_measures with insufficient optimization history."""
        self.optimization_result.optimization_history = [{}]
        plot_stability_measures(
            self.backtester,
            self.scenario_name,
            self.optimization_result,
            self.optimal_returns,
        )
        self.backtester.logger.info.assert_called_with(
            "Skipping trial P&L visualization: insufficient optimization history (need >=2)."
        )



