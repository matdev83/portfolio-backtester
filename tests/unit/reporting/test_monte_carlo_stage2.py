"""
Unit tests for the monte_carlo_stage2 module.
"""
import pytest
from unittest.mock import Mock, patch
import pandas as pd

from portfolio_backtester.reporting.monte_carlo_stage2 import _plot_monte_carlo_robustness_analysis


class TestMonteCarloStage2:
    """Test suite for the monte_carlo_stage2 module."""

    def setup_method(self):
        """Set up the test environment."""
        self.backtester = Mock()
        self.backtester.logger = Mock()
        self.backtester.global_config = {}
        self.scenario_name = "test_scenario"
        self.scenario_config = {}
        self.optimal_params = {}
        self.monthly_data = None
        self.daily_data = None
        self.rets_full = None

    def test_plot_monte_carlo_robustness_analysis_synthetic_disabled(self):
        """Test with synthetic data disabled."""
        self.backtester.global_config = {"monte_carlo_config": {"enable_synthetic_data": False}}
        _plot_monte_carlo_robustness_analysis(
            self.backtester,
            self.scenario_name,
            self.scenario_config,
            self.optimal_params,
            self.monthly_data,
            self.daily_data,
            self.rets_full,
        )
        self.backtester.logger.warning.assert_called_with(
            "Stage 2 MC: Synthetic data generation is disabled. Cannot create robustness analysis."
        )

    def test_plot_monte_carlo_robustness_analysis_stage2_disabled(self):
        """Test with Stage 2 stress testing disabled."""
        self.backtester.global_config = {
            "monte_carlo_config": {
                "enable_synthetic_data": True,
                "enable_stage2_stress_testing": False,
            }
        }
        _plot_monte_carlo_robustness_analysis(
            self.backtester,
            self.scenario_name,
            self.scenario_config,
            self.optimal_params,
            self.monthly_data,
            self.daily_data,
            self.rets_full,
        )
        self.backtester.logger.info.assert_called_with(
            "Stage 2 MC: Stage 2 stress testing is disabled for faster optimization. Skipping robustness analysis."
        )


