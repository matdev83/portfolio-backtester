"""
Unit tests for the monte_carlo module.
"""
import pytest
import pandas as pd
from unittest.mock import patch
from portfolio_backtester.monte_carlo.monte_carlo import MonteCarloSimulator, run_monte_carlo_simulation


class TestMonteCarloSimulator:
    """Test suite for MonteCarloSimulator."""

    def setup_method(self):
        """Set up the test environment."""
        self.simulator = MonteCarloSimulator()
        self.returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01])

    def test_run_simulation(self):
        """Test the simulation run."""
        results = self.simulator.run_simulation(self.returns)
        assert isinstance(results, pd.DataFrame)
        assert results.shape == (121, 1000)

    @patch("portfolio_backtester.monte_carlo.monte_carlo.plot_monte_carlo_results")
    def test_plot_results(self, mock_plot):
        """Test the plotting of results."""
        simulation_results = self.simulator.run_simulation(self.returns)
        self.simulator.plot_results(simulation_results)
        mock_plot.assert_called_once()


def test_run_monte_carlo_simulation():
    """Test the run_monte_carlo_simulation function."""
    returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01])
    results = run_monte_carlo_simulation(returns)
    assert isinstance(results, pd.DataFrame)
    assert results.shape == (121, 1000)
