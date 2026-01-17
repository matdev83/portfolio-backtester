"""
Unit tests for the monte_carlo module.
"""
import pytest
import pandas as pd
import numpy as np
import os
import shutil
from unittest.mock import patch, MagicMock
from portfolio_backtester.monte_carlo.monte_carlo import (
    MonteCarloSimulator,
    run_monte_carlo_simulation,
    plot_monte_carlo_results
)


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


class TestMonteCarloFunctions:
    """Detailed tests for standalone Monte Carlo functions."""

    def test_run_monte_carlo_simulation_defaults(self):
        """Test default parameters."""
        returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01])
        results = run_monte_carlo_simulation(returns)
        # Default n_years=10 -> 120 months + 1 initial row = 121 rows
        # Default n_simulations=1000 -> 1000 cols
        assert results.shape == (121, 1000)
        # Check initial capital (first row should be all 1.0)
        assert np.allclose(results.iloc[0], 1.0)

    def test_run_monte_carlo_simulation_custom_params(self):
        """Test custom parameters."""
        returns = pd.Series([0.01, 0.02])
        n_sims = 50
        n_years = 2
        initial_cap = 10000.0
        
        results = run_monte_carlo_simulation(
            returns, 
            n_simulations=n_sims, 
            n_years=n_years, 
            initial_capital=initial_cap
        )
        
        # 2 years * 12 months = 24 steps + 1 initial = 25 rows
        assert results.shape == (25, 50)
        assert np.allclose(results.iloc[0], 10000.0)

    def test_run_monte_carlo_constant_returns(self):
        """Test with constant returns (std dev = 0)."""
        # If returns are constant, std_dev is 0. 
        # Simulation should produce deterministic paths (straight lines in log space).
        returns = pd.Series([0.01, 0.01, 0.01])
        results = run_monte_carlo_simulation(returns, n_simulations=10, n_years=1)
        
        # All paths should be identical
        first_path = results.iloc[:, 0]
        for col in results.columns:
            pd.testing.assert_series_equal(results[col], first_path, check_names=False)
            
        # Check values: should be compounding 1% monthly
        # (approximately, since it uses mean of input series which is 0.01)
        expected_final = 1.0 * (1.01 ** 12)
        assert np.isclose(results.iloc[-1, 0], expected_final)

    def test_plot_monte_carlo_results_file_creation(self, tmp_path):
        """Test that plotting creates a file."""
        df = pd.DataFrame(np.random.randn(10, 5) + 1.0).cumprod()
        output_dir = tmp_path / "mc_plots"
        
        # Run plotting
        plot_monte_carlo_results(
            df,
            title="Test Plot",
            scenario_name="test_scenario",
            params={"a": 1},
            output_dir=str(output_dir),
            interactive=False
        )
        
        # Check if directory and file exist
        assert output_dir.exists()
        files = list(output_dir.glob("*.png"))
        assert len(files) == 1
        assert "test_scenario" in files[0].name

    @patch("matplotlib.pyplot.show")
    def test_plot_monte_carlo_interactive(self, mock_show, tmp_path):
        """Test interactive plotting flag."""
        df = pd.DataFrame(np.random.randn(10, 5))
        plot_monte_carlo_results(
            df,
            output_dir=str(tmp_path),
            interactive=True
        )
        mock_show.assert_called()

