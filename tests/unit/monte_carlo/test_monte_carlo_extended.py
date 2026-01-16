import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.portfolio_backtester.monte_carlo.monte_carlo import MonteCarloSimulator, run_monte_carlo_simulation

class TestMonteCarloSimulator:
    def test_run_simulation_basic(self):
        # Create some dummy returns with known mean/std
        # Mean ~ 0.01, Std ~ 0.02
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.01, 0.02, 100))
        
        simulator = MonteCarloSimulator(n_simulations=10, n_years=1, initial_capital=100.0)
        results = simulator.run_simulation(returns)
        
        assert isinstance(results, pd.DataFrame)
        # n_years=1, so 12 months + 1 initial row = 13 rows
        assert len(results) == 13 
        assert len(results.columns) == 10 # 10 simulations
        
        # Check initial capital
        assert (results.iloc[0] == 100.0).all()

    def test_run_monte_carlo_simulation_function(self):
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.01, 0.02, 100))
        
        results = run_monte_carlo_simulation(
            returns, n_simulations=50, n_years=2, initial_capital=1000.0
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 2 * 12 + 1 # 25 rows
        assert len(results.columns) == 50
        assert (results.iloc[0] == 1000.0).all()

    @patch('src.portfolio_backtester.monte_carlo.monte_carlo.plot_monte_carlo_results')
    def test_plot_results_delegation(self, mock_plot):
        simulator = MonteCarloSimulator()
        results = pd.DataFrame()
        simulator.plot_results(results, title="Test")
        
        mock_plot.assert_called_once()
        args, kwargs = mock_plot.call_args
        assert args[0] is results
        assert kwargs["title"] == "Test"

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plotting_function_file_creation(self, mock_close, mock_show, mock_savefig):
        # We need to test plot_monte_carlo_results directly, or via simulator
        # Since it's imported in the module, we can import it in test
        from src.portfolio_backtester.monte_carlo.monte_carlo import plot_monte_carlo_results
        
        results = pd.DataFrame(np.random.randn(13, 10) + 100)
        
        with patch('os.makedirs') as mock_makedirs:
            plot_monte_carlo_results(
                results, 
                title="Test Plot", 
                output_dir="test_output", 
                interactive=False
            )
            
            mock_makedirs.assert_called_with("test_output", exist_ok=True)
            mock_savefig.assert_called_once()
            mock_show.assert_not_called()
            mock_close.assert_called()
