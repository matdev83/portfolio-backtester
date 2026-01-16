import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import MagicMock, patch

from portfolio_backtester.monte_carlo.monte_carlo import MonteCarloSimulator, run_monte_carlo_simulation

@pytest.fixture
def strategy_returns():
    # 1 year of monthly returns
    return pd.Series(np.random.normal(0.01, 0.05, 12))

def test_monte_carlo_simulator_init():
    sim = MonteCarloSimulator(n_simulations=500, n_years=5, initial_capital=100.0)
    assert sim.n_simulations == 500
    assert sim.n_years == 5
    assert sim.initial_capital == 100.0

def test_run_simulation_output_shape(strategy_returns):
    sim = MonteCarloSimulator(n_simulations=10, n_years=1, initial_capital=1.0)
    results = sim.run_simulation(strategy_returns)
    
    # Output should be (n_months + 1) rows x n_simulations cols
    # +1 for initial capital row
    assert results.shape == (12 + 1, 10)
    assert results.iloc[0].unique()[0] == 1.0

def test_run_monte_carlo_simulation_logic(strategy_returns):
    # Test the standalone function
    results = run_monte_carlo_simulation(
        strategy_returns, n_simulations=50, n_years=2, initial_capital=1000.0
    )
    
    assert results.shape == (24 + 1, 50)
    # Check that it runs and produces numbers
    assert not results.isna().any().any()
    # Check start value
    assert (results.iloc[0] == 1000.0).all()

def test_plot_results_saving(strategy_returns, tmp_path):
    sim = MonteCarloSimulator(n_simulations=10, n_years=1)
    results = sim.run_simulation(strategy_returns)
    
    output_dir = tmp_path / "charts"
    
    # Mock plt to avoid display issues
    with patch("matplotlib.pyplot.savefig") as mock_save, \
         patch("matplotlib.pyplot.show"), \
         patch("matplotlib.pyplot.close"):
        
        sim.plot_results(
            results, 
            output_dir=str(output_dir), 
            interactive=False,
            params={"a": 1}
        )
        
        # Check if savefig was called with correct path structure
        assert mock_save.called
        args, _ = mock_save.call_args
        filepath = args[0]
        assert str(output_dir) in filepath
        assert ".png" in filepath

def test_plot_results_param_serialization_failure():
    # If params fail serialization, should fallback safely
    sim = MonteCarloSimulator()
    results = pd.DataFrame(np.random.randn(10, 5))
    
    # Object that fails json dump
    class Unserializable:
        pass
        
    bad_params = {"obj": Unserializable()}
    
    with patch("matplotlib.pyplot.savefig") as mock_save, \
         patch("matplotlib.pyplot.close"):
         
        sim.plot_results(results, params=bad_params, output_dir=".")
        
        assert mock_save.called
        args, _ = mock_save.call_args
        filename = args[0]
        # Should contain fallback "params" string or handle gracefully
        # Implementation says it defaults to "params" on exception
        assert "params" in filename
