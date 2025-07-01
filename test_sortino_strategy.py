#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from portfolio_backtester.strategies.sortino_momentum_strategy import SortinoMomentumStrategy

def test_sortino_strategy():
    # Create sample data
    dates = pd.date_range(start='2020-01-01', periods=12, freq='ME')
    data = pd.DataFrame({
        'AAPL': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210],
        'MSFT': [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 1],
        'GOOGL': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
    }, index=dates)
    
    benchmark_data = pd.Series([100] * 12, index=dates)
    
    # Test strategy configuration
    strategy_config = {
        'rolling_window': 3,
        'top_decile_fraction': 0.5,
        'smoothing_lambda': 0.5,
        'leverage': 1.0,
        'long_only': True,
        'target_return': 0.0
    }
    
    strategy = SortinoMomentumStrategy(strategy_config)
    # Mock features for testing
    mock_features = {
        f"sortino_{strategy_config['rolling_window']}m": pd.DataFrame(np.random.rand(*data.shape), index=data.index, columns=data.columns),
        f"benchmark_sma_{strategy_config['rolling_window']}m": pd.Series(True, index=data.index) # Mock all True for risk_on
    }
    try:
        weights = strategy.generate_signals(data, mock_features, benchmark_data)

        assert not weights.empty, "Generated weights DataFrame should not be empty"
        assert weights.shape[0] > 0, "Generated weights DataFrame should have rows"
        assert weights.shape[1] > 0, "Generated weights DataFrame should have columns"

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise  # Re-raise the exception to fail the test

    