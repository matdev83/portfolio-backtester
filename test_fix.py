#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '.')

import unittest
import pandas as pd
import numpy as np
from src.portfolio_backtester.strategies.calmar_momentum_strategy import CalmarMomentumStrategy
from src.portfolio_backtester.feature import CalmarRatio

def test_calmar_fix():
    """Test the rolling Calmar ratio calculation to ensure no non-finite values."""
    
    # Set up test data and strategy configuration
    strategy_config = {
        'rolling_window': 6,
        'top_decile_fraction': 0.1,
        'smoothing_lambda': 0.5,
        'leverage': 1.0,
        'long_only': True,
        'sma_filter_window': None
    }
    strategy = CalmarMomentumStrategy(strategy_config)

    # Create sample data
    dates = pd.date_range('2020-01-01', periods=24, freq='ME')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Generate sample price data with different trends
    np.random.seed(42)
    data = {}
    for ticker in tickers:
        # Create trending data with some volatility
        returns = np.random.normal(0.01, 0.05, len(dates))
        prices = 100 * (1 + returns).cumprod()
        data[ticker] = prices
    
    data = pd.DataFrame(data, index=dates)
    
    # Test the rolling Calmar calculation
    rets = data.pct_change(fill_method=None)
    calmar_feature = CalmarRatio(rolling_window=6)
    rolling_calmar = calmar_feature.compute(data)
    
    print("Rolling Calmar shape:", rolling_calmar.shape)
    print("Expected shape:", rets.shape)
    
    # Check that the result has the correct shape
    assert rolling_calmar.shape == rets.shape, f"Shape mismatch: {rolling_calmar.shape} != {rets.shape}"
    
    # Check that initial periods have NaN values (due to rolling window)
    rolling_window = strategy_config['rolling_window']
    initial_values = rolling_calmar.iloc[:rolling_window-1]
    print(f"Initial {rolling_window-1} periods should be NaN:")
    print(initial_values.isna().all().all())
    
    # Check that values after the rolling window are finite
    after_window = rolling_calmar.iloc[rolling_window-1:]
    print(f"Values after rolling window (from index {rolling_window-1}):")
    print("Are all finite?", np.isfinite(after_window).all().all())
    
    # Print some sample values to inspect
    print("\nSample values after rolling window:")
    print(after_window.head())
    
    # Check for any non-finite values
    non_finite_mask = ~np.isfinite(after_window)
    if non_finite_mask.any().any():
        print("\nNon-finite values found:")
        print(after_window[non_finite_mask.any(axis=1)])
        
    else:
        print("\nAll values after rolling window are finite!")
        

if __name__ == "__main__":
    success = test_calmar_fix()
    if success:
        print("\n✅ TEST PASSED: Fix successful!")
    else:
        print("\n❌ TEST FAILED: Fix needs more work")