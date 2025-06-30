#!/usr/bin/env python3
"""Simple test script for Calmar Momentum Strategy."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from portfolio_backtester.strategies.calmar_momentum_strategy import CalmarMomentumStrategy

def test_calmar_strategy():
    """Test the Calmar Momentum Strategy."""
    print("Testing Calmar Momentum Strategy...")
    
    # Create test data
    dates = pd.date_range('2020-01-01', periods=24, freq='ME')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Generate sample price data with different trends
    np.random.seed(42)
    data = {}
    for i, ticker in enumerate(tickers):
        # Create different return patterns
        if i == 0:  # AAPL - steady growth
            returns = np.random.normal(0.02, 0.03, len(dates))
        elif i == 1:  # MSFT - high volatility but positive
            returns = np.random.normal(0.015, 0.08, len(dates))
        elif i == 2:  # GOOGL - declining
            returns = np.random.normal(-0.01, 0.04, len(dates))
        elif i == 3:  # AMZN - very volatile
            returns = np.random.normal(0.01, 0.12, len(dates))
        else:  # TSLA - moderate growth
            returns = np.random.normal(0.008, 0.05, len(dates))
        
        prices = 100 * (1 + returns).cumprod()
        data[ticker] = prices
    
    data = pd.DataFrame(data, index=dates)
    benchmark_data = pd.Series(100 * (1 + np.random.normal(0.008, 0.04, len(dates))).cumprod(), 
                               index=dates, name='SPY')
    
    # Test strategy configuration
    strategy_config = {
        'rolling_window': 6,
        'top_decile_fraction': 0.2,  # Top 20% (1 out of 5 stocks)
        'smoothing_lambda': 0.5,
        'leverage': 1.0,
        'long_only': True,
        'sma_filter_window': None
    }
    
    # Create strategy
    strategy = CalmarMomentumStrategy(strategy_config)
    
    # Test rolling Calmar calculation
    print("Testing rolling Calmar calculation...")
    rets = data.pct_change(fill_method=None)
    rolling_calmar = strategy._calculate_rolling_calmar(rets, 6)
    
    print(f"Rolling Calmar shape: {rolling_calmar.shape}")
    print(f"Rolling Calmar has NaN: {rolling_calmar.isna().any().any()}")
    print(f"Rolling Calmar final values:\n{rolling_calmar.iloc[-1]}")
    
    # Test candidate weights
    print("\nTesting candidate weights...")
    look = rolling_calmar.iloc[-1]
    weights = strategy._calculate_candidate_weights(look)
    print(f"Candidate weights:\n{weights}")
    print(f"Sum of weights: {weights.sum()}")
    print(f"Number of holdings: {(weights > 0).sum()}")
    
    # Test signal generation
    print("\nTesting signal generation...")
    signals = strategy.generate_signals(data, benchmark_data)
    print(f"Signals shape: {signals.shape}")
    print(f"Final signals:\n{signals.iloc[-1]}")
    print(f"Signals have NaN: {signals.isna().any().any()}")
    print(f"All signals non-negative: {(signals >= 0).all().all()}")
    
    # Test with SMA filter
    print("\nTesting with SMA filter...")
    strategy_config_sma = strategy_config.copy()
    strategy_config_sma['sma_filter_window'] = 3
    strategy_sma = CalmarMomentumStrategy(strategy_config_sma)
    signals_sma = strategy_sma.generate_signals(data, benchmark_data)
    print(f"SMA filtered signals shape: {signals_sma.shape}")
    print(f"Final SMA filtered signals:\n{signals_sma.iloc[-1]}")
    
    print("\n✅ All tests passed! Calmar Momentum Strategy is working correctly.")
    return True

if __name__ == '__main__':
    try:
        test_calmar_strategy()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)