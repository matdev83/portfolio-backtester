#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from portfolio_backtester.strategies.sortino_momentum_strategy import SortinoMomentumStrategy

def test_sortino_strategy():
    print("Testing Sortino Momentum Strategy...")
    
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
    
    try:
        strategy = SortinoMomentumStrategy(strategy_config)
        weights = strategy.generate_signals(data, benchmark_data)
        
        print(f"Strategy created successfully!")
        print(f"Weights shape: {weights.shape}")
        print(f"Final weights:\n{weights.iloc[-1]}")
        print(f"Sum of final weights: {weights.iloc[-1].sum():.4f}")
        
        # Test rolling Sortino calculation
        rets = data.pct_change(fill_method=None)
        rolling_sortino = strategy._calculate_rolling_sortino(rets, 3, 0.0)
        print(f"\nFinal Sortino ratios:\n{rolling_sortino.iloc[-1]}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sortino_strategy()
    if success:
        print("\n✅ Sortino Momentum Strategy test passed!")
    else:
        print("\n❌ Sortino Momentum Strategy test failed!")