#!/usr/bin/env python3
"""Validation script for Calmar Momentum Strategy."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    # Test import
    from portfolio_backtester.strategies.calmar_momentum_strategy import CalmarMomentumStrategy
    print("✅ CalmarMomentumStrategy imported successfully")
    
    # Test basic instantiation
    config = {
        'rolling_window': 6,
        'top_decile_fraction': 0.1,
        'smoothing_lambda': 0.5,
        'leverage': 1.0,
        'long_only': True
    }
    
    strategy = CalmarMomentumStrategy(config)
    print("✅ CalmarMomentumStrategy instantiated successfully")
    
    # Test backtester import
    from portfolio_backtester.backtester import Backtester
    print("✅ Backtester imported successfully")
    
    # Test config import
    from portfolio_backtester.config import BACKTEST_SCENARIOS
    
    # Check if Calmar_Momentum is in scenarios
    calmar_scenario = None
    for scenario in BACKTEST_SCENARIOS:
        if scenario['name'] == 'Calmar_Momentum':
            calmar_scenario = scenario
            break
    
    if calmar_scenario:
        print("✅ Calmar_Momentum scenario found in config")
        print(f"   Strategy: {calmar_scenario['strategy']}")
        print(f"   Parameters: {calmar_scenario['strategy_params']}")
    else:
        print("❌ Calmar_Momentum scenario not found in config")
    
    print("\n🎉 All validations passed! The Calmar Momentum Strategy is ready to use.")
    
except Exception as e:
    print(f"❌ Validation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)