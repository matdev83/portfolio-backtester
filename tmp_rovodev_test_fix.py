#!/usr/bin/env python3
"""
Test script to verify that the populate_default_optimizations fix works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from portfolio_backtester.config_loader import load_config, BACKTEST_SCENARIOS, OPTIMIZER_PARAMETER_DEFAULTS
from portfolio_backtester.config_initializer import populate_default_optimizations

def test_momentum_dvol_sizer_params():
    """Test that Momentum_DVOL_Sizer scenario gets the required parameters."""
    
    # Load fresh config
    load_config()
    
    # Find the Momentum_DVOL_Sizer scenario
    momentum_dvol_scenario = None
    for scenario in BACKTEST_SCENARIOS:
        if scenario["name"] == "Momentum_DVOL_Sizer":
            momentum_dvol_scenario = scenario.copy()  # Make a copy to avoid modifying original
            break
    
    if not momentum_dvol_scenario:
        print("ERROR: Momentum_DVOL_Sizer scenario not found!")
        return False
    
    print("Before populate_default_optimizations:")
    print(f"  strategy_params: {momentum_dvol_scenario.get('strategy_params', {})}")
    print(f"  optimize parameters: {[opt['parameter'] for opt in momentum_dvol_scenario.get('optimize', [])]}")
    
    # Apply the populate_default_optimizations function
    populate_default_optimizations([momentum_dvol_scenario], OPTIMIZER_PARAMETER_DEFAULTS)
    
    print("\nAfter populate_default_optimizations:")
    print(f"  strategy_params: {momentum_dvol_scenario.get('strategy_params', {})}")
    print(f"  optimize parameters: {[opt['parameter'] for opt in momentum_dvol_scenario.get('optimize', [])]}")
    
    # Check if the required parameters are now present
    strategy_params = momentum_dvol_scenario.get('strategy_params', {})
    required_params = ['sizer_dvol_window', 'target_volatility']
    
    success = True
    for param in required_params:
        if param in strategy_params:
            print(f"✓ {param}: {strategy_params[param]}")
        else:
            print(f"✗ {param}: MISSING")
            success = False
    
    return success

if __name__ == "__main__":
    print("Testing populate_default_optimizations fix...")
    success = test_momentum_dvol_sizer_params()
    
    if success:
        print("\n✓ SUCCESS: All required parameters are now present in strategy_params!")
    else:
        print("\n✗ FAILURE: Some required parameters are still missing!")
    
    sys.exit(0 if success else 1)