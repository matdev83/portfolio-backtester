#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

from portfolio_backtester.config_loader import OPTIMIZER_PARAMETER_DEFAULTS
from portfolio_backtester.config_initializer import populate_default_optimizations, _get_default_value_for_parameter

# Test the _get_default_value_for_parameter function
print("=== Testing _get_default_value_for_parameter ===")
lookback_config = OPTIMIZER_PARAMETER_DEFAULTS.get("lookback_months", {})
print(f"lookback_months config: {lookback_config}")
default_value = _get_default_value_for_parameter(lookback_config)
print(f"Default value for lookback_months: {default_value}")

# Test with a minimal scenario
test_scenario = {
    "name": "Test_Debug",
    "strategy": "momentum",
    "strategy_params": {
        "long_only": True,
        "top_decile_fraction": 0.2,
        "smoothing_lambda": 0.5,
        "leverage": 1.0,
        "sma_filter_window": None
    },
    "optimize": [
        {"parameter": "lookback_months"},
        {"parameter": "num_holdings"}
    ]
}

print(f"\n=== Before populate_default_optimizations ===")
print(f"Strategy params: {test_scenario['strategy_params']}")
print(f"Optimize list: {[opt['parameter'] for opt in test_scenario['optimize']]}")

# Apply the function
populate_default_optimizations([test_scenario], OPTIMIZER_PARAMETER_DEFAULTS)

print(f"\n=== After populate_default_optimizations ===")
print(f"Strategy params: {test_scenario['strategy_params']}")
print(f"Optimize list: {[opt['parameter'] for opt in test_scenario['optimize']]}")

# Check specific parameters
lookback_value = test_scenario['strategy_params'].get('lookback_months')
num_holdings_value = test_scenario['strategy_params'].get('num_holdings')
print(f"\nlookback_months: {lookback_value} (type: {type(lookback_value)})")
print(f"num_holdings: {num_holdings_value} (type: {type(num_holdings_value)})") 