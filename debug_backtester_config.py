#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

from portfolio_backtester.backtester import Backtester
from portfolio_backtester.config_loader import GLOBAL_CONFIG, BACKTEST_SCENARIOS

# Create a simple mock args object
class MockArgs:
    def __init__(self):
        self.mode = 'backtest'
        self.scenario = 'Test_Optuna_Minimal'
        self.optimizer = 'optuna'
        self.n_trials = 2
        self.n_jobs = 1
        self.verbose = True
        self.save_results = False
        self.output_path = None
        self.storage_url = None
        self.study_name = None
        self.random_seed = 42

args = MockArgs()

# Find the Test_Optuna_Minimal scenario
test_scenario = next(s for s in BACKTEST_SCENARIOS if s['name'] == 'Test_Optuna_Minimal')

print("=== BEFORE Backtester initialization ===")
print(f"Strategy params: {test_scenario.get('strategy_params', {})}")
print(f"Optimize list: {[opt['parameter'] for opt in test_scenario.get('optimize', [])]}")

# Create backtester (this will call populate_default_optimizations)
backtester = Backtester(GLOBAL_CONFIG, [test_scenario], args)

print("\n=== AFTER Backtester initialization ===")
print(f"Strategy params: {backtester.scenarios[0].get('strategy_params', {})}")
print(f"Optimize list: {[opt['parameter'] for opt in backtester.scenarios[0].get('optimize', [])]}")

# Check specific parameters
scenario = backtester.scenarios[0]
lookback_value = scenario.get('strategy_params', {}).get('lookback_months')
num_holdings_value = scenario.get('strategy_params', {}).get('num_holdings')
print(f"\nlookback_months: {lookback_value} (type: {type(lookback_value)})")
print(f"num_holdings: {num_holdings_value} (type: {type(num_holdings_value)})")

# Check if these values are None
if lookback_value is None:
    print("ERROR: lookback_months is None - this will cause no momentum feature to be generated!")
if num_holdings_value is None:
    print("ERROR: num_holdings is None - this will cause issues with position sizing!") 