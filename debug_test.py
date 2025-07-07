#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

from portfolio_backtester.backtester import Backtester
from portfolio_backtester.config_loader import GLOBAL_CONFIG, BACKTEST_SCENARIOS
import pandas as pd
import numpy as np

# Find the Test_Optuna_Minimal scenario
scenario = next(s for s in BACKTEST_SCENARIOS if s['name'] == 'Test_Optuna_Minimal')

print("=== Test_Optuna_Minimal Configuration ===")
print(f"Strategy: {scenario['strategy']}")
print(f"Strategy params: {scenario['strategy_params']}")
print(f"Optimize: {scenario['optimize']}")

# Create a simple mock args object
class MockArgs:
    def __init__(self):
        self.mode = 'optimize'
        self.scenario = 'Test_Optuna_Minimal'
        self.optimizer = 'optuna'
        self.n_trials = 2  # Very small for debugging
        self.n_jobs = 1
        self.verbose = True
        self.save_results = False
        self.output_path = None
        self.storage_url = None
        self.study_name = None

args = MockArgs()

# Create backtester
backtester = Backtester(GLOBAL_CONFIG, [scenario], args)

print("\n=== Running Optimization ===")
try:
    results = backtester.run()
    print(f"Results: {results}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 