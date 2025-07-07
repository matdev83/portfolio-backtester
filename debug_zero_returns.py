#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from portfolio_backtester.backtester import Backtester
from portfolio_backtester.config_loader import GLOBAL_CONFIG, BACKTEST_SCENARIOS

class MockArgs:
    def __init__(self):
        self.mode = 'backtest'
        self.scenario = 'Momentum_Unfiltered'
        self.optimizer = 'optuna'
        self.n_trials = 2
        self.n_jobs = 1
        self.verbose = True
        self.save_results = False
        self.output_path = None
        self.storage_url = None
        self.study_name = None
        self.random_seed = 42
        self.log_level = 'DEBUG'

def main():
    print("=== Debug Zero Returns Issue ===")
    
    # Find the scenario
    scenarios = [s for s in BACKTEST_SCENARIOS if s['name'] == 'Momentum_Unfiltered']
    if not scenarios:
        print('ERROR: Momentum_Unfiltered not found')
        return
    
    scenario = scenarios[0]
    print(f"Testing scenario: {scenario['name']}")
    print(f"Strategy: {scenario['strategy']}")
    print(f"SMA filter: {scenario['strategy_params'].get('sma_filter_window')}")
    
    # Create backtester
    args = MockArgs()
    backtester = Backtester(GLOBAL_CONFIG, [scenario], args, random_state=42)
    
    # Run the scenario
    try:
        print("\n=== Running Backtester ===")
        backtester.run()
        
        # Check results
        if backtester.results:
            for name, result in backtester.results.items():
                returns = result['returns']
                print(f"\nResults for {name}:")
                print(f"  Returns length: {len(returns)}")
                print(f"  Returns sum: {returns.sum():.6f}")
                print(f"  Returns mean: {returns.mean():.6f}")
                print(f"  Returns std: {returns.std():.6f}")
                print(f"  Non-zero returns: {(returns != 0).sum()}")
                print(f"  First 10 returns: {returns.head(10).tolist()}")
                print(f"  Last 10 returns: {returns.tail(10).tolist()}")
        else:
            print("No results found!")
            
    except Exception as e:
        print(f"Error running backtester: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 