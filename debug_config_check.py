#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

from portfolio_backtester.backtester import Backtester
from portfolio_backtester.config_loader import GLOBAL_CONFIG, BACKTEST_SCENARIOS

def main():
    print("=== Configuration Check ===")
    
    # Find the scenario
    scenarios = [s for s in BACKTEST_SCENARIOS if s['name'] == 'Momentum_Unfiltered']
    if not scenarios:
        print('ERROR: Momentum_Unfiltered not found')
        return
    
    scenario = scenarios[0]
    print(f'Strategy: {scenario["strategy"]}')
    print(f'SMA filter in strategy_params: {scenario["strategy_params"].get("sma_filter_window")}')
    print(f'Optimize list length: {len(scenario.get("optimize", []))}')
    
    # Check if SMA is in optimize list
    has_sma = False
    for opt in scenario.get('optimize', []):
        if 'sma_filter_window' in str(opt):
            has_sma = True
            print(f'Found SMA in optimize: {opt}')
    
    print(f'Has SMA in optimize: {has_sma}')
    
    # Check Test_Optuna_Minimal
    test_scenarios = [s for s in BACKTEST_SCENARIOS if s['name'] == 'Test_Optuna_Minimal']
    if test_scenarios:
        test_scenario = test_scenarios[0]
        print("\n=== Test_Optuna_Minimal Check ===")
        print(f'Strategy: {test_scenario["strategy"]}')
        print(f'SMA filter in strategy_params: {test_scenario["strategy_params"].get("sma_filter_window")}')
        print(f'Optimize list: {test_scenario.get("optimize", [])}')

if __name__ == "__main__":
    main() 