#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

from portfolio_backtester.config_loader import BACKTEST_SCENARIOS

# Check Momentum_Unfiltered scenario
momentum_scenario = next(s for s in BACKTEST_SCENARIOS if s['name'] == 'Momentum_Unfiltered')
print("=== Momentum_Unfiltered Configuration ===")
print(f"Strategy: {momentum_scenario['strategy']}")
print(f"Strategy params: {momentum_scenario.get('strategy_params', {})}")
print(f"Optimize list: {[opt['parameter'] for opt in momentum_scenario.get('optimize', [])]}")
print(f"Total optimize specs: {len(momentum_scenario.get('optimize', []))}")

# Check Test_Optuna_Minimal scenario
optuna_scenario = next(s for s in BACKTEST_SCENARIOS if s['name'] == 'Test_Optuna_Minimal')
print("\n=== Test_Optuna_Minimal Configuration ===")
print(f"Strategy: {optuna_scenario['strategy']}")
print(f"Strategy params: {optuna_scenario.get('strategy_params', {})}")
print(f"Optimize list: {[opt['parameter'] for opt in optuna_scenario.get('optimize', [])]}")
print(f"Total optimize specs: {len(optuna_scenario.get('optimize', []))}")

# Check if sma_filter_window is causing issues
for scenario_name, scenario in [("Momentum_Unfiltered", momentum_scenario), ("Test_Optuna_Minimal", optuna_scenario)]:
    sma_window = scenario.get('strategy_params', {}).get('sma_filter_window')
    print(f"\n{scenario_name} sma_filter_window: {sma_window} (type: {type(sma_window)})")
    
    # Check if any SMA features would be generated
    if sma_window is not None:
        print(f"  -> SMA feature would be generated: BenchmarkSMA(sma_filter_window={sma_window})")
    else:
        print(f"  -> No SMA feature should be generated")
        
    # Check lookback_months
    lookback = scenario.get('strategy_params', {}).get('lookback_months')
    print(f"  lookback_months: {lookback} (type: {type(lookback)})")
    
    if lookback is not None:
        print(f"  -> Momentum feature would be generated: Momentum(lookback_months={lookback})")
    else:
        print(f"  -> No Momentum feature should be generated") 