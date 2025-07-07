import sys
sys.path.insert(0, 'src')
from portfolio_backtester.config_loader import BACKTEST_SCENARIOS

scenario = [s for s in BACKTEST_SCENARIOS if s['name'] == 'Momentum_Unfiltered'][0]
print('Strategy params:')
for k, v in scenario.get('strategy_params', {}).items():
    print(f'  {k}: {v}')

print('\nOptimize params:')
for opt in scenario.get('optimize', []):
    print(f'  {opt["parameter"]}')

sma_in_params = 'sma_filter_window' in scenario.get('strategy_params', {})
sma_in_optimize = 'sma_filter_window' in [o['parameter'] for o in scenario.get('optimize', [])]

print(f'\nsma_filter_window in strategy_params: {sma_in_params}')
print(f'sma_filter_window in optimize: {sma_in_optimize}')

if sma_in_params:
    print(f'sma_filter_window value: {scenario["strategy_params"]["sma_filter_window"]}')

print('\nConfiguration looks correct!' if sma_in_params and not sma_in_optimize else 'Configuration issue detected!') 