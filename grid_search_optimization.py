#!/usr/bin/env python3
"""
Grid search optimization for Momentum_Unfiltered_ATR strategy.
This systematically tests all combinations of specified parameter ranges.
"""

import sys
import os
import pandas as pd
import numpy as np
import itertools
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from portfolio_backtester.backtester import Backtester
from portfolio_backtester.reporting.performance_metrics import calculate_metrics

def grid_search_optimization():
    """Perform grid search optimization."""
    
    print("Starting grid search optimization for Momentum_Unfiltered_ATR...")
    
    # Initialize backtester
    backtester = Backtester()
    
    # Define parameter grids (start with smaller ranges for speed)
    param_grid = {
        'lookback_months': [6, 9, 12],
        'num_holdings': [10, 15, 20],
        'top_decile_fraction': [0.05, 0.1, 0.15],
        'leverage': [0.8, 1.0, 1.2],
        'atr_length': [10, 14, 20],
        'atr_multiple': [2.0, 2.5, 3.0]
    }
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))
    
    print(f"Testing {len(combinations)} parameter combinations...")
    
    results = []
    
    for i, combo in enumerate(combinations):
        if i % 10 == 0:  # Progress update every 10 iterations
            print(f"Progress: {i}/{len(combinations)} ({i/len(combinations)*100:.1f}%)")
        
        # Create parameter dict
        params = dict(zip(param_names, combo))
        params['smoothing_lambda'] = 0.0  # Fixed parameter
        
        # Create scenario config
        scenario_config = {
            "name": f"GridSearch_{i}",
            "strategy": "momentum_unfiltered_atr",
            "rebalance_frequency": "ME",
            "position_sizer": "equal_weight",
            "transaction_costs_bps": 10,
            "strategy_params": {
                "lookback_months": params['lookback_months'],
                "num_holdings": params['num_holdings'],
                "top_decile_fraction": params['top_decile_fraction'],
                "long_only": True,
                "smoothing_lambda": params['smoothing_lambda'],
                "leverage": params['leverage'],
                "sma_filter_window": None,
                "derisk_days_under_sma": 10,
                "apply_trading_lag": False,
                "price_column_asset": "Close",
                "price_column_benchmark": "Close",
            },
            "stop_loss_config": {
                "type": "AtrBasedStopLoss",
                "atr_length": params['atr_length'],
                "atr_multiple": params['atr_multiple']
            }
        }
        
        try:
            # Run backtest
            returns = backtester.run_scenario(
                scenario_config, 
                backtester.monthly_data, 
                backtester.daily_data_ohlc, 
                backtester.rets_full
            )
            
            if returns is not None and not returns.empty:
                # Calculate metrics
                benchmark_returns = backtester.rets_full
                metrics = calculate_metrics(returns, benchmark_returns)
                
                result = {
                    'combination_id': i,
                    'params': params,
                    'sortino': metrics.get('Sortino', np.nan),
                    'sharpe': metrics.get('Sharpe', np.nan),
                    'annual_return': metrics.get('Ann. Return', np.nan),
                    'max_drawdown': metrics.get('Max DD', np.nan),
                    'total_return': metrics.get('Total Return', np.nan),
                    'calmar': metrics.get('Calmar', np.nan)
                }
                results.append(result)
                
        except Exception as e:
            print(f"Error with combination {i}: {e}")
            continue
    
    # Sort and display results
    if results:
        # Sort by Sortino ratio
        results.sort(key=lambda x: x['sortino'] if not np.isnan(x['sortino']) else -999, reverse=True)
        
        print("\n" + "="*100)
        print("GRID SEARCH RESULTS (Top 10 by Sortino ratio)")
        print("="*100)
        
        for i, result in enumerate(results[:10]):
            print(f"\nRank {i+1}:")
            print(f"  Parameters: {result['params']}")
            print(f"  Sortino: {result['sortino']:.4f}")
            print(f"  Sharpe: {result['sharpe']:.4f}")
            print(f"  Calmar: {result['calmar']:.4f}")
            print(f"  Annual Return: {result['annual_return']:.2%}")
            print(f"  Max Drawdown: {result['max_drawdown']:.2%}")
            print(f"  Total Return: {result['total_return']:.2%}")
        
        # Save results to CSV
        df_results = pd.DataFrame([
            {**r['params'], **{k: v for k, v in r.items() if k != 'params'}} 
            for r in results
        ])
        df_results.to_csv('grid_search_results.csv', index=False)
        print(f"\nResults saved to grid_search_results.csv")
        
        # Best parameters
        best = results[0]
        print("\n" + "="*50)
        print("BEST PARAMETERS:")
        print("="*50)
        for param, value in best['params'].items():
            print(f"  {param}: {value}")
            
    else:
        print("No successful results obtained.")

if __name__ == "__main__":
    grid_search_optimization() 