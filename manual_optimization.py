#!/usr/bin/env python3
"""
Manual parameter optimization for Momentum_Unfiltered_ATR strategy.
This approach tests specific parameter combinations without using Optuna.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from portfolio_backtester.backtester import Backtester
from portfolio_backtester.reporting.performance_metrics import calculate_metrics

def test_parameter_combination(params, backtester):
    """Test a specific parameter combination."""
    
    # Create scenario config with the parameters
    scenario_config = {
        "name": f"Test_{params['num_holdings']}_{params['atr_length']}_{params['atr_multiple']}",
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
        
        if returns is None or returns.empty:
            return None
            
        # Calculate metrics
        benchmark_returns = backtester.rets_full
        metrics = calculate_metrics(returns, benchmark_returns)
        
        return {
            'params': params,
            'sortino': metrics.get('Sortino', np.nan),
            'sharpe': metrics.get('Sharpe', np.nan),
            'annual_return': metrics.get('Ann. Return', np.nan),
            'max_drawdown': metrics.get('Max DD', np.nan),
            'total_return': metrics.get('Total Return', np.nan)
        }
        
    except Exception as e:
        print(f"Error testing params {params}: {e}")
        return None

def main():
    """Run manual parameter optimization."""
    
    print("Starting manual parameter optimization for Momentum_Unfiltered_ATR...")
    
    # Initialize backtester
    backtester = Backtester()
    
    # Define parameter combinations to test
    param_combinations = [
        # Format: lookback_months, num_holdings, top_decile_fraction, smoothing_lambda, leverage, atr_length, atr_multiple
        {"lookback_months": 6, "num_holdings": 10, "top_decile_fraction": 0.1, "smoothing_lambda": 0.0, "leverage": 1.0, "atr_length": 14, "atr_multiple": 2.5},
        {"lookback_months": 9, "num_holdings": 15, "top_decile_fraction": 0.1, "smoothing_lambda": 0.0, "leverage": 1.0, "atr_length": 14, "atr_multiple": 2.5},
        {"lookback_months": 12, "num_holdings": 20, "top_decile_fraction": 0.1, "smoothing_lambda": 0.0, "leverage": 1.0, "atr_length": 14, "atr_multiple": 2.5},
        
        # Test different ATR parameters
        {"lookback_months": 12, "num_holdings": 15, "top_decile_fraction": 0.1, "smoothing_lambda": 0.0, "leverage": 1.0, "atr_length": 10, "atr_multiple": 2.0},
        {"lookback_months": 12, "num_holdings": 15, "top_decile_fraction": 0.1, "smoothing_lambda": 0.0, "leverage": 1.0, "atr_length": 20, "atr_multiple": 3.0},
        
        # Test different leverage
        {"lookback_months": 12, "num_holdings": 15, "top_decile_fraction": 0.1, "smoothing_lambda": 0.0, "leverage": 0.8, "atr_length": 14, "atr_multiple": 2.5},
        {"lookback_months": 12, "num_holdings": 15, "top_decile_fraction": 0.1, "smoothing_lambda": 0.0, "leverage": 1.2, "atr_length": 14, "atr_multiple": 2.5},
        
        # Test different top_decile_fraction
        {"lookback_months": 12, "num_holdings": 15, "top_decile_fraction": 0.05, "smoothing_lambda": 0.0, "leverage": 1.0, "atr_length": 14, "atr_multiple": 2.5},
        {"lookback_months": 12, "num_holdings": 15, "top_decile_fraction": 0.15, "smoothing_lambda": 0.0, "leverage": 1.0, "atr_length": 14, "atr_multiple": 2.5},
    ]
    
    results = []
    
    for i, params in enumerate(param_combinations):
        print(f"\nTesting combination {i+1}/{len(param_combinations)}: {params}")
        
        result = test_parameter_combination(params, backtester)
        if result:
            results.append(result)
            print(f"  Sortino: {result['sortino']:.4f}, Sharpe: {result['sharpe']:.4f}, Annual Return: {result['annual_return']:.4f}")
        else:
            print("  Failed to calculate metrics")
    
    # Sort results by Sortino ratio
    results.sort(key=lambda x: x['sortino'] if not np.isnan(x['sortino']) else -999, reverse=True)
    
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS (sorted by Sortino ratio)")
    print("="*80)
    
    for i, result in enumerate(results[:5]):  # Show top 5
        print(f"\nRank {i+1}:")
        print(f"  Parameters: {result['params']}")
        print(f"  Sortino: {result['sortino']:.4f}")
        print(f"  Sharpe: {result['sharpe']:.4f}")
        print(f"  Annual Return: {result['annual_return']:.2%}")
        print(f"  Max Drawdown: {result['max_drawdown']:.2%}")
        print(f"  Total Return: {result['total_return']:.2%}")

if __name__ == "__main__":
    main() 