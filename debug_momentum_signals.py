#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from portfolio_backtester.backtester import Backtester
from portfolio_backtester.config_loader import GLOBAL_CONFIG, BACKTEST_SCENARIOS
from portfolio_backtester.data_sources.stooq_data_source import StooqDataSource
from portfolio_backtester.feature_engineering import precompute_features

def main():
    print("=== Debug Momentum Signal Generation ===")
    
    # Get the scenario
    scenarios = [s for s in BACKTEST_SCENARIOS if s['name'] == 'Momentum_Unfiltered']
    scenario = scenarios[0]
    
    print(f"Strategy params: {scenario['strategy_params']}")
    
    # Fetch data manually like the backtester does
    print("\n=== Fetching Data ===")
    data_source = StooqDataSource()
    daily_data, monthly_data = data_source.fetch_data(
        GLOBAL_CONFIG["universe"], 
        start_date="2010-01-01", 
        end_date="2025-06-30"
    )
    
    print(f"Monthly data shape: {monthly_data.shape}")
    print(f"Monthly data date range: {monthly_data.index[0]} to {monthly_data.index[-1]}")
    print(f"Monthly data head:\n{monthly_data.head()}")
    
    # Precompute features
    print("\n=== Computing Features ===")
    monthly_closes = monthly_data
    rets_full = daily_data.pct_change(fill_method=None).fillna(0)
    
    features = precompute_features(
        monthly_data, daily_data, rets_full, legacy_monthly_closes=monthly_closes
    )
    
    print(f"Features computed: {list(features.keys())}")
    
    # Check momentum feature specifically
    if 'momentum_12m' in features:
        momentum_feature = features['momentum_12m']
        print(f"\nMomentum feature shape: {momentum_feature.shape}")
        print(f"Momentum feature date range: {momentum_feature.index[0]} to {momentum_feature.index[-1]}")
        print(f"Momentum feature head:\n{momentum_feature.head()}")
        print(f"Momentum feature tail:\n{momentum_feature.tail()}")
        print(f"Momentum non-NaN values: {momentum_feature.notna().sum().sum()}")
        print(f"Momentum non-zero values: {(momentum_feature != 0).sum().sum()}")
        
        # Check specific values
        print(f"Sample momentum values (first asset):")
        first_asset = momentum_feature.columns[0]
        sample_values = momentum_feature[first_asset].dropna().head(10)
        print(f"  {first_asset}: {sample_values.tolist()}")
    else:
        print("ERROR: momentum_12m feature not found!")
        
    # Now test strategy signal generation
    print("\n=== Testing Strategy Signal Generation ===")
    from portfolio_backtester.strategies.momentum_strategy import MomentumStrategy
    
    strategy = MomentumStrategy(scenario['strategy_params'])
    
    # Test signal generation for a few dates
    test_dates = monthly_data.index[12:17]  # Skip first 12 months for momentum
    
    for date in test_dates:
        print(f"\nTesting signals for {date}:")
        
        # Get data up to this date
        data_slice = monthly_data.loc[:date]
        features_slice = {name: f.loc[:date] for name, f in features.items()}
        
        try:
            signals = strategy.generate_signals(data_slice, features_slice, date)
            print(f"  Signals shape: {signals.shape}")
            print(f"  Non-zero signals: {(signals != 0).sum()}")
            print(f"  Signal sum: {signals.sum():.6f}")
            print(f"  Sample signals: {signals.head().tolist()}")
            
            if (signals != 0).sum() > 0:
                print(f"  SUCCESS: Found non-zero signals!")
                break
                
        except Exception as e:
            print(f"  ERROR generating signals: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 