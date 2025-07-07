#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from portfolio_backtester.config_loader import GLOBAL_CONFIG, BACKTEST_SCENARIOS
from portfolio_backtester.data_sources.stooq_data_source import StooqDataSource
from portfolio_backtester.feature_engineering import precompute_features
from portfolio_backtester.strategies.momentum_strategy import MomentumStrategy
from portfolio_backtester.feature import get_required_features_from_scenarios
from portfolio_backtester import strategies

def main():
    print("=== Simple Momentum Debug ===")
    
    # Get scenario
    scenarios = [s for s in BACKTEST_SCENARIOS if s['name'] == 'Momentum_Unfiltered']
    scenario = scenarios[0]
    
    print(f"Strategy params: {scenario['strategy_params']}")
    
    # Get data using the same approach as backtester
    data_source = StooqDataSource()
    daily_data = data_source.get_data(
        tickers=GLOBAL_CONFIG["universe"] + [GLOBAL_CONFIG["benchmark"]],
        start_date=GLOBAL_CONFIG["start_date"],
        end_date=GLOBAL_CONFIG["end_date"]
    )
    
    print(f"Daily data shape: {daily_data.shape}")
    print(f"Daily data columns: {daily_data.columns[:5].tolist()}...")  # First 5 tickers
    
    # Resample to monthly (like backtester does)
    monthly_closes = daily_data.resample("BME").last()
    
    print(f"Monthly data shape: {monthly_closes.shape}")
    print(f"Monthly data date range: {monthly_closes.index[0]} to {monthly_closes.index[-1]}")
    
    # Get required features (like backtester does)
    strategy_registry = {
        "momentum": strategies.MomentumStrategy,
    }
    
    required_features = get_required_features_from_scenarios([scenario], strategy_registry)
    print(f"Required features: {[f.name for f in required_features]}")
    
    # Precompute features (like backtester does)
    benchmark_monthly_closes = monthly_closes[GLOBAL_CONFIG["benchmark"]]
    rets_full = daily_data.pct_change(fill_method=None).fillna(0)
    
    # Create empty monthly OHLC data (old format warning)
    empty_cols = pd.MultiIndex.from_tuples([], names=['Ticker', 'Field'])
    monthly_data_for_features = pd.DataFrame(index=monthly_closes.index, columns=empty_cols)
    
    features = precompute_features(
        data=monthly_data_for_features,
        required_features=required_features, 
        benchmark_data=benchmark_monthly_closes,
        legacy_monthly_closes=monthly_closes
    )
    
    print(f"Features computed: {list(features.keys())}")
    
    # Check momentum feature
    if 'momentum_12m' in features:
        momentum = features['momentum_12m']
        print(f"\nMomentum feature:")
        print(f"  Shape: {momentum.shape}")
        print(f"  Date range: {momentum.index[0]} to {momentum.index[-1]}")
        print(f"  Non-NaN count: {momentum.notna().sum().sum()}")
        print(f"  Sample values (first asset): {momentum.iloc[12:17, 0].tolist()}")
    
    # Test strategy signal generation
    print("\n=== Testing Strategy ===")
    strategy = MomentumStrategy(scenario['strategy_params'])
    
    # Test for a date where momentum should be available
    test_date = monthly_closes.index[15]  # Should have 12+ months of history
    print(f"Testing signals for {test_date}")
    
    signals = strategy.generate_signals(monthly_closes.loc[:test_date], features, test_date)
    print(f"Signals shape: {signals.shape}")
    print(f"Non-zero signals: {(signals != 0).sum()}")
    signal_sum = signals.sum()
    print(f"Signal sum: {signal_sum}")
    print(f"Sample signals: {signals.head().tolist()}")
    
    if (signals != 0).sum() == 0:
        print("\n❌ PROBLEM: All signals are zero!")
        
        # Debug the strategy internals
        print("\n=== Debugging Strategy Internals ===")
        
        # Check if momentum values exist for this date
        if 'momentum_12m' in features:
            momentum_at_date = features['momentum_12m'].loc[test_date]
            print(f"Momentum values at {test_date}:")
            print(f"  Non-NaN: {momentum_at_date.notna().sum()}")
            print(f"  Non-zero: {(momentum_at_date != 0).sum()}")
            print(f"  Sample values: {momentum_at_date.head().tolist()}")
        
    else:
        print("✅ SUCCESS: Found non-zero signals!")

if __name__ == "__main__":
    main() 