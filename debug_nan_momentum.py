#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from portfolio_backtester.config_loader import GLOBAL_CONFIG, BACKTEST_SCENARIOS
from portfolio_backtester.data_sources.stooq_data_source import StooqDataSource
from portfolio_backtester.feature_engineering import precompute_features
from portfolio_backtester.feature import get_required_features_from_scenarios
from portfolio_backtester import strategies

def main():
    print("=== Debug NaN Momentum Values ===")
    
    # Get scenario
    scenarios = [s for s in BACKTEST_SCENARIOS if s['name'] == 'Momentum_Unfiltered']
    scenario = scenarios[0]
    
    # Get data and features
    data_source = StooqDataSource()
    daily_data = data_source.get_data(
        tickers=GLOBAL_CONFIG["universe"] + [GLOBAL_CONFIG["benchmark"]],
        start_date=GLOBAL_CONFIG["start_date"],
        end_date=GLOBAL_CONFIG["end_date"]
    )
    
    monthly_closes = daily_data.resample("BME").last()
    strategy_registry = {"momentum": strategies.MomentumStrategy}
    required_features = get_required_features_from_scenarios([scenario], strategy_registry)
    
    benchmark_monthly_closes = monthly_closes[GLOBAL_CONFIG["benchmark"]]
    empty_cols = pd.MultiIndex.from_tuples([], names=['Ticker', 'Field'])
    monthly_data_for_features = pd.DataFrame(index=monthly_closes.index, columns=empty_cols)
    
    features = precompute_features(
        data=monthly_data_for_features,
        required_features=required_features, 
        benchmark_data=benchmark_monthly_closes,
        legacy_monthly_closes=monthly_closes
    )
    
    # Check momentum feature
    momentum = features['momentum_12m']
    test_date = monthly_closes.index[15]  # 2011-04-29
    
    print(f"Checking momentum on {test_date}")
    print(f"Monthly closes shape: {monthly_closes.shape}")
    print(f"Momentum shape: {momentum.shape}")
    
    # Check which assets have NaN momentum
    momentum_at_date = momentum.loc[test_date]
    nan_assets = momentum_at_date[momentum_at_date.isna()].index.tolist()
    valid_assets = momentum_at_date[momentum_at_date.notna()].index.tolist()
    
    print(f"\nAssets with NaN momentum: {len(nan_assets)}")
    print(f"Assets with valid momentum: {len(valid_assets)}")
    print(f"NaN assets: {nan_assets}")
    
    # Check the data for NaN assets
    if nan_assets:
        print(f"\nChecking data for first NaN asset: {nan_assets[0]}")
        asset = nan_assets[0]
        
        # Check if asset exists in monthly_closes
        if asset in monthly_closes.columns:
            asset_prices = monthly_closes[asset].loc[:test_date]
            print(f"  Asset prices shape: {asset_prices.shape}")
            print(f"  Asset prices head: {asset_prices.head().tolist()}")
            print(f"  Asset prices tail: {asset_prices.tail().tolist()}")
            print(f"  Asset prices NaN count: {asset_prices.isna().sum()}")
            
            # Check if there are enough non-NaN values for 12-month momentum
            non_nan_count = asset_prices.notna().sum()
            print(f"  Non-NaN price count: {non_nan_count}")
            
            if non_nan_count >= 13:  # Need at least 13 points for 12-month momentum
                print(f"  Should have valid momentum (has {non_nan_count} points)")
                
                # Manually calculate momentum to see what happens
                shifted = asset_prices.shift(0)  # skip_months = 0
                momentum_manual = (shifted / shifted.shift(12)) - 1
                momentum_manual_at_date = momentum_manual.loc[test_date]
                print(f"  Manual momentum calculation: {momentum_manual_at_date}")
                
                # Check the specific values used
                current_price = shifted.loc[test_date]
                past_price = shifted.shift(12).loc[test_date]
                print(f"  Current price: {current_price}")
                print(f"  Past price (12m ago): {past_price}")
                
            else:
                print(f"  Not enough data for momentum (only {non_nan_count} points)")
        else:
            print(f"  Asset {asset} not found in monthly_closes!")
    
    # Check if this is a universe vs data mismatch
    print(f"\nUniverse check:")
    print(f"  Universe size: {len(GLOBAL_CONFIG['universe'])}")
    print(f"  Monthly closes columns: {len(monthly_closes.columns)}")
    print(f"  Momentum columns: {len(momentum.columns)}")
    
    universe_assets = set(GLOBAL_CONFIG['universe'])
    data_assets = set(monthly_closes.columns)
    momentum_assets = set(momentum.columns)
    
    missing_in_data = universe_assets - data_assets
    missing_in_momentum = data_assets - momentum_assets
    
    print(f"  Assets in universe but not in data: {missing_in_data}")
    print(f"  Assets in data but not in momentum: {missing_in_momentum}")

if __name__ == "__main__":
    main() 