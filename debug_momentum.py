#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
sys.path.insert(0, 'src')

from portfolio_backtester.config_loader import BACKTEST_SCENARIOS, GLOBAL_CONFIG
from portfolio_backtester.strategies.momentum_strategy import MomentumStrategy
from portfolio_backtester.data_sources.yfinance_data_source import YFinanceDataSource
from portfolio_backtester.data_sources.stooq_data_source import StooqDataSource
from portfolio_backtester.feature_engineering import precompute_features
from portfolio_backtester.feature import get_required_features_from_scenarios
from portfolio_backtester.utils import _resolve_strategy

def debug_momentum():
    print("=== Debugging Momentum Strategy ===")
    
    # Get the scenario
    scenarios = [s for s in BACKTEST_SCENARIOS if s['name'] == 'Momentum_Unfiltered']
    if not scenarios:
        print("Scenario not found!")
        return
    
    scenario = scenarios[0]
    print(f"Scenario: {scenario['name']}")
    print(f"Strategy params: {scenario.get('strategy_params', {})}")
    print(f"Optimize params: {[o['parameter'] for o in scenario.get('optimize', [])]}")
    
    # Create strategy instance
    strategy = MomentumStrategy(scenario)
    print(f"Strategy created: {type(strategy).__name__}")
    
    # Get signal generator
    generator = strategy.get_signal_generator()
    print(f"Signal generator: {type(generator).__name__}")
    print(f"zero_if_nan: {generator.zero_if_nan}")
    
    # Get required features
    required_features = strategy.get_required_features(scenario)
    print(f"Required features: {[f.name for f in required_features]}")
    print(f"Required features objects: {[type(f).__name__ for f in required_features]}")
    print(f"Required features raw: {required_features}")
    
    # Check if they're actually Feature objects
    for f in required_features:
        print(f"Feature: {f}, type: {type(f)}, has compute: {hasattr(f, 'compute')}")
    
    # Get data (use same config as actual backtester)
    data_source = StooqDataSource()
    universe = GLOBAL_CONFIG["universe"][:10]  # Use first 10 assets to keep it manageable
    benchmark = GLOBAL_CONFIG["benchmark"]
    start_date = GLOBAL_CONFIG["start_date"]
    end_date = "2012-12-31"  # Shorter range for debugging
    
    print(f"Using data source: Stooq")
    print(f"Universe (first 10): {universe}")
    print(f"Benchmark: {benchmark}")
    print("Fetching data...")
    data = data_source.get_data(universe + [benchmark], start_date, end_date)
    
    if data is None or data.empty:
        print("No data retrieved!")
        return
    
    print(f"Data shape: {data.shape}")
    print(f"Data date range: {data.index.min()} to {data.index.max()}")
    
    # Only proceed if we have valid Feature objects
    valid_features = [f for f in required_features if hasattr(f, 'compute')]
    if not valid_features:
        print("No valid Feature objects found!")
        return
    
    # Precompute features
    print("Precomputing features...")
    features = precompute_features(
        data, valid_features, benchmark_data=data[benchmark]
    )
    
    print(f"Features computed: {list(features.keys())}")
    
    # Check momentum feature
    momentum_feature_name = None
    for name in features.keys():
        if 'momentum' in name:
            momentum_feature_name = name
            break
    
    if momentum_feature_name:
        momentum_data = features[momentum_feature_name]
        print(f"\nMomentum feature: {momentum_feature_name}")
        print(f"Shape: {momentum_data.shape}")
        print(f"Date range: {momentum_data.index.min()} to {momentum_data.index.max()}")
        print(f"First 5 values:\n{momentum_data.head()}")
        print(f"Last 5 values:\n{momentum_data.tail()}")
        print(f"NaN count: {momentum_data.isna().sum().sum()}")
        print(f"Non-NaN count: {momentum_data.notna().sum().sum()}")
    
    # Generate signals
    print("\nGenerating signals...")
    monthly_data = data.resample('ME').last()
    prices = monthly_data[universe]
    benchmark_data = monthly_data[benchmark]
    
    signals = strategy.generate_signals(prices, features, benchmark_data)
    print(f"Signals shape: {signals.shape}")
    print(f"Signals date range: {signals.index.min()} to {signals.index.max()}")
    print(f"First 5 signals:\n{signals.head()}")
    print(f"Last 5 signals:\n{signals.tail()}")
    print(f"Non-zero signal count: {(signals.abs() > 1e-9).sum().sum()}")
    print(f"Total signal sum by date:\n{signals.sum(axis=1).head(10)}")

if __name__ == "__main__":
    debug_momentum() 