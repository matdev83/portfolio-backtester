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
    print("=== Detailed Strategy Debug ===")
    
    # Get scenario
    scenarios = [s for s in BACKTEST_SCENARIOS if s['name'] == 'Momentum_Unfiltered']
    scenario = scenarios[0]
    
    print(f"Strategy params: {scenario['strategy_params']}")
    
    # Get data
    data_source = StooqDataSource()
    daily_data = data_source.get_data(
        tickers=GLOBAL_CONFIG["universe"] + [GLOBAL_CONFIG["benchmark"]],
        start_date=GLOBAL_CONFIG["start_date"],
        end_date=GLOBAL_CONFIG["end_date"]
    )
    
    monthly_closes = daily_data.resample("BME").last()
    
    # Get features
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
    
    # Create strategy
    strategy = MomentumStrategy(scenario['strategy_params'])
    
    # Get signal generator and test it
    generator = strategy.get_signal_generator()
    print(f"\nSignal generator: {type(generator).__name__}")
    print(f"Generator params: {generator._params()}")
    
    # Test scores generation
    scores = generator.scores(features)
    print(f"\nScores shape: {scores.shape}")
    print(f"Scores date range: {scores.index[0]} to {scores.index[-1]}")
    print(f"Scores non-NaN count: {scores.notna().sum().sum()}")
    
    # Test for a specific date
    test_date = monthly_closes.index[15]  # Should have momentum data
    print(f"\n=== Testing date: {test_date} ===")
    
    if test_date in scores.index:
        look = scores.loc[test_date]
        print(f"Scores for {test_date}:")
        print(f"  Shape: {look.shape}")
        print(f"  Non-NaN count: {look.notna().sum()}")
        print(f"  Non-zero count: {(look != 0).sum()}")
        print(f"  Sample values: {look.head().tolist()}")
        
        # Test candidate weight calculation
        if look.notna().any():
            look_clean = look.dropna()
            print(f"\nAfter dropna:")
            print(f"  Shape: {look_clean.shape}")
            print(f"  Sample values: {look_clean.head().tolist()}")
            
            # Test the candidate weight calculation manually
            cand = strategy._calculate_candidate_weights(look_clean)
            print(f"\nCandidate weights:")
            print(f"  Shape: {cand.shape}")
            print(f"  Non-zero count: {(cand != 0).sum()}")
            print(f"  Sum: {cand.sum():.6f}")
            print(f"  Sample values: {cand.head().tolist()}")
            
        else:
            print("All scores are NaN!")
    else:
        print(f"Date {test_date} not in scores index!")
        print(f"Available dates: {scores.index[:5].tolist()} ... {scores.index[-5:].tolist()}")
    
    # Test full signal generation
    print("\n=== Testing Full Signal Generation ===")
    test_prices = monthly_closes.loc[:test_date]
    test_features = {name: f.loc[:test_date] for name, f in features.items()}
    
    signals = strategy.generate_signals(test_prices, test_features, benchmark_monthly_closes.loc[:test_date])
    print(f"Final signals shape: {signals.shape}")
    print(f"Final signals non-zero count: {(signals != 0).sum().sum()}")
    
    if signals.shape[0] > 0:
        last_signals = signals.iloc[-1]
        print(f"Last row signals:")
        print(f"  Non-zero: {(last_signals != 0).sum()}")
        print(f"  Sum: {last_signals.sum():.6f}")
        print(f"  Sample: {last_signals.head().tolist()}")

if __name__ == "__main__":
    main() 