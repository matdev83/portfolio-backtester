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
    print("=== BaseStrategy Debug ===")
    
    # Get scenario
    scenarios = [s for s in BACKTEST_SCENARIOS if s['name'] == 'Momentum_Unfiltered']
    scenario = scenarios[0]
    
    # Get data and features (shortened version)
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
    
    # Create strategy
    strategy = MomentumStrategy(scenario['strategy_params'])
    
    # Test the generate_signals method by manually stepping through it
    generator = strategy.get_signal_generator()
    scores = generator.scores(features)
    
    # Take a subset of data for testing
    test_date = monthly_closes.index[15]
    prices = monthly_closes.loc[:test_date]
    
    print(f"Testing with data up to: {test_date}")
    print(f"Strategy config: {strategy.strategy_config}")
    
    # Manually step through generate_signals logic
    weights = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    w_prev = pd.Series(index=prices.columns, dtype=float).fillna(0.0)
    
    # Check SMA filter setup
    sma_window = strategy.strategy_config.get("sma_filter_window")
    derisk_days = strategy.strategy_config.get("derisk_days_under_sma", 10)
    use_sma_derisk = sma_window and derisk_days > 0
    
    print(f"\nSMA Filter Setup:")
    print(f"  sma_window: {sma_window}")
    print(f"  derisk_days: {derisk_days}")
    print(f"  use_sma_derisk: {use_sma_derisk}")
    
    if sma_window is not None:
        sma_name = f"benchmark_sma_{sma_window}m"
        print(f"  Looking for feature: {sma_name}")
        print(f"  Available features: {list(features.keys())}")
        
        if sma_name in features:
            sma_risk_on_series = features[sma_name].reindex(prices.index, fill_value=True).astype(bool)
            print(f"  SMA risk on series shape: {sma_risk_on_series.shape}")
            print(f"  SMA risk on sample: {sma_risk_on_series.head().tolist()}")
        else:
            print(f"  ERROR: {sma_name} not found in features!")
            return
    else:
        sma_risk_on_series = pd.Series(True, index=prices.index, name="sma_risk_on")
        print(f"  Using default True series")
    
    # Check RoRo signal
    roro_signal_instance = strategy.get_roro_signal()
    print(f"\nRoRo Signal:")
    print(f"  Instance: {roro_signal_instance}")
    
    if roro_signal_instance:
        roro_signal_values = roro_signal_instance.generate_signal(prices.index)
        roro_risk_on_series = roro_signal_values.astype(bool)
        print(f"  RoRo risk on sample: {roro_risk_on_series.head().tolist()}")
    else:
        roro_risk_on_series = pd.Series(True, index=prices.index, name="roro_risk_on")
        print(f"  Using default True series")
    
    # Test for our specific date
    date = test_date
    print(f"\n=== Processing date: {date} ===")
    
    if date not in scores.index:
        print("Date not in scores index!")
        return
        
    look = scores.loc[date]
    print(f"Scores shape: {look.shape}, non-NaN: {look.notna().sum()}")
    
    if look.isna().any():
        print(f"Has NaN values, zero_if_nan: {generator.zero_if_nan}")
        if generator.zero_if_nan:
            print("Would zero out weights due to NaN")
            return
        else:
            print("Would maintain previous weights due to NaN")
            return
    
    if look.count() == 0:
        print("Look count is 0")
        return
    
    look = look.dropna()
    print(f"After dropna: {look.shape}")
    
    # Calculate candidate weights
    cand = strategy._calculate_candidate_weights(look)
    print(f"Candidate weights: non-zero={cand.nonzero()[0].shape[0]}, sum={cand.sum():.6f}")
    
    # Apply leverage and smoothing
    w_target_pre_filter = strategy._apply_leverage_and_smoothing(cand, w_prev)
    print(f"Pre-filter weights: non-zero={(w_target_pre_filter != 0).sum()}, sum={w_target_pre_filter.sum():.6f}")
    
    # Check stop loss
    sl_handler = strategy.get_stop_loss_handler()
    print(f"Stop loss handler: {type(sl_handler).__name__}")
    
    # Apply filters
    w_final = w_target_pre_filter.copy()
    
    print(f"\nApplying filters:")
    print(f"  use_sma_derisk: {use_sma_derisk}")
    if use_sma_derisk:
        derisk_flags = strategy._calculate_derisk_flags(sma_risk_on_series, derisk_days)
        if derisk_flags.loc[date]:
            print(f"  DERISKING due to SMA filter!")
            w_final[:] = 0.0
    
    print(f"  sma_window check: {sma_window is not None}")
    if sma_window is not None:
        sma_risk_on = sma_risk_on_series.loc[date]
        print(f"  SMA risk on for {date}: {sma_risk_on}")
        if not sma_risk_on:
            print(f"  ZEROING due to SMA risk off!")
            w_final[:] = 0.0
    
    roro_risk_on = roro_risk_on_series.loc[date]
    print(f"  RoRo risk on for {date}: {roro_risk_on}")
    if not roro_risk_on:
        print(f"  ZEROING due to RoRo risk off!")
        w_final[:] = 0.0
    
    print(f"\nFinal weights: non-zero={(w_final != 0).sum()}, sum={w_final.sum():.6f}")
    print(f"Sample final weights: {w_final.head().tolist()}")

if __name__ == "__main__":
    main() 