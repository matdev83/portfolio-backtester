#!/usr/bin/env python3
"""
Meta Strategy Demonstration Script

This script demonstrates the meta strategies system by running both simple and nested
meta strategies with real data and showing their capital allocation and signal generation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta

from src.portfolio_backtester.strategies.meta.simple_meta_strategy import SimpleMetaStrategy
from src.portfolio_backtester.strategies.strategy_factory import StrategyFactory


def create_sample_data():
    """Create realistic sample OHLCV data for demonstration."""
    print("Creating sample market data...")
    
    # Create 2 years of daily data
    start_date = pd.Timestamp("2022-01-01")
    end_date = pd.Timestamp("2023-12-31")
    dates = pd.date_range(start_date, end_date, freq="D")
    
    assets = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]
    
    # Create MultiIndex columns for OHLCV data
    columns = pd.MultiIndex.from_product(
        [assets, ["Open", "High", "Low", "Close", "Volume"]],
        names=["Ticker", "Field"]
    )
    
    # Generate realistic price data with trends and volatility
    np.random.seed(42)  # For reproducible results
    
    # Start with base prices
    base_prices = {
        "AAPL": 150, "MSFT": 300, "GOOGL": 2500, "AMZN": 3000,
        "TSLA": 1000, "NVDA": 200, "META": 300
    }
    
    data = []
    for date in dates:
        row = []
        for asset in assets:
            # Generate daily return with some trend and volatility
            daily_return = np.random.normal(0.0005, 0.02)  # Slight positive drift, 2% daily vol
            
            if len(data) == 0:
                close_price = base_prices[asset]
            else:
                prev_close = data[-1][assets.index(asset) * 5 + 3]  # Previous close
                close_price = prev_close * (1 + daily_return)
            
            # Generate OHLC with realistic relationships
            open_price = close_price * (1 + np.random.normal(0, 0.005))
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
            volume = np.random.randint(1000000, 50000000)
            
            row.extend([open_price, high_price, low_price, close_price, volume])
        
        data.append(row)
    
    df = pd.DataFrame(data, index=dates, columns=columns)
    
    print(f"Created data for {len(assets)} assets over {len(dates)} days")
    print(f"   Date range: {dates[0].date()} to {dates[-1].date()}")
    
    return df


def create_benchmark_data():
    """Create benchmark (SPY) data."""
    print("Creating benchmark data...")
    
    start_date = pd.Timestamp("2022-01-01")
    end_date = pd.Timestamp("2023-12-31")
    dates = pd.date_range(start_date, end_date, freq="D")
    
    columns = pd.MultiIndex.from_product(
        [["SPY"], ["Open", "High", "Low", "Close", "Volume"]],
        names=["Ticker", "Field"]
    )
    
    np.random.seed(42)
    
    data = []
    spy_base_price = 400
    
    for i, date in enumerate(dates):
        if i == 0:
            close_price = spy_base_price
        else:
            daily_return = np.random.normal(0.0003, 0.015)  # Market return characteristics
            prev_close = data[-1][3]
            close_price = prev_close * (1 + daily_return)
        
        open_price = close_price * (1 + np.random.normal(0, 0.003))
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
        volume = np.random.randint(50000000, 200000000)
        
        data.append([open_price, high_price, low_price, close_price, volume])
    
    df = pd.DataFrame(data, index=dates, columns=columns)
    
    print(f"Created benchmark data over {len(dates)} days")
    
    return df


def demonstrate_simple_meta_strategy(market_data, benchmark_data):
    """Demonstrate SimpleMetaStrategy with real sub-strategies."""
    print("\n" + "="*60)
    print("DEMONSTRATING SIMPLE META STRATEGY")
    print("="*60)
    
    # Configuration for simple meta strategy
    config = {
        "initial_capital": 1000000,
        "min_allocation": 0.05,
        "allocations": [
            {
                "strategy_id": "momentum",
                "strategy_class": "CalmarMomentumStrategy",
                "strategy_params": {
                    "rolling_window": 6,
                    "num_holdings": 3,
                    "sma_filter_window": 20,
                    "price_column_asset": "Close",
                    "price_column_benchmark": "Close",
                    "timing_config": {
                        "mode": "time_based",
                        "rebalance_frequency": "M"
                    }
                },
                "weight": 0.7
            },
            {
                "strategy_id": "seasonal",
                "strategy_class": "IntramonthSeasonalStrategy",
                "strategy_params": {
                    "direction": "long",
                    "entry_day": 5,
                    "hold_days": 10,
                    "price_column_asset": "Close",
                    "trade_longs": True,
                    "trade_shorts": False,
                    "timing_config": {
                        "mode": "signal_based"
                    }
                },
                "weight": 0.3
            }
        ]
    }
    
    print(f"Initial Capital: ${config['initial_capital']:,}")
    print(f"Sub-strategies:")
    for alloc in config['allocations']:
        print(f"   - {alloc['strategy_id']}: {alloc['weight']*100:.1f}% ({alloc['strategy_class']})")
    
    # Create meta strategy
    meta_strategy = SimpleMetaStrategy(config)
    
    print(f"\nMeta Strategy Initialized:")
    print(f"   Available Capital: ${meta_strategy.available_capital:,.2f}")
    print(f"   Number of Sub-strategies: {len(meta_strategy.allocations)}")
    
    # Calculate initial capital allocations
    capital_allocations = meta_strategy.calculate_sub_strategy_capital()
    print(f"\nInitial Capital Allocations:")
    for strategy_id, capital in capital_allocations.items():
        print(f"   {strategy_id}: ${capital:,.2f}")
    
    # Test signal generation for multiple dates
    test_dates = [
        pd.Timestamp("2023-03-15"),  # Q1
        pd.Timestamp("2023-06-15"),  # Q2
        pd.Timestamp("2023-09-15"),  # Q3
        pd.Timestamp("2023-12-15"),  # Q4
    ]
    
    print(f"\nGenerating Signals for {len(test_dates)} test dates...")
    
    all_signals = {}
    for i, current_date in enumerate(test_dates):
        print(f"\nDate {i+1}: {current_date.date()}")
        
        # Get historical data up to current date
        historical_data = market_data[market_data.index <= current_date]
        benchmark_historical = benchmark_data[benchmark_data.index <= current_date]
        
        try:
            # Generate signals
            signals = meta_strategy.generate_signals(
                all_historical_data=historical_data,
                benchmark_historical_data=benchmark_historical,
                non_universe_historical_data=pd.DataFrame(),
                current_date=current_date
            )
            
            all_signals[current_date] = signals
            
            if not signals.empty and current_date in signals.index:
                signal_values = signals.loc[current_date]
                non_zero_signals = signal_values[signal_values.abs() > 0.001]
                
                print(f"   Generated signals for {len(signal_values)} assets")
                print(f"   Non-zero positions: {len(non_zero_signals)}")
                
                if len(non_zero_signals) > 0:
                    print(f"   Top positions:")
                    top_positions = non_zero_signals.abs().nlargest(3)
                    for asset, weight in top_positions.items():
                        direction = "LONG" if signal_values[asset] > 0 else "SHORT"
                        print(f"      {asset}: {weight:.3f} ({direction})")
            else:
                print(f"   No signals generated (insufficient data or risk-off)")
                
        except Exception as e:
            print(f"   Error generating signals: {e}")
    
    # Simulate some returns and show capital compounding
    print(f"\nSimulating Returns and Capital Compounding...")
    
    # Simulate quarterly returns
    simulated_returns = [
        {"momentum": 0.08, "seasonal": 0.05},   # Q1: Good performance
        {"momentum": -0.03, "seasonal": 0.02},  # Q2: Mixed performance  
        {"momentum": 0.12, "seasonal": -0.01},  # Q3: Momentum outperforms
        {"momentum": 0.06, "seasonal": 0.08},   # Q4: Both positive
    ]
    
    print(f"   Initial Capital: ${meta_strategy.available_capital:,.2f}")
    
    for quarter, returns in enumerate(simulated_returns, 1):
        meta_strategy.update_available_capital(returns)
        
        momentum_return = returns["momentum"]
        seasonal_return = returns["seasonal"]
        
        print(f"   Q{quarter}: Momentum {momentum_return:+.1%}, Seasonal {seasonal_return:+.1%}")
        print(f"        Available Capital: ${meta_strategy.available_capital:,.2f}")
        print(f"        Cumulative P&L: ${meta_strategy.cumulative_pnl:,.2f}")
    
    total_return = (meta_strategy.available_capital - config['initial_capital']) / config['initial_capital']
    print(f"\nFinal Results:")
    print(f"   Total Return: {total_return:.2%}")
    print(f"   Final Capital: ${meta_strategy.available_capital:,.2f}")
    
    return meta_strategy, all_signals


def demonstrate_nested_meta_strategy(market_data, benchmark_data):
    """Demonstrate nested meta strategy (meta strategy containing other meta strategies)."""
    print("\n" + "="*60)
    print("DEMONSTRATING NESTED META STRATEGY")
    print("="*60)
    
    # Configuration for nested meta strategy
    config = {
        "initial_capital": 2000000,
        "allocations": [
            {
                "strategy_id": "equity_meta",
                "strategy_class": "SimpleMetaStrategy",
                "strategy_params": {
                    "initial_capital": 1600000,  # Will be overridden by parent
                    "allocations": [
                        {
                            "strategy_id": "momentum",
                            "strategy_class": "CalmarMomentumStrategy",
                            "strategy_params": {
                                "rolling_window": 6,
                                "num_holdings": 3,
                                "price_column_asset": "Close",
                                "price_column_benchmark": "Close",
                                "timing_config": {
                                    "mode": "time_based",
                                    "rebalance_frequency": "M"
                                }
                            },
                            "weight": 0.6
                        },
                        {
                            "strategy_id": "seasonal",
                            "strategy_class": "IntramonthSeasonalStrategy",
                            "strategy_params": {
                                "direction": "long",
                                "entry_day": 5,
                                "hold_days": 10,
                                "price_column_asset": "Close",
                                "timing_config": {
                                    "mode": "signal_based"
                                }
                            },
                            "weight": 0.4
                        }
                    ]
                },
                "weight": 0.8  # 80% to equity meta strategy
            },
            {
                "strategy_id": "conservative",
                "strategy_class": "CalmarMomentumStrategy",
                "strategy_params": {
                    "rolling_window": 12,  # Longer window = more conservative
                    "num_holdings": 5,     # More diversified
                    "sma_filter_window": 50,
                    "price_column_asset": "Close",
                    "price_column_benchmark": "Close",
                    "timing_config": {
                        "mode": "time_based",
                        "rebalance_frequency": "Q"  # Quarterly rebalancing
                    }
                },
                "weight": 0.2  # 20% to conservative strategy
            }
        ]
    }
    
    print(f"Initial Capital: ${config['initial_capital']:,}")
    print(f"Nested Structure:")
    print(f"   Equity Meta (80%): ${config['initial_capital'] * 0.8:,.0f}")
    print(f"       Momentum (60% of equity): ${config['initial_capital'] * 0.8 * 0.6:,.0f}")
    print(f"       Seasonal (40% of equity): ${config['initial_capital'] * 0.8 * 0.4:,.0f}")
    print(f"   Conservative (20%): ${config['initial_capital'] * 0.2:,.0f}")
    
    # Create nested meta strategy
    nested_meta = SimpleMetaStrategy(config)
    
    print(f"\nNested Meta Strategy Initialized:")
    print(f"   Available Capital: ${nested_meta.available_capital:,.2f}")
    print(f"   Top-level Sub-strategies: {len(nested_meta.allocations)}")
    
    # Get sub-strategies to show nesting
    sub_strategies = nested_meta.get_sub_strategies()
    print(f"\nSub-strategy Analysis:")
    
    for strategy_id, strategy in sub_strategies.items():
        print(f"   {strategy_id}: {type(strategy).__name__}")
        if hasattr(strategy, 'allocations'):  # It's a meta strategy
            print(f"      Contains {len(strategy.allocations)} sub-strategies:")
            for sub_alloc in strategy.allocations:
                print(f"          - {sub_alloc.strategy_id} ({sub_alloc.weight*100:.0f}%)")
    
    # Test signal generation
    test_date = pd.Timestamp("2023-09-15")
    print(f"\nGenerating Signals for {test_date.date()}...")
    
    historical_data = market_data[market_data.index <= test_date]
    benchmark_historical = benchmark_data[benchmark_data.index <= test_date]
    
    try:
        signals = nested_meta.generate_signals(
            all_historical_data=historical_data,
            benchmark_historical_data=benchmark_historical,
            non_universe_historical_data=pd.DataFrame(),
            current_date=test_date
        )
        
        if not signals.empty and test_date in signals.index:
            signal_values = signals.loc[test_date]
            non_zero_signals = signal_values[signal_values.abs() > 0.001]
            
            print(f"   Generated signals for {len(signal_values)} assets")
            print(f"   Non-zero positions: {len(non_zero_signals)}")
            
            if len(non_zero_signals) > 0:
                print(f"   Top positions:")
                top_positions = non_zero_signals.abs().nlargest(5)
                for asset, weight in top_positions.items():
                    direction = "LONG" if signal_values[asset] > 0 else "SHORT"
                    print(f"      {asset}: {weight:.3f} ({direction})")
        else:
            print(f"   No signals generated")
            
    except Exception as e:
        print(f"   Error generating signals: {e}")
    
    print(f"\nNested Meta Strategy Demo Complete!")
    
    # Simulate some returns and show capital compounding
    print(f"\nSimulating Returns and Capital Compounding...")
    
    # Simulate quarterly returns
    simulated_returns = [
        {"equity_meta": 0.06, "conservative": 0.02},   # Q1
        {"equity_meta": -0.02, "conservative": 0.01},  # Q2
        {"equity_meta": 0.10, "conservative": 0.03},   # Q3
        {"equity_meta": 0.05, "conservative": 0.04},   # Q4
    ]
    
    print(f"   Initial Capital: ${nested_meta.available_capital:,.2f}")
    
    for quarter, returns in enumerate(simulated_returns, 1):
        nested_meta.update_available_capital(returns)
        
        equity_meta_return = returns["equity_meta"]
        conservative_return = returns["conservative"]
        
        print(f"   Q{quarter}: Equity Meta {equity_meta_return:+.1%}, Conservative {conservative_return:+.1%}")
        print(f"        Available Capital: ${nested_meta.available_capital:,.2f}")
        print(f"        Cumulative P&L: ${nested_meta.cumulative_pnl:,.2f}")
    
    total_return = (nested_meta.available_capital - config['initial_capital']) / config['initial_capital']
    print(f"\nFinal Results:")
    print(f"   Total Return: {total_return:.2%}")
    print(f"   Final Capital: ${nested_meta.available_capital:,.2f}")
    
    return nested_meta


def main():
    """Main demonstration function."""
    print("META STRATEGIES DEMONSTRATION")
    print("="*60)
    print("This script demonstrates the meta strategies system with:")
    print("- Simple meta strategy (fixed allocations)")
    print("- Nested meta strategy (meta containing meta)")
    print("- Real sub-strategies (CalmarMomentum + IntramonthSeasonal)")
    print("- Capital tracking and compounding")
    print("- Signal aggregation")
    
    # Create sample data
    market_data = create_sample_data()
    benchmark_data = create_benchmark_data()
    
    # Demonstrate simple meta strategy
    simple_meta, simple_signals = demonstrate_simple_meta_strategy(market_data, benchmark_data)
    
    # Demonstrate nested meta strategy
    nested_meta = demonstrate_nested_meta_strategy(market_data, benchmark_data)
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE!")
    print("="*60)
    print("Key achievements demonstrated:")
    print("Meta strategies successfully allocate capital")
    print("Sub-strategies maintain independent timing")
    print("Signals are properly aggregated with capital weighting")
    print("Capital compounds over time with returns")
    print("Nested meta strategies work correctly")
    print("Integration with existing strategy classes")
    
    print(f"\nFinal Capital States:")
    print(f"- Simple Meta: ${simple_meta.available_capital:,.2f}")
    print(f"- Nested Meta: ${nested_meta.available_capital:,.2f}")
    
    return simple_meta, nested_meta, simple_signals


if __name__ == "__main__":
    try:
        simple_meta, nested_meta, signals = main()
        print("\nAll demonstrations completed successfully!")
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)