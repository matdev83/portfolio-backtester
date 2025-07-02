#!/usr/bin/env python3
"""Demo script showing how to use the new Calmar Momentum Strategy."""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from portfolio_backtester.strategies.calmar_momentum_strategy import CalmarMomentumStrategy

def demo_calmar_strategy():
    """Demonstrate the Calmar Momentum Strategy with sample data."""
    
    print("Calmar Momentum Strategy Demo")
    print("=" * 50)
    
    # Create sample data
    print("Creating sample data...")
    dates = pd.date_range('2020-01-01', periods=24, freq='ME')
    
    # Create 5 stocks with different risk/return profiles
    np.random.seed(42)
    
    # Stock A: High return, moderate volatility, low drawdowns (good Calmar)
    stock_a_rets = np.random.normal(0.02, 0.03, len(dates))
    stock_a_rets[5:7] = [-0.05, -0.02]  # Small drawdown
    
    # Stock B: High return, high volatility, large drawdowns (poor Calmar)
    stock_b_rets = np.random.normal(0.025, 0.08, len(dates))
    stock_b_rets[8:12] = [-0.15, -0.10, -0.08, -0.05]  # Large drawdown
    
    # Stock C: Moderate return, low volatility, minimal drawdowns (excellent Calmar)
    stock_c_rets = np.random.normal(0.015, 0.02, len(dates))
    
    # Stock D: Low return, high volatility (poor Calmar)
    stock_d_rets = np.random.normal(0.005, 0.06, len(dates))
    
    # Stock E: Declining stock (negative Calmar)
    stock_e_rets = np.random.normal(-0.01, 0.04, len(dates))
    
    # Convert to prices
    data = pd.DataFrame({
        'STOCK_A': 100 * (1 + stock_a_rets).cumprod(),
        'STOCK_B': 100 * (1 + stock_b_rets).cumprod(),
        'STOCK_C': 100 * (1 + stock_c_rets).cumprod(),
        'STOCK_D': 100 * (1 + stock_d_rets).cumprod(),
        'STOCK_E': 100 * (1 + stock_e_rets).cumprod(),
    }, index=dates)
    
    # Create benchmark
    benchmark_rets = np.random.normal(0.01, 0.04, len(dates))
    benchmark_data = pd.Series(100 * (1 + benchmark_rets).cumprod(), 
                              index=dates, name='BENCHMARK')
    
    print(f"   Created data for {len(data.columns)} stocks over {len(data)} months")
    print(f"   Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    
    # Configure strategy
    print("\nConfiguring Calmar Momentum Strategy...")
    strategy_config = {
        'rolling_window': 6,           # 6-month rolling window
        'top_decile_fraction': 0.4,    # Top 40% (2 out of 5 stocks)
        'smoothing_lambda': 0.3,       # 30% smoothing (70% new signals)
        'leverage': 1.0,               # 100% leverage
        'long_only': True,             # Long-only strategy
        'sma_filter_window': None      # No SMA filter
    }
    
    for key, value in strategy_config.items():
        print(f"   {key}: {value}")
    
    # Create strategy
    strategy = CalmarMomentumStrategy(strategy_config)
    
    # Calculate returns for Calmar analysis
    print("\n📈 Calculating Calmar ratios...")
    returns = data.pct_change(fill_method=None)
    rolling_calmar = strategy._calculate_rolling_calmar(returns, strategy_config['rolling_window'])
    
    # Show final Calmar ratios
    final_calmar = rolling_calmar.iloc[-1]
    print("   Final Calmar Ratios:")
    for stock, calmar in final_calmar.items():
        print(f"     {stock}: {calmar:.4f}")
    
    # Generate trading signals
    print("\n📊 Generating trading signals...")
    weights = strategy.generate_signals(data, benchmark_data)
    
    # Show final weights
    final_weights = weights.iloc[-1]
    print("   Final Portfolio Weights:")
    for stock, weight in final_weights.items():
        if weight > 0:
            print(f"     {stock}: {weight:.1%}")
    
    # Show some statistics
    print("\nStrategy Statistics:")
    print(f"   Total periods: {len(weights)}")
    print(f"   Periods with positions: {(weights.sum(axis=1) > 0).sum()}")
    print(f"   Average number of holdings: {(weights > 0).sum(axis=1).mean():.1f}")
    print(f"   Maximum leverage used: {weights.sum(axis=1).max():.1%}")
    
    # Calculate simple performance metrics
    portfolio_returns = (weights.shift(1) * returns).sum(axis=1).dropna()
    benchmark_returns = benchmark_data.pct_change().dropna()
    
    print("\nPerformance Summary:")
    print(f"   Portfolio Total Return: {(1 + portfolio_returns).prod() - 1:.1%}")
    print(f"   Benchmark Total Return: {(1 + benchmark_returns).prod() - 1:.1%}")
    print(f"   Portfolio Volatility: {portfolio_returns.std() * np.sqrt(12):.1%}")
    print(f"   Benchmark Volatility: {benchmark_returns.std() * np.sqrt(12):.1%}")
    
    # Calculate portfolio Calmar ratio
    portfolio_cumret = (1 + portfolio_returns).cumprod()
    portfolio_peak = portfolio_cumret.expanding().max()
    portfolio_dd = (portfolio_cumret / portfolio_peak - 1).min()
    portfolio_calmar = (portfolio_returns.mean() * 12) / abs(portfolio_dd) if portfolio_dd != 0 else np.inf
    
    print(f"   Portfolio Calmar Ratio: {portfolio_calmar:.2f}")
    
    print("\n✅ Demo completed successfully!")
    print("\nTo run this strategy in the backtester, use:")
    print('   python src/portfolio_backtester/backtester.py --portfolios "Calmar_Momentum"')

if __name__ == '__main__':
    try:
        demo_calmar_strategy()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)