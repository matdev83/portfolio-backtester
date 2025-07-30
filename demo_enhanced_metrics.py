#!/usr/bin/env python3
"""
Demo script showing enhanced performance metrics in action.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from portfolio_backtester.trading.trade_tracker import TradeTracker
from portfolio_backtester.reporting.performance_metrics import calculate_metrics

def create_sample_backtest_data():
    """Create realistic sample backtest data."""
    print("Creating sample backtest data...")
    
    # Create 1 year of daily data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    # Create sample returns with some realistic patterns
    np.random.seed(42)
    base_returns = np.random.normal(0.0008, 0.015, len(dates))  # ~20% annual return, 15% vol
    
    # Add some momentum and mean reversion patterns
    for i in range(1, len(base_returns)):
        # Add some autocorrelation
        base_returns[i] += 0.1 * base_returns[i-1]
        
        # Add some volatility clustering
        if abs(base_returns[i-1]) > 0.02:
            base_returns[i] *= 1.5
    
    strategy_returns = pd.Series(base_returns, index=dates)
    
    # Create benchmark returns (slightly lower performance)
    benchmark_returns = strategy_returns * 0.8 + np.random.normal(0, 0.005, len(dates))
    
    return strategy_returns, benchmark_returns

def simulate_detailed_trading():
    """Simulate detailed trading activity with realistic patterns."""
    print("Simulating detailed trading activity...")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    
    # Initialize trade tracker
    tracker = TradeTracker(portfolio_value=1000000)  # $1M portfolio
    
    # Simulate monthly rebalancing with realistic trading patterns
    rebalance_dates = pd.date_range('2023-01-01', '2023-12-31', freq='MS')  # Month start
    
    for i, rebal_date in enumerate(rebalance_dates):
        if rebal_date not in dates:
            continue
            
        # Create realistic portfolio weights (sum to ~0.95 to account for cash)
        np.random.seed(42 + i)  # Reproducible but varying
        raw_weights = np.random.dirichlet([2, 2, 1.5, 1.5, 1], 1)[0]  # Favor first two stocks
        weights = raw_weights * 0.95  # Leave 5% cash
        
        weight_series = pd.Series(dict(zip(tickers, weights)))
        
        # Create realistic price movements
        base_prices = [150, 300, 2500, 3000, 400]  # Realistic starting prices
        price_changes = np.random.normal(0.02, 0.05, len(tickers))  # Monthly price changes
        current_prices = [base * (1 + change * (i + 1)) for base, change in zip(base_prices, price_changes)]
        
        price_series = pd.Series(dict(zip(tickers, current_prices)))
        
        # Calculate realistic transaction costs (0.1% of trade value)
        transaction_cost = 10.0  # $10 per ticker traded
        
        # Update positions
        tracker.update_positions(rebal_date, weight_series, price_series, transaction_cost)
        
        # Simulate daily price updates for MFE/MAE tracking
        month_end = min(rebal_date + pd.DateOffset(months=1), dates[-1])
        daily_dates = pd.date_range(rebal_date, month_end, freq='D')\n        \n        for daily_date in daily_dates[1:]:  # Skip first day (already processed)\n            if daily_date in dates:\n                # Create daily price variations\n                daily_changes = np.random.normal(0, 0.02, len(tickers))  # 2% daily vol\n                daily_prices = [price * (1 + change) for price, change in zip(current_prices, daily_changes)]\n                daily_price_series = pd.Series(dict(zip(tickers, daily_prices)))\n                \n                # Update MFE/MAE\n                tracker.update_mfe_mae(daily_date, daily_price_series)\n                \n                # Update current prices for next iteration\n                current_prices = daily_prices\n    \n    # Close all positions at the end\n    final_prices = pd.Series(dict(zip(tickers, current_prices)))\n    tracker.close_all_positions(dates[-1], final_prices)\n    \n    return tracker

def demonstrate_enhanced_metrics():
    \"\"\"Demonstrate the enhanced performance metrics.\"\"\"\n    print(\"\\n\" + \"=\"*80)\n    print(\"ENHANCED PERFORMANCE METRICS DEMONSTRATION\")\n    print(\"=\"*80)\n    \n    # Create sample data\n    strategy_returns, benchmark_returns = create_sample_backtest_data()\n    \n    # Simulate trading\n    trade_tracker = simulate_detailed_trading()\n    \n    # Get trade statistics\n    trade_stats = trade_tracker.get_trade_statistics()\n    \n    print(\"\\n📊 TRADE STATISTICS:\")\n    print(\"-\" * 40)\n    for key, value in trade_stats.items():\n        if isinstance(value, float):\n            if 'pct' in key.lower() or 'rate' in key.lower():\n                print(f\"{key:.<30} {value:.2f}%\")\n            elif 'days' in key.lower():\n                print(f\"{key:.<30} {value:.0f} days\")\n            else:\n                print(f\"{key:.<30} {value:.4f}\")\n        else:\n            print(f\"{key:.<30} {value}\")\n    \n    # Calculate enhanced performance metrics\n    enhanced_metrics = calculate_metrics(\n        strategy_returns,\n        benchmark_returns,\n        'SPY',\n        name='Enhanced Strategy',\n        trade_stats=trade_stats\n    )\n    \n    print(\"\\n📈 ENHANCED PERFORMANCE METRICS:\")\n    print(\"-\" * 50)\n    \n    # Group metrics for better presentation\n    return_metrics = ['Total Return', 'Ann. Return', 'Ann. Vol']\n    risk_metrics = ['Sharpe', 'Sortino', 'Calmar', 'Max Drawdown']\n    trade_metrics = ['Number of Trades', 'Win Rate (%)', 'Commissions Paid']\n    duration_metrics = ['Min Trade Duration (days)', 'Max Trade Duration (days)', 'Mean Trade Duration (days)']\n    advanced_metrics = ['Information Score', 'Avg MFE', 'Avg MAE']\n    margin_metrics = ['Max Margin Load', 'Mean Margin Load']\n    timing_metrics = ['Max DD Recovery Time (days)', 'Max Flat Period (days)', 'Trades per Month']\n    \n    def print_metric_group(title, metrics):\n        print(f\"\\n{title}:\")\n        for metric in metrics:\n            if metric in enhanced_metrics.index:\n                value = enhanced_metrics[metric]\n                if isinstance(value, float):\n                    if 'return' in metric.lower() or 'drawdown' in metric.lower():\n                        print(f\"  {metric:.<35} {value:.2%}\")\n                    elif 'ratio' in metric.lower() or 'score' in metric.lower():\n                        print(f\"  {metric:.<35} {value:.3f}\")\n                    elif 'days' in metric.lower():\n                        print(f\"  {metric:.<35} {value:.0f} days\")\n                    elif '%' in metric:\n                        print(f\"  {metric:.<35} {value:.2f}%\")\n                    else:\n                        print(f\"  {metric:.<35} {value:.4f}\")\n                else:\n                    print(f\"  {metric:.<35} {value}\")\n    \n    print_metric_group(\"📊 RETURN METRICS\", return_metrics)\n    print_metric_group(\"⚠️  RISK METRICS\", risk_metrics)\n    print_metric_group(\"🔄 TRADE METRICS\", trade_metrics)\n    print_metric_group(\"⏱️  DURATION METRICS\", duration_metrics)\n    print_metric_group(\"🎯 ADVANCED METRICS\", advanced_metrics)\n    print_metric_group(\"💰 MARGIN METRICS\", margin_metrics)\n    print_metric_group(\"📅 TIMING METRICS\", timing_metrics)\n    \n    # Show trade history sample\n    trade_history = trade_tracker.get_trade_history_dataframe()\n    if not trade_history.empty:\n        print(\"\\n📋 SAMPLE TRADE HISTORY:\")\n        print(\"-\" * 50)\n        print(trade_history.head(10).to_string(index=False))\n        \n        if len(trade_history) > 10:\n            print(f\"\\n... and {len(trade_history) - 10} more trades\")\n    \n    print(\"\\n\" + \"=\"*80)\n    print(\"✅ DEMONSTRATION COMPLETE!\")\n    print(\"\\nThe enhanced performance metrics provide comprehensive insights into:\")\n    print(\"• Trade execution efficiency (MFE/MAE)\")\n    print(\"• Portfolio utilization (margin load)\")\n    print(\"• Trading frequency and timing\")\n    print(\"• Risk-adjusted performance measures\")\n    print(\"• Detailed trade statistics\")\n    print(\"=\"*80)\n    \n    return enhanced_metrics, trade_stats, trade_history

if __name__ == \"__main__\":\n    try:\n        metrics, stats, history = demonstrate_enhanced_metrics()\n        print(\"\\n🎉 Demo completed successfully!\")\n    except Exception as e:\n        print(f\"\\n❌ Demo failed: {e}\")\n        import traceback\n        traceback.print_exc()\n        sys.exit(1)