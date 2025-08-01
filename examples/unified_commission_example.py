"""
Example demonstrating the unified commission calculation system.

This example shows how the new unified commission calculator provides
consistent commission calculations across both portfolio-based and 
signal-based strategies, with detailed per-trade information.
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.portfolio_backtester.trading.unified_commission_calculator import (
    get_unified_commission_calculator,
    TradeCommissionInfo
)
from src.portfolio_backtester.trading.transaction_costs import RealisticTransactionCostModel


def main():
    print("=== Unified Commission Calculation Example ===\n")
    
    # Configuration for commission calculation
    config = {
        'commission_per_share': 0.005,      # $0.005 per share
        'commission_min_per_order': 1.0,    # $1.00 minimum per order
        'commission_max_percent_of_trade': 0.005,  # 0.5% max of trade value
        'slippage_bps': 2.5,                # 2.5 basis points slippage
        'default_transaction_cost_bps': 10.0 # 10 bps for simplified calculations
    }
    
    # Create unified commission calculator
    calculator = get_unified_commission_calculator(config)
    
    print("1. Single Trade Commission Calculation")
    print("=" * 50)
    
    # Example 1: Calculate commission for a single trade
    trade_info = calculator.calculate_trade_commission(
        asset="AAPL",
        date=pd.Timestamp("2023-01-01"),
        quantity=100.0,
        price=150.0,
        transaction_costs_bps=None  # Use detailed IBKR-style calculation
    )
    
    print(f"Trade: Buy 100 shares of AAPL at $150.00")
    print(f"Trade Value: ${trade_info.trade_value:,.2f}")
    print(f"Commission: ${trade_info.commission_amount:.2f}")
    print(f"Slippage: ${trade_info.slippage_amount:.2f}")
    print(f"Total Cost: ${trade_info.total_cost:.2f}")
    print(f"Commission Rate: {trade_info.commission_rate_bps:.2f} bps")
    print(f"Slippage Rate: {trade_info.slippage_rate_bps:.2f} bps")
    print()
    
    print("2. Simplified Commission Calculation (Basis Points)")
    print("=" * 50)
    
    # Example 2: Simplified calculation using basis points
    trade_info_simple = calculator.calculate_trade_commission(
        asset="MSFT",
        date=pd.Timestamp("2023-01-02"),
        quantity=50.0,
        price=250.0,
        transaction_costs_bps=15.0  # 15 basis points total
    )
    
    print(f"Trade: Buy 50 shares of MSFT at $250.00")
    print(f"Trade Value: ${trade_info_simple.trade_value:,.2f}")
    print(f"Commission: ${trade_info_simple.commission_amount:.2f}")
    print(f"Slippage: ${trade_info_simple.slippage_amount:.2f}")
    print(f"Total Cost: ${trade_info_simple.total_cost:.2f}")
    print(f"Total Rate: {trade_info_simple.commission_rate_bps + trade_info_simple.slippage_rate_bps:.1f} bps")
    print()
    
    print("3. Portfolio-Level Commission Calculation")
    print("=" * 50)
    
    # Example 3: Portfolio-level calculation (as used by portfolio strategies)
    dates = pd.date_range("2023-01-01", periods=3, freq="D")
    assets = ["AAPL", "MSFT", "GOOGL"]
    
    # Portfolio turnover
    turnover = pd.Series([0.0, 0.15, 0.08], index=dates)
    
    # Daily portfolio weights
    weights_daily = pd.DataFrame({
        "AAPL": [0.4, 0.5, 0.3],
        "MSFT": [0.3, 0.3, 0.4],
        "GOOGL": [0.3, 0.2, 0.3]
    }, index=dates)
    
    # Price data
    price_data = pd.DataFrame({
        "AAPL": [150.0, 152.0, 148.0],
        "MSFT": [250.0, 248.0, 252.0],
        "GOOGL": [2800.0, 2820.0, 2790.0]
    }, index=dates)
    
    # Calculate portfolio commissions
    total_costs, breakdown, detailed_info = calculator.calculate_portfolio_commissions(
        turnover=turnover,
        weights_daily=weights_daily,
        price_data=price_data,
        portfolio_value=1000000.0,  # $1M portfolio
        transaction_costs_bps=None  # Use detailed calculation
    )
    
    print("Portfolio Commission Results:")
    print(f"Total Costs: {total_costs}")
    print(f"Commission Costs: {breakdown['commission_costs']}")
    print(f"Slippage Costs: {breakdown['slippage_costs']}")
    print()
    
    # Show detailed trade information for one day
    if dates[1] in detailed_info and detailed_info[dates[1]]:
        print(f"Detailed Trade Info for {dates[1].date()}:")
        for asset, info in detailed_info[dates[1]].items():
            print(f"  {asset}: Commission=${info.commission_amount:.2f}, "
                  f"Slippage=${info.slippage_amount:.2f}, "
                  f"Total=${info.total_cost:.2f}")
        print()
    
    print("4. Commission Summary Statistics")
    print("=" * 50)
    
    # Generate summary statistics
    summary = calculator.get_commission_summary(detailed_info)
    
    print(f"Total Trades: {summary['total_trades']}")
    print(f"Total Commission: ${summary['total_commission']:.2f}")
    print(f"Total Slippage: ${summary['total_slippage']:.2f}")
    print(f"Total Costs: ${summary['total_costs']:.2f}")
    print(f"Average Commission per Trade: ${summary['avg_commission_per_trade']:.2f}")
    print(f"Average Slippage per Trade: ${summary['avg_slippage_per_trade']:.2f}")
    print(f"Average Cost per Trade: ${summary['avg_cost_per_trade']:.2f}")
    print()
    
    print("5. Integration with Existing Transaction Cost Model")
    print("=" * 50)
    
    # Example 4: Show how the existing RealisticTransactionCostModel now uses
    # the unified calculator internally
    tx_model = RealisticTransactionCostModel(config)
    
    # This now uses the unified calculator internally
    model_costs, model_breakdown = tx_model.calculate(
        turnover=turnover,
        weights_daily=weights_daily,
        price_data=price_data,
        portfolio_value=1000000.0
    )
    
    print("Transaction Cost Model Results (using unified calculator):")
    print(f"Total Costs: {model_costs}")
    print(f"Breakdown: {model_breakdown}")
    print()
    
    # Access detailed trade information
    detailed_trade_info = tx_model.get_last_detailed_trade_info()
    print(f"Detailed trade info available for {len(detailed_trade_info)} dates")
    
    print("6. Benefits of the Unified System")
    print("=" * 50)
    print("✓ Consistent commission calculations across all strategy types")
    print("✓ Detailed per-trade commission information available")
    print("✓ Support for both simplified (bps) and detailed (IBKR-style) calculations")
    print("✓ Backward compatibility with existing code")
    print("✓ Centralized configuration and maintenance")
    print("✓ Enhanced reporting and analysis capabilities")


if __name__ == "__main__":
    main()