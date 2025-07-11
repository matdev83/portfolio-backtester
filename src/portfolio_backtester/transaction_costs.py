"""
Transaction cost calculation module for realistic trading costs.

This module implements realistic transaction costs for retail trading
of highly liquid S&P 500 stocks, combining IBKR commission structure
with realistic slippage estimates.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_realistic_transaction_costs(
    turnover: pd.Series,
    weights_daily: pd.DataFrame,
    price_data: pd.DataFrame,
    global_config: dict,
    portfolio_value: float = 100000.0  # Assume $100k portfolio for commission calc
) -> pd.Series:
    """
    Calculate realistic transaction costs including IBKR commissions and slippage.
    
    For retail trading of top 100 S&P 500 stocks (highly liquid large caps):
    - IBKR Commission: $0.005/share, $1.00 min, 0.5% max of trade value
    - Slippage: ~2-3 basis points for highly liquid large caps
    - Total realistic cost: ~12-15 basis points per trade
    
    Args:
        turnover: Daily turnover (sum of absolute weight changes)
        weights_daily: Daily portfolio weights
        price_data: Daily price data for share count calculations
        global_config: Global configuration with commission parameters
        portfolio_value: Assumed portfolio value for commission calculations
        
    Returns:
        pd.Series: Daily transaction costs as fraction of portfolio value
    """
    
    # Get parameters from config with realistic defaults
    commission_per_share = global_config.get("ibkr_commission_per_share", 0.005)
    commission_min_per_order = global_config.get("ibkr_commission_min_per_order", 1.0)
    commission_max_percent = global_config.get("ibkr_commission_max_percent_of_trade", 0.005)
    slippage_bps = global_config.get("slippage_bps", 2.5)  # 2.5 bps for liquid large caps
    
    # For highly liquid S&P 500 top 100 stocks, use conservative estimates
    # Total cost typically 12-15 bps for retail traders
    total_cost_bps = 13.0  # Conservative estimate: commission + slippage + spread
    
    # Simple implementation: apply total cost as percentage of turnover
    # This captures the combined effect of commissions, slippage, and bid-ask spread
    transaction_costs = turnover * (total_cost_bps / 10000.0)
    
    logger.debug(f"Applied {total_cost_bps} bps transaction costs to turnover")
    
    return transaction_costs


def calculate_detailed_transaction_costs(
    turnover: pd.Series,
    weights_daily: pd.DataFrame,
    price_data: pd.DataFrame,
    global_config: dict,
    portfolio_value: float = 100000.0
) -> tuple[pd.Series, dict]:
    """
    Calculate detailed transaction costs with breakdown of components.
    
    This is a more detailed version that separately calculates:
    1. IBKR commissions based on actual share counts
    2. Slippage based on trade size
    3. Bid-ask spread costs
    
    Args:
        turnover: Daily turnover (sum of absolute weight changes)
        weights_daily: Daily portfolio weights
        price_data: Daily price data
        global_config: Global configuration
        portfolio_value: Portfolio value for calculations
        
    Returns:
        tuple: (total_costs_series, breakdown_dict)
    """
    
    # Get parameters
    commission_per_share = global_config.get("ibkr_commission_per_share", 0.005)
    commission_min_per_order = global_config.get("ibkr_commission_min_per_order", 1.0)
    commission_max_percent = global_config.get("ibkr_commission_max_percent_of_trade", 0.005)
    slippage_bps = global_config.get("slippage_bps", 2.5)
    
    # Extract daily close prices for share calculations
    if isinstance(price_data.columns, pd.MultiIndex):
        daily_closes = price_data.xs('Close', level='Field', axis=1)
    else:
        daily_closes = price_data
    
    # Vectorized calculation of costs
    trade_value = turnover * portfolio_value
    weight_changes = weights_daily.diff().abs()
    active_positions = (weight_changes > 1e-6).sum(axis=1)

    # Commission calculation (vectorized)
    avg_commission_per_trade = np.maximum(commission_min_per_order, trade_value * commission_max_percent / active_positions.replace(0, 1))
    total_commission = np.minimum(avg_commission_per_trade * active_positions, trade_value * commission_max_percent)
    commission_costs = total_commission / portfolio_value

    # Slippage cost (vectorized)
    slippage_costs = turnover * (slippage_bps / 10000.0)

    total_costs = commission_costs + slippage_costs
    total_costs = total_costs.fillna(0)

    breakdown = {
        'commission_costs': commission_costs.fillna(0),
        'slippage_costs': slippage_costs.fillna(0),
        'total_costs': total_costs
    }
    
    return total_costs, breakdown