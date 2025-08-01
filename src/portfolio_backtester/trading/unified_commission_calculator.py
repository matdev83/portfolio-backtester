"""
Unified commission calculation system for the backtester framework.

This module provides a centralized commission calculation facility that works
for both portfolio-based and signal-based strategies, eliminating duplication
and ensuring consistent commission calculations across the framework.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class TradeCommissionInfo:
    """Information about commission costs for a specific trade."""
    asset: str
    date: pd.Timestamp
    quantity: float
    price: float
    trade_value: float
    commission_amount: float
    slippage_amount: float
    total_cost: float
    commission_rate_bps: float
    slippage_rate_bps: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization."""
        return {
            'asset': self.asset,
            'date': self.date,
            'quantity': self.quantity,
            'price': self.price,
            'trade_value': self.trade_value,
            'commission_amount': self.commission_amount,
            'slippage_amount': self.slippage_amount,
            'total_cost': self.total_cost,
            'commission_rate_bps': self.commission_rate_bps,
            'slippage_rate_bps': self.slippage_rate_bps
        }


class UnifiedCommissionCalculator:
    """
    Unified commission calculator that provides consistent commission calculations
    across all strategy types in the backtester framework.
    
    This calculator supports both the detailed IBKR-style commission structure
    and simplified basis points calculations, ensuring consistency while
    providing flexibility for different use cases.
    """
    
    def __init__(self, global_config: Dict[str, Any]):
        """
        Initialize the unified commission calculator.
        
        Args:
            global_config: Global configuration containing commission parameters
        """
        self.global_config = global_config
        
        # IBKR-style commission parameters
        self.commission_per_share = global_config.get("commission_per_share", 0.005)
        self.commission_min_per_order = global_config.get("commission_min_per_order", 1.0)
        self.commission_max_percent = global_config.get("commission_max_percent_of_trade", 0.005)
        
        # Slippage parameters
        self.slippage_bps = global_config.get("slippage_bps", 2.5)
        
        # Default transaction cost for simplified calculations
        self.default_transaction_cost_bps = global_config.get("default_transaction_cost_bps", 10.0)
        
        # Cache for commission calculations to improve performance
        self._commission_cache: Dict[str, TradeCommissionInfo] = {}
        
    def calculate_trade_commission(
        self,
        asset: str,
        date: pd.Timestamp,
        quantity: float,
        price: float,
        transaction_costs_bps: Optional[float] = None
    ) -> TradeCommissionInfo:
        """
        Calculate commission and slippage costs for a single trade.
        
        Args:
            asset: Asset symbol
            date: Trade date
            quantity: Number of shares (positive for buy, negative for sell)
            price: Price per share
            transaction_costs_bps: Override transaction costs in basis points
            
        Returns:
            TradeCommissionInfo with detailed cost breakdown
        """
        trade_value = abs(quantity) * price
        
        # Use simplified calculation if transaction_costs_bps is provided
        if transaction_costs_bps is not None:
            return self._calculate_simplified_commission(
                asset, date, quantity, price, trade_value, transaction_costs_bps
            )
        
        # Use detailed IBKR-style calculation
        return self._calculate_detailed_commission(
            asset, date, quantity, price, trade_value
        )
    
    
    
    def _calculate_detailed_commission(
        self,
        asset: str,
        date: pd.Timestamp,
        quantity: float,
        price: float,
        trade_value: float
    ) -> TradeCommissionInfo:
        """Calculate commission using detailed IBKR-style method."""
        shares_traded = abs(quantity)
        
        # Calculate commission per share
        commission_per_trade = shares_traded * self.commission_per_share
        
        # Apply minimum commission for non-zero trades
        if shares_traded > 0:
            commission_per_trade = max(commission_per_trade, self.commission_min_per_order)
        
        # Apply maximum commission cap
        max_commission = trade_value * self.commission_max_percent
        commission_amount = min(commission_per_trade, max_commission)
        
        # Calculate slippage
        slippage_amount = trade_value * (self.slippage_bps / 10000.0)
        
        # Total cost
        total_cost = commission_amount + slippage_amount
        
        return TradeCommissionInfo(
            asset=asset,
            date=date,
            quantity=quantity,
            price=price,
            trade_value=trade_value,
            commission_amount=commission_amount,
            slippage_amount=slippage_amount,
            total_cost=total_cost,
            commission_rate_bps=(commission_amount / trade_value * 10000.0) if trade_value > 0 else 0.0,
            slippage_rate_bps=self.slippage_bps
        )
    
    def calculate_portfolio_commissions(
        self,
        turnover: pd.Series,
        weights_daily: pd.DataFrame,
        price_data: pd.DataFrame,
        portfolio_value: float = 100000.0,
        transaction_costs_bps: Optional[float] = None
    ) -> Tuple[pd.Series, Dict[str, Any], Dict[pd.Timestamp, Dict[str, TradeCommissionInfo]]]:
        """
        Calculate commissions for portfolio-based strategies.
        
        This method maintains compatibility with the existing portfolio logic
        while providing detailed per-trade commission information.
        
        Args:
            turnover: Portfolio turnover series
            weights_daily: Daily portfolio weights
            price_data: Price data DataFrame
            portfolio_value: Total portfolio value
            transaction_costs_bps: Override transaction costs in basis points
            
        Returns:
            Tuple of (total_costs_series, breakdown_dict, detailed_trade_info)
        """
        # Extract close prices
        if isinstance(price_data.columns, pd.MultiIndex):
            daily_closes = price_data.xs('Close', level='Field', axis=1)
        else:
            daily_closes = price_data
        
        # Calculate weight changes
        weight_changes = weights_daily.diff().abs().fillna(0)
        
        # Align columns
        aligned_weights, aligned_closes = weight_changes.align(
            daily_closes, join='left', axis=1, fill_value=0.0
        )
        
        # Initialize results
        total_costs = pd.Series(0.0, index=turnover.index)
        commission_costs = pd.Series(0.0, index=turnover.index)
        slippage_costs = pd.Series(0.0, index=turnover.index)
        detailed_trade_info = {}
        
        # Calculate costs for each date
        for date in turnover.index:
            if date not in aligned_weights.index or date not in aligned_closes.index:
                continue
                
            date_weights = aligned_weights.loc[date]
            date_prices = aligned_closes.loc[date]
            date_trade_info = {}
            
            date_commission_total = 0.0
            date_slippage_total = 0.0
            
            # Calculate costs for each asset with non-zero weight change
            for asset in date_weights.index:
                weight_change = date_weights[asset]
                if abs(weight_change) < 1e-8:
                    continue
                    
                price = date_prices[asset] if asset in date_prices.index else 0.0
                if price <= 0 or pd.isna(price):
                    continue
                
                # Calculate trade details
                trade_value = weight_change * portfolio_value
                quantity = trade_value / price
                
                # Calculate commission for this trade
                commission_info = self.calculate_trade_commission(
                    asset=asset,
                    date=date,
                    quantity=quantity,
                    price=price,
                    transaction_costs_bps=transaction_costs_bps
                )
                
                date_trade_info[asset] = commission_info
                date_commission_total += commission_info.commission_amount
                
                # For detailed calculation, add slippage per trade
                if transaction_costs_bps is None:
                    date_slippage_total += commission_info.slippage_amount
            
            # For detailed calculation, if no individual trades but turnover exists, apply slippage
            if transaction_costs_bps is None:
                turnover_value = turnover.loc[date] if date in turnover.index else 0.0
                # If turnover_value is a Series (e.g., after erroneous multi-column aggregation), reduce to scalar
                if isinstance(turnover_value, (pd.Series, pd.DataFrame)):
                    turnover_value = float(turnover_value.sum())
                if turnover_value > 0 and date_slippage_total == 0:
                    # Apply slippage to total turnover when no individual trades calculated
                    date_slippage_total = portfolio_value * turnover_value * (self.slippage_bps / 10000.0)
            
            # Store results
            detailed_trade_info[date] = date_trade_info
            commission_costs.loc[date] = date_commission_total / portfolio_value
            slippage_costs.loc[date] = date_slippage_total / portfolio_value
            total_costs.loc[date] = commission_costs.loc[date] + slippage_costs.loc[date]
        
        # Create breakdown dictionary for backward compatibility
        breakdown = {
            'commission_costs': commission_costs.fillna(0),
            'slippage_costs': slippage_costs.fillna(0),
            'total_costs': total_costs.fillna(0)
        }
        
        return total_costs.fillna(0), breakdown, detailed_trade_info
    
    def get_commission_summary(
        self, 
        detailed_trade_info: Dict[pd.Timestamp, Dict[str, TradeCommissionInfo]]
    ) -> Dict[str, Any]:
        """
        Generate a summary of commission costs from detailed trade information.
        
        Args:
            detailed_trade_info: Detailed trade commission information
            
        Returns:
            Dictionary with commission summary statistics
        """
        if not detailed_trade_info:
            return {
                'total_trades': 0,
                'total_commission': 0.0,
                'total_slippage': 0.0,
                'total_costs': 0.0,
                'avg_commission_per_trade': 0.0,
                'avg_slippage_per_trade': 0.0,
                'avg_cost_per_trade': 0.0,
                'commission_rate_bps_avg': 0.0,
                'slippage_rate_bps_avg': 0.0
            }
        
        all_trades = []
        for date_trades in detailed_trade_info.values():
            all_trades.extend(date_trades.values())
        
        if not all_trades:
            return {
                'total_trades': 0,
                'total_commission': 0.0,
                'total_slippage': 0.0,
                'total_costs': 0.0,
                'avg_commission_per_trade': 0.0,
                'avg_slippage_per_trade': 0.0,
                'avg_cost_per_trade': 0.0,
                'commission_rate_bps_avg': 0.0,
                'slippage_rate_bps_avg': 0.0
            }
        
        total_commission = sum(trade.commission_amount for trade in all_trades)
        total_slippage = sum(trade.slippage_amount for trade in all_trades)
        total_costs = sum(trade.total_cost for trade in all_trades)
        
        return {
            'total_trades': len(all_trades),
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'total_costs': total_costs,
            'avg_commission_per_trade': total_commission / len(all_trades),
            'avg_slippage_per_trade': total_slippage / len(all_trades),
            'avg_cost_per_trade': total_costs / len(all_trades),
            'commission_rate_bps_avg': np.mean([trade.commission_rate_bps for trade in all_trades]),
            'slippage_rate_bps_avg': np.mean([trade.slippage_rate_bps for trade in all_trades])
        }


def get_unified_commission_calculator(config: Dict[str, Any]) -> UnifiedCommissionCalculator:
    """
    Factory function to get the unified commission calculator.
    
    Args:
        config: Global configuration dictionary
        
    Returns:
        UnifiedCommissionCalculator instance
    """
    return UnifiedCommissionCalculator(config)