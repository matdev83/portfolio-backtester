"""
Detailed commission calculator for IBKR-style commission calculations.

This module provides detailed commission calculations using Interactive Brokers
style fee structure with per-share fees, minimum commissions, and maximum
percentage caps.
"""

import pandas as pd
from typing import Dict, Any

from .trade_commission_info import TradeCommissionInfo


class DetailedCommissionCalculator:
    """
    Calculator for detailed IBKR-style commission calculations.

    This class handles commission calculations with per-share fees,
    minimum commission thresholds, and maximum percentage caps,
    following Interactive Brokers fee structure.
    """

    def __init__(self, global_config: Dict[str, Any]):
        """
        Initialize the detailed commission calculator.

        Args:
            global_config: Global configuration containing commission parameters
        """
        # IBKR-style commission parameters
        self.commission_per_share = global_config.get("commission_per_share", 0.005)
        self.commission_min_per_order = global_config.get("commission_min_per_order", 1.0)
        self.commission_max_percent = global_config.get("commission_max_percent_of_trade", 0.005)

        # Slippage parameters
        self.slippage_bps = global_config.get("slippage_bps", 2.5)

    def calculate_commission(
        self, asset: str, date: pd.Timestamp, quantity: float, price: float, trade_value: float
    ) -> TradeCommissionInfo:
        """
        Calculate commission using detailed IBKR-style method.

        Args:
            asset: Asset symbol
            date: Trade date
            quantity: Number of shares (positive for buy, negative for sell)
            price: Price per share
            trade_value: Total trade value (abs(quantity) * price)

        Returns:
            TradeCommissionInfo with detailed cost breakdown
        """
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
            commission_rate_bps=(
                (commission_amount / trade_value * 10000.0) if trade_value > 0 else 0.0
            ),
            slippage_rate_bps=self.slippage_bps,
        )
