"""
Simplified commission calculator for basis points commission calculations.

This module provides simplified commission calculations using a flat
basis points model without additional slippage or complex fee structures.
"""

import pandas as pd
from typing import Dict, Any

from .trade_commission_info import TradeCommissionInfo


class SimplifiedCommissionCalculator:
    """
    Calculator for simplified basis points commission calculations.

    This class handles commission calculations using a simple basis points
    model where commission is a flat percentage of trade value without
    additional slippage costs.
    """

    def __init__(self, global_config: Dict[str, Any]):
        """
        Initialize the simplified commission calculator.

        Args:
            global_config: Global configuration containing commission parameters
        """
        # Default transaction cost for simplified calculations
        self.default_transaction_cost_bps = global_config.get("default_transaction_cost_bps", 10.0)

    def calculate_commission(
        self,
        asset: str,
        date: pd.Timestamp,
        quantity: float,
        price: float,
        trade_value: float,
        transaction_costs_bps: float,
    ) -> TradeCommissionInfo:
        """
        Calculate commission using simple basis points model.

        Args:
            asset: Asset symbol
            date: Trade date
            quantity: Number of shares (positive for buy, negative for sell)
            price: Price per share
            trade_value: Total trade value (abs(quantity) * price)
            transaction_costs_bps: Transaction costs in basis points

        Returns:
            TradeCommissionInfo with cost breakdown
        """
        # Commission is a flat percentage of trade value
        commission_amount = trade_value * (transaction_costs_bps / 10000.0)
        # No additional slippage when simplified cost model is requested
        slippage_amount = 0.0
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
            commission_rate_bps=transaction_costs_bps,
            slippage_rate_bps=0.0,
        )
