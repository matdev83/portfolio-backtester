"""
Trade commission information data structure.

This module provides the TradeCommissionInfo dataclass used across
all commission calculators for consistent trade cost information.
"""

import pandas as pd
from typing import Dict, Any
from dataclasses import dataclass


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
            "asset": self.asset,
            "date": self.date,
            "quantity": self.quantity,
            "price": self.price,
            "trade_value": self.trade_value,
            "commission_amount": self.commission_amount,
            "slippage_amount": self.slippage_amount,
            "total_cost": self.total_cost,
            "commission_rate_bps": self.commission_rate_bps,
            "slippage_rate_bps": self.slippage_rate_bps,
        }
