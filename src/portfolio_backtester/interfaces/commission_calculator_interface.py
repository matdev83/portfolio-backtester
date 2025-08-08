"""
Commission calculation strategy interface.

This module defines the interface for commission calculation strategies.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Tuple, Callable

from portfolio_backtester.trading.trade_commission_info import TradeCommissionInfo


class ICommissionCalculationStrategy(ABC):
    """Interface for commission calculation strategies."""

    @abstractmethod
    def calculate_portfolio_commissions(
        self,
        turnover: pd.Series,
        weights_daily: pd.DataFrame,
        price_data: pd.DataFrame,
        portfolio_value: float,
        trade_commission_calculator: Callable[..., TradeCommissionInfo],
        transaction_costs_bps: float | None = None,
    ) -> Tuple[pd.Series, Dict[str, Any], Dict[pd.Timestamp, Dict[str, TradeCommissionInfo]]]:
        """Calculate commissions for portfolio-based strategies."""
        pass
