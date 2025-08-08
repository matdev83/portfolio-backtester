"""
Unified commission calculation system for the backtester framework.

This module provides a centralized commission calculation facility that works
for both portfolio-based and signal-based strategies, eliminating duplication
and ensuring consistent commission calculations across the framework.

This class now acts as a facade, delegating to specialized commission calculators
to improve maintainability and adhere to SOLID principles.
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional, Tuple

# Import and re-export for backward compatibility
from .trade_commission_info import TradeCommissionInfo
from .detailed_commission_calculator import DetailedCommissionCalculator
from .simplified_commission_calculator import SimplifiedCommissionCalculator
from .portfolio_commission_calculator import PortfolioCommissionCalculator

# Re-export TradeCommissionInfo for backward compatibility
__all__ = [
    "UnifiedCommissionCalculator",
    "TradeCommissionInfo",
    "get_unified_commission_calculator",
]

logger = logging.getLogger(__name__)


class UnifiedCommissionCalculator:
    """
    Unified commission calculator that provides consistent commission calculations
    across all strategy types in the backtester framework.

    This calculator acts as a facade, delegating to specialized calculators:
    - DetailedCommissionCalculator for IBKR-style calculations
    - SimplifiedCommissionCalculator for basis points calculations
    - PortfolioCommissionCalculator for portfolio-level calculations

    This design improves maintainability and adheres to SOLID principles while
    maintaining full backward compatibility.
    """

    def __init__(self, global_config: Dict[str, Any]):
        """
        Initialize the unified commission calculator.

        Args:
            global_config: Global configuration containing commission parameters
        """
        self.global_config = global_config

        # Initialize specialized calculators via composition
        self._detailed_calculator = DetailedCommissionCalculator(global_config)
        self._simplified_calculator = SimplifiedCommissionCalculator(global_config)
        self._portfolio_calculator = PortfolioCommissionCalculator(global_config)

        # Expose key configuration parameters for backward compatibility
        self.commission_per_share = global_config.get("commission_per_share", 0.005)
        self.commission_min_per_order = global_config.get("commission_min_per_order", 1.0)
        self.commission_max_percent = global_config.get("commission_max_percent_of_trade", 0.005)
        self.slippage_bps = global_config.get("slippage_bps", 2.5)
        self.default_transaction_cost_bps = global_config.get("default_transaction_cost_bps", 10.0)

        # Cache for commission calculations to improve performance
        self._commission_cache: Dict[str, TradeCommissionInfo] = {}

    def calculate_trade_commission(
        self,
        asset: str,
        date: pd.Timestamp,
        quantity: float,
        price: float,
        transaction_costs_bps: Optional[float] = None,
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
            return self._simplified_calculator.calculate_commission(
                asset, date, quantity, price, trade_value, transaction_costs_bps
            )

        # Use detailed IBKR-style calculation
        return self._detailed_calculator.calculate_commission(
            asset, date, quantity, price, trade_value
        )

    def calculate(
        self,
        turnover: pd.Series,
        weights_daily: pd.DataFrame,
        price_data: pd.DataFrame,
        portfolio_value: float = 100000.0,
        transaction_costs_bps: Optional[float] = None,
    ) -> Tuple[pd.Series, Dict[str, Any], Dict[pd.Timestamp, Dict[str, TradeCommissionInfo]]]:
        """
        Calculate commissions for portfolio-based strategies.

        Thin wrapper kept for backward compatibility. Delegates to
        calculate_portfolio_commissions to keep this method simple for
        radon/xenon complexity grading.
        """
        return self.calculate_portfolio_commissions(
            turnover=turnover,
            weights_daily=weights_daily,
            price_data=price_data,
            portfolio_value=portfolio_value,
            transaction_costs_bps=transaction_costs_bps,
        )

    # Backward-compatible API expected by tests
    def calculate_portfolio_commissions(
        self,
        turnover: pd.Series,
        weights_daily: pd.DataFrame,
        price_data: pd.DataFrame,
        portfolio_value: float = 100000.0,
        transaction_costs_bps: Optional[float] = None,
    ) -> Tuple[pd.Series, Dict[str, Any], Dict[pd.Timestamp, Dict[str, TradeCommissionInfo]]]:
        """
        Calculate commissions for portfolio-based strategies.

        This method maintains compatibility with existing tests expecting
        calculate_portfolio_commissions to be available. Now delegates to
        the specialized PortfolioCommissionCalculator.
        """
        return self._portfolio_calculator.calculate_portfolio_commissions(
            turnover=turnover,
            weights_daily=weights_daily,
            price_data=price_data,
            portfolio_value=portfolio_value,
            trade_commission_calculator=self.calculate_trade_commission,
            transaction_costs_bps=transaction_costs_bps,
        )

    def get_commission_summary(
        self, detailed_trade_info: Dict[pd.Timestamp, Dict[str, TradeCommissionInfo]]
    ) -> Dict[str, Any]:
        """
        Generate a summary of commission costs from detailed trade information.

        Args:
            detailed_trade_info: Detailed trade commission information

        Returns:
            Dictionary with commission summary statistics
        """
        return self._portfolio_calculator.get_commission_summary(detailed_trade_info)


def get_unified_commission_calculator(config: Dict[str, Any]) -> UnifiedCommissionCalculator:
    """
    Factory function to get the unified commission calculator.

    Args:
        config: Global configuration dictionary

    Returns:
        UnifiedCommissionCalculator instance
    """
    return UnifiedCommissionCalculator(config)
