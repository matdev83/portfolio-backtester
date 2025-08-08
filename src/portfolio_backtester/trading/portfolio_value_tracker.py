"""
Portfolio value tracking system.

This module handles portfolio value calculations, cash balance tracking,
margin usage monitoring, and capital timeline management.
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional
from .trade_lifecycle_manager import Trade
from ..interfaces.validator_interface import (
    IAllocationModeValidator,
    create_allocation_mode_validator,
)

logger = logging.getLogger(__name__)


class PortfolioValueTracker:
    """
    Tracks portfolio value, cash balance, and margin usage over time.

    This class handles:
    - Portfolio value updates based on trade P&L
    - Cash balance calculations
    - Margin usage tracking
    - Capital timeline management
    - Total return calculations
    """

    def __init__(
        self,
        initial_portfolio_value: float = 100000.0,
        allocation_mode: str = "reinvestment",
        allocation_validator: Optional[IAllocationModeValidator] = None,
    ):
        """
        Initialize the portfolio value tracker.

        Args:
            initial_portfolio_value: Starting capital amount
            allocation_mode: Capital allocation mode
            allocation_validator: Injected validator for allocation mode validation (DIP)
        """
        self.initial_portfolio_value = initial_portfolio_value
        self.current_portfolio_value = initial_portfolio_value
        self.allocation_mode = allocation_mode
        self.daily_margin_usage: pd.Series = pd.Series(dtype=float)
        self.daily_portfolio_value: pd.Series = pd.Series(dtype=float)
        self.daily_cash_balance: pd.Series = pd.Series(dtype=float)

        # Dependency injection for allocation mode validation (DIP)
        self._allocation_validator = allocation_validator or create_allocation_mode_validator()

        # Validate allocation mode using injected validator
        validation_result = self._allocation_validator.validate_allocation_mode(allocation_mode)
        if not validation_result.is_valid:
            raise ValueError(validation_result.message)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"PortfolioValueTracker initialized with allocation_mode='{allocation_mode}'"
            )

    def update_portfolio_value(self, trade_pnl: float) -> None:
        """
        Update portfolio value with trade P&L.

        Args:
            trade_pnl: Net profit/loss from a closed trade
        """
        self.current_portfolio_value += trade_pnl

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Portfolio value updated by ${trade_pnl:.2f}, new value: ${self.current_portfolio_value:.2f}"
            )

    def get_base_capital_for_allocation(self) -> float:
        """
        Get the base capital amount to use for position sizing.

        Returns:
            The capital amount based on allocation mode
        """
        if self.allocation_mode in ["reinvestment", "compound"]:
            # Use current portfolio value for compounding
            return self.current_portfolio_value
        else:  # fixed_fractional or fixed_capital
            # Use initial portfolio value (no compounding)
            return self.initial_portfolio_value

    def update_daily_metrics(
        self,
        date: pd.Timestamp,
        target_quantities: Dict[Any, Any],
        prices: pd.Series,
        open_positions: Dict[str, Trade],
    ) -> None:
        """
        Update daily margin usage, portfolio value, and cash balance.

        Args:
            date: Current date
            target_quantities: Target quantities for each asset
            prices: Current asset prices
            open_positions: Currently open positions
        """
        # Calculate total position value
        total_position_value = sum(
            abs(qty) * prices.get(ticker, 0)
            for ticker, qty in target_quantities.items()
            if abs(qty) > 1e-6
        )

        # Always use current portfolio value for margin calculation (actual account balance)
        self.daily_margin_usage[date] = (
            total_position_value / self.current_portfolio_value
            if self.current_portfolio_value > 0
            else 0.0
        )

        # Track daily portfolio value
        self.daily_portfolio_value[date] = self.current_portfolio_value

        # Calculate cash balance (portfolio value minus position values)
        current_position_value = sum(
            abs(trade.quantity) * prices.get(trade.ticker, 0)
            for trade in open_positions.values()
            if trade.ticker in prices
        )
        self.daily_cash_balance[date] = self.current_portfolio_value - current_position_value

    def get_current_portfolio_value(self) -> float:
        """Get the current portfolio value after all trades."""
        return self.current_portfolio_value

    def get_total_return(self) -> float:
        """Get the total return as a percentage."""
        if self.initial_portfolio_value == 0:
            return 0.0
        return (
            self.current_portfolio_value - self.initial_portfolio_value
        ) / self.initial_portfolio_value

    def get_capital_timeline(self) -> pd.DataFrame:
        """Get a timeline of portfolio value, cash balance, and position values."""
        if self.daily_portfolio_value.empty:
            return pd.DataFrame()

        timeline_df = pd.DataFrame(
            {
                "portfolio_value": self.daily_portfolio_value,
                "cash_balance": self.daily_cash_balance,
                "margin_usage": self.daily_margin_usage,
            }
        )

        # Calculate position value as portfolio_value - cash_balance
        timeline_df["position_value"] = timeline_df["portfolio_value"] - timeline_df["cash_balance"]

        return timeline_df

    def get_portfolio_level_stats(self) -> Dict[str, Any]:
        """
        Get portfolio-level statistics for inclusion in trade statistics.

        Returns:
            Dictionary containing portfolio-level metrics
        """
        max_margin_load = (
            self.daily_margin_usage.max() if not self.daily_margin_usage.empty else 0.0
        )
        mean_margin_load = (
            self.daily_margin_usage.mean() if not self.daily_margin_usage.empty else 0.0
        )

        return {
            "max_margin_load": max_margin_load,
            "mean_margin_load": mean_margin_load,
            "allocation_mode": self.allocation_mode,
            "initial_capital": self.initial_portfolio_value,
            "final_capital": self.current_portfolio_value,
            "total_return_pct": self.get_total_return() * 100,
            "capital_growth_factor": (
                self.current_portfolio_value / self.initial_portfolio_value
                if self.initial_portfolio_value > 0
                else 1.0
            ),
        }
