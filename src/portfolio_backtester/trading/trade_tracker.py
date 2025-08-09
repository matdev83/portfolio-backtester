"""
TradeTracker facade system.

This module unifies various trading components into a cohesive interface
for managing trade tracking, statistics, and portfolio evaluation.
"""

from .trade_lifecycle_manager import TradeLifecycleManager, Trade
from .trade_statistics_calculator import TradeStatisticsCalculator
from .portfolio_value_tracker import PortfolioValueTracker
from .trade_table_formatter import TradeTableFormatter
from ..interfaces.commission_parameter_handler_interface import CommissionParameterHandlerFactory

import logging
import pandas as pd
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class TradeTracker:
    """
    Comprehensive trade tracking and analysis system.

    This facade coordinates the lifecycle management of trades,
    portfolio value tracking, and detailed statistics calculation
    for complete portfolio management.
    """

    def __init__(
        self,
        initial_portfolio_value: float = 100000.0,
        allocation_mode: str = "reinvestment",
    ):
        """
        Initialize the TradeTracker facade.

        Args:
            initial_portfolio_value: Starting capital amount
            allocation_mode: Capital allocation mode
        """
        self.trade_lifecycle_manager = TradeLifecycleManager()
        self.trade_statistics_calculator = TradeStatisticsCalculator()
        self.portfolio_value_tracker = PortfolioValueTracker(
            initial_portfolio_value, allocation_mode
        )
        self.table_formatter = TradeTableFormatter()

    @property
    def allocation_mode(self) -> str:
        """Get the allocation mode."""
        return self.portfolio_value_tracker.allocation_mode

    @property
    def initial_portfolio_value(self) -> float:
        """Get the initial portfolio value."""
        return self.portfolio_value_tracker.initial_portfolio_value

    @property
    def current_portfolio_value(self) -> float:
        """Get the current portfolio value."""
        return self.portfolio_value_tracker.current_portfolio_value

    def update_positions(
        self,
        date: pd.Timestamp,
        new_weights: pd.Series,
        prices: pd.Series,
        commissions,  # Union[Dict[str, float], float]
        detailed_commission_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update positions based on new target weights with dynamic capital tracking.

        Args:
            date: Current date
            new_weights: Target portfolio weights
            prices: Asset prices
            commissions: Commission costs per asset
            detailed_commission_info: Detailed commission information from unified calculator
        """
        # Handle commissions parameter using polymorphic interface
        commission_handler = CommissionParameterHandlerFactory.create_handler(commissions)
        commissions_dict = commission_handler.normalize_commissions(
            commissions, new_weights, self.trade_lifecycle_manager.get_open_positions()
        )

        # Calculate target quantities based on allocation mode
        base_capital = self.portfolio_value_tracker.get_base_capital_for_allocation()

        target_quantities = {}
        for ticker, weight in new_weights.items():
            price = prices.get(ticker)
            if price is not None and price > 0:
                target_value = weight * base_capital
                target_quantities[ticker] = target_value / price

        # Process position changes
        current_tickers = set(self.trade_lifecycle_manager.get_open_positions().keys())
        target_tickers = set(
            str(ticker) for ticker, qty in target_quantities.items() if abs(qty) > 1e-6
        )

        # Close positions not in target
        for ticker in current_tickers - target_tickers:
            if ticker in prices:
                commission = commissions_dict.get(ticker, 0.0)
                
                # Extract detailed commission info for this ticker if available
                ticker_commission_details = None
                if detailed_commission_info and ticker in detailed_commission_info:
                    ticker_commission_details = detailed_commission_info[ticker]
                
                closed_trade = self.trade_lifecycle_manager.close_position(
                    date, ticker, prices[ticker], commission, ticker_commission_details
                )
                if closed_trade:
                    self.portfolio_value_tracker.update_portfolio_value(closed_trade.pnl_net or 0.0)

        # Open new positions
        for ticker in target_tickers - current_tickers:
            if ticker in prices:
                commission = commissions_dict.get(str(ticker), 0.0)
                
                # Extract detailed commission info for this ticker if available
                ticker_commission_details = None
                if detailed_commission_info and ticker in detailed_commission_info:
                    ticker_commission_details = detailed_commission_info[ticker]
                
                self.trade_lifecycle_manager.open_position(
                    date, str(ticker), target_quantities[ticker], prices[ticker], 
                    commission, ticker_commission_details
                )

        # Update daily metrics
        self.portfolio_value_tracker.update_daily_metrics(
            date, target_quantities, prices, self.trade_lifecycle_manager.get_open_positions()
        )

    def update_mfe_mae(self, date: pd.Timestamp, prices: pd.Series) -> None:
        """Update Maximum Favorable/Adverse Excursion for open positions."""
        self.trade_lifecycle_manager.update_mfe_mae(date, prices)

    def get_trade_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive trade statistics."""
        completed_trades = self.trade_lifecycle_manager.get_completed_trades()
        portfolio_stats = self.portfolio_value_tracker.get_portfolio_level_stats()
        trade_stats = self.trade_statistics_calculator.calculate_statistics(
            completed_trades,
            self.portfolio_value_tracker.initial_portfolio_value,
            self.portfolio_value_tracker.allocation_mode,
        )
        trade_stats.update(portfolio_stats)
        return trade_stats

    def get_trade_statistics_table(self) -> pd.DataFrame:
        """Get trade statistics formatted as a table with All/Long/Short columns."""
        stats = self.get_trade_statistics()
        return self.table_formatter.format_statistics_table(stats)

    def get_directional_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get a summary of key metrics by direction for easy comparison."""
        stats = self.get_trade_statistics()
        return self.table_formatter.format_directional_summary(stats)

    def close_all_positions(
        self, date: pd.Timestamp, prices: pd.Series, commissions: Optional[Dict[str, float]] = None
    ) -> None:
        """Close all open positions at the end of backtesting."""
        closed_trades = self.trade_lifecycle_manager.close_all_positions(date, prices, commissions)
        for trade in closed_trades:
            self.portfolio_value_tracker.update_portfolio_value(trade.pnl_net or 0.0)

    def get_current_portfolio_value(self) -> float:
        """Get the current portfolio value after all trades."""
        return self.portfolio_value_tracker.get_current_portfolio_value()

    def get_total_return(self) -> float:
        """Get the total return as a percentage."""
        return self.portfolio_value_tracker.get_total_return()

    def get_capital_timeline(self) -> pd.DataFrame:
        """Get a timeline of portfolio value, cash balance, and position values."""
        return self.portfolio_value_tracker.get_capital_timeline()


# Export Trade class
__all__ = ["TradeTracker", "Trade"]
