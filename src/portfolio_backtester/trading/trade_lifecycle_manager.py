"""
Trade lifecycle management system.

This module handles the opening, closing, and management of trade positions,
including MFE/MAE tracking and position updates.
"""

import logging
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade with all relevant information."""

    ticker: str
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    quantity: float  # Positive for long, negative for short
    entry_value: float  # Absolute value of position
    commission_entry: float
    commission_exit: float
    mfe: float = 0.0  # Maximum Favorable Excursion
    mae: float = 0.0  # Maximum Adverse Excursion
    duration_days: Optional[int] = None
    pnl_gross: Optional[float] = None
    pnl_net: Optional[float] = None
    is_winner: Optional[bool] = None
    # Enhanced commission tracking
    detailed_commission_entry: Optional[Dict[str, Any]] = None
    detailed_commission_exit: Optional[Dict[str, Any]] = None

    def finalize(self):
        """Calculate derived fields when trade is closed."""
        if self.exit_date is not None and self.exit_price is not None:
            # Calculate business day duration instead of calendar days
            # This is more accurate for trading strategies that use business day logic
            try:
                # Use pandas business day range to get actual business days
                business_days = pd.bdate_range(start=self.entry_date, end=self.exit_date)
                # Subtract 1 because bdate_range includes both start and end dates
                # but we want the number of days held, not the number of dates
                self.duration_days = max(0, len(business_days) - 1)
            except Exception:
                # Fallback to calendar days if business day calculation fails
                self.duration_days = (self.exit_date - self.entry_date).days

            # Calculate P&L
            if self.quantity > 0:  # Long position
                self.pnl_gross = (self.exit_price - self.entry_price) * abs(self.quantity)
            else:  # Short position
                self.pnl_gross = (self.entry_price - self.exit_price) * abs(self.quantity)

            self.pnl_net = self.pnl_gross - self.commission_entry - self.commission_exit
            self.is_winner = self.pnl_net > 0


class TradeLifecycleManager:
    """
    Manages the lifecycle of trades from opening to closing.

    This class handles:
    - Opening new positions
    - Closing existing positions
    - Updating MFE/MAE for open positions
    - Managing the collection of completed and open trades
    """

    def __init__(self):
        """Initialize the trade lifecycle manager."""
        self.trades: List[Trade] = []
        self.open_positions: Dict[str, Trade] = {}

    def open_position(
        self,
        date: pd.Timestamp,
        ticker: str,
        quantity: float,
        price: float,
        commission: float,
        detailed_commission_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Open a new position.

        Args:
            date: Entry date
            ticker: Asset ticker symbol
            quantity: Position size (positive for long, negative for short)
            price: Entry price
            commission: Entry commission cost
            detailed_commission_info: Optional detailed commission breakdown
        """
        entry_value = abs(quantity) * price

        trade = Trade(
            ticker=ticker,
            entry_date=date,
            exit_date=None,
            entry_price=price,
            exit_price=None,
            quantity=quantity,
            entry_value=entry_value,
            commission_entry=commission,
            commission_exit=0.0,
            detailed_commission_entry=detailed_commission_info,
        )

        self.open_positions[ticker] = trade

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Opened position: {ticker} {quantity:.2f} shares at ${price:.2f}")

    def close_position(
        self,
        date: pd.Timestamp,
        ticker: str,
        price: float,
        commission: float,
        detailed_commission_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[Trade]:
        """
        Close an existing position.

        Args:
            date: Exit date
            ticker: Asset ticker symbol
            price: Exit price
            commission: Exit commission cost
            detailed_commission_info: Optional detailed commission breakdown

        Returns:
            The closed trade if it existed, None otherwise
        """
        if ticker not in self.open_positions:
            return None

        trade = self.open_positions[ticker]
        trade.exit_date = date
        trade.exit_price = price
        trade.commission_exit = commission
        trade.detailed_commission_exit = detailed_commission_info

        # Finalize MFE/MAE in dollar terms
        trade.mfe = trade.mfe * abs(trade.quantity)
        trade.mae = trade.mae * abs(trade.quantity)

        trade.finalize()

        # Move to completed trades
        self.trades.append(trade)
        del self.open_positions[ticker]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Closed position: {ticker} on {date}, P&L: ${trade.pnl_net:.2f}")

        return trade

    def update_mfe_mae(self, date: pd.Timestamp, prices: pd.Series) -> None:
        """
        Update Maximum Favorable/Adverse Excursion for open positions.

        Args:
            date: Current date
            prices: Current asset prices
        """
        for ticker, trade in self.open_positions.items():
            if ticker in prices and not pd.isna(prices[ticker]):
                current_price = prices[ticker]

                # Calculate current P&L per share
                if trade.quantity > 0:  # Long position
                    pnl_per_share = current_price - trade.entry_price
                else:  # Short position
                    pnl_per_share = trade.entry_price - current_price

                # Update MFE (most favorable)
                if pnl_per_share > trade.mfe:
                    trade.mfe = pnl_per_share

                # Update MAE (most adverse)
                if pnl_per_share < trade.mae:
                    trade.mae = pnl_per_share

    def close_all_positions(
        self, date: pd.Timestamp, prices: pd.Series, commissions: Optional[Dict[str, float]] = None
    ) -> List[Trade]:
        """
        Close all open positions.

        Args:
            date: Closing date
            prices: Final asset prices
            commissions: Commission costs per asset (optional)

        Returns:
            List of closed trades
        """
        closed_trades = []
        tickers_to_close = list(self.open_positions.keys())
        commissions_dict = commissions or {}

        for ticker in tickers_to_close:
            if ticker in prices and not pd.isna(prices[ticker]):
                commission = commissions_dict.get(ticker, 0.0)
                closed_trade = self.close_position(date, ticker, prices[ticker], commission)
                if closed_trade:
                    closed_trades.append(closed_trade)

        return closed_trades

    def get_completed_trades(self) -> List[Trade]:
        """Get all completed trades."""
        return self.trades.copy()

    def get_open_positions(self) -> Dict[str, Trade]:
        """Get all open positions."""
        return self.open_positions.copy()

    def has_open_positions(self) -> bool:
        """Check if there are any open positions."""
        return len(self.open_positions) > 0

    def get_trade_count(self) -> int:
        """Get total number of completed trades."""
        return len(self.trades)
