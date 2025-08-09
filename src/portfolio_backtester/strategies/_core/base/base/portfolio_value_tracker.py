"""Portfolio value tracking for meta strategies with market data integration."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

from .trade_record import TradeRecord, PositionRecord

logger = logging.getLogger(__name__)


class PortfolioValueTracker:
    """
    Tracks portfolio value over time using actual market prices.

    This class provides more accurate portfolio valuation by using current market prices
    rather than just book values from trade records.
    """

    def __init__(self, initial_capital: float):
        """
        Initialize the portfolio value tracker.

        Args:
            initial_capital: Starting capital
        """
        self.initial_capital = initial_capital
        self._cash_balance = initial_capital
        self._positions: Dict[str, PositionRecord] = {}
        self._value_history: Dict[pd.Timestamp, Dict[str, float]] = {}

    def update_from_trade(self, trade: TradeRecord) -> None:
        """
        Update portfolio state from a trade.

        Args:
            trade: TradeRecord to process
        """
        # Update positions
        if trade.asset not in self._positions:
            self._positions[trade.asset] = PositionRecord(
                asset=trade.asset,
                quantity=0.0,
                average_price=0.0,
                last_update=trade.date,
                strategy_contributions={},
            )

        self._positions[trade.asset].add_trade(trade)

        # Remove position if it's effectively zero
        if self._positions[trade.asset].is_flat:
            del self._positions[trade.asset]

        # Update cash balance
        if trade.is_buy:
            tv = float(trade.trade_value) if trade.trade_value is not None else 0.0
            tc = float(trade.transaction_cost) if trade.transaction_cost is not None else 0.0
            self._cash_balance -= tv + tc
        else:
            tv = float(trade.trade_value) if trade.trade_value is not None else 0.0
            tc = float(trade.transaction_cost) if trade.transaction_cost is not None else 0.0
            self._cash_balance += tv - tc

    def calculate_portfolio_value(
        self, date: pd.Timestamp, market_data: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Calculate portfolio value at a specific date using market prices.

        Args:
            date: Date to calculate value for
            market_data: Market data DataFrame with current prices

        Returns:
            Total portfolio value
        """
        if not self._positions:
            return float(self._cash_balance)

        position_value = 0.0

        for asset, position in self._positions.items():
            if market_data is not None:
                # Use market price if available
                market_price = self._get_market_price(asset, date, market_data)
                if market_price is not None:
                    position_value += position.quantity * market_price
                else:
                    # Fallback to book value
                    position_value += position.market_value
                    logger.warning(f"No market price for {asset} on {date}, using book value")
            else:
                # Use book value
                position_value += position.market_value

        total_value = float(self._cash_balance) + float(position_value)

        # Store in history
        self._value_history[date] = {
            "total_value": total_value,
            "cash_balance": self._cash_balance,
            "position_value": position_value,
            "num_positions": len(self._positions),
        }

        return total_value

    def _get_market_price(
        self, asset: str, date: pd.Timestamp, market_data: pd.DataFrame
    ) -> Optional[float]:
        """
        Get market price for an asset on a specific date.

        Args:
            asset: Asset symbol
            date: Date to get price for
            market_data: Market data DataFrame

        Returns:
            Market price or None if not available
        """
        try:
            if isinstance(market_data.columns, pd.MultiIndex):
                # MultiIndex columns (Ticker, Field)
                if (asset, "Close") in market_data.columns:
                    price_series = market_data[(asset, "Close")]
                    if date in price_series.index:
                        price = price_series.loc[date]
                        return float(price) if not pd.isna(price) else None
            else:
                # Simple columns
                if asset in market_data.columns:
                    price_series = market_data[asset]
                    if date in price_series.index:
                        price = price_series.loc[date]
                        return float(price) if not pd.isna(price) else None

            return None

        except Exception as e:
            logger.warning(f"Error getting market price for {asset} on {date}: {e}")
            return None

    def get_value_timeline(
        self, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        Get portfolio value timeline as DataFrame.

        Args:
            start_date: Start date for timeline
            end_date: End date for timeline

        Returns:
            DataFrame with portfolio value timeline
        """
        if not self._value_history:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(self._value_history, orient="index")
        df.index.name = "date"
        df = df.sort_index()

        # Filter by date range
        if start_date is not None:
            df = df[df.index >= start_date]
        if end_date is not None:
            df = df[df.index <= end_date]

        if df.empty:
            return df

        # Calculate returns and other metrics
        df["returns"] = df["total_value"].pct_change().fillna(0.0)
        df["cumulative_return"] = (df["total_value"] / self.initial_capital) - 1

        # Calculate drawdown
        df["running_max"] = df["total_value"].expanding().max()
        df["drawdown"] = (df["total_value"] - df["running_max"]) / df["running_max"]

        return df

    def get_position_summary(
        self, date: Optional[pd.Timestamp] = None
    ) -> Dict[str, Dict[str, float | bool]]:
        """
        Get summary of current positions.

        Args:
            date: Date for position summary (unused for now, for future enhancement)

        Returns:
            Dictionary with position summaries
        """
        summary = {}

        for asset, position in self._positions.items():
            summary[asset] = {
                "quantity": float(position.quantity),
                "average_price": float(position.average_price),
                "market_value": float(position.market_value),
                "is_long": bool(position.is_long),
                "is_short": bool(position.is_short),
            }

        return summary

    def calculate_performance_metrics(
        self,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        risk_free_rate: float = 0.0,
    ) -> Dict[str, float]:
        """
        Calculate performance metrics based on portfolio value timeline.

        Args:
            start_date: Start date for calculation
            end_date: End date for calculation
            risk_free_rate: Risk-free rate for Sharpe ratio calculation

        Returns:
            Dictionary with performance metrics
        """
        timeline = self.get_value_timeline(start_date, end_date)

        if timeline.empty or len(timeline) < 2:
            return {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "current_value": self.initial_capital,
            }

        returns = timeline["returns"].dropna()

        if returns.empty:
            return {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "current_value": timeline["total_value"].iloc[-1],
            }

        # Calculate metrics
        total_return = timeline["cumulative_return"].iloc[-1]
        annualized_return = returns.mean() * 252  # Assuming daily returns
        volatility = returns.std() * np.sqrt(252)
        max_drawdown = timeline["drawdown"].min()

        # Sharpe ratio
        sharpe_ratio = 0.0
        if volatility > 0:
            sharpe_ratio = (annualized_return - risk_free_rate) / volatility

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "current_value": timeline["total_value"].iloc[-1],
            "cash_balance": self._cash_balance,
            "num_positions": len(self._positions),
            "trading_days": len(returns),
        }

    def get_cash_balance(self) -> float:
        """Get current cash balance."""
        return self._cash_balance

    def get_positions(self) -> Dict[str, PositionRecord]:
        """Get current positions."""
        return self._positions.copy()

    def reset(self) -> None:
        """Reset the tracker to initial state."""
        self._cash_balance = self.initial_capital
        self._positions.clear()
        self._value_history.clear()
        logger.debug("Portfolio value tracker reset to initial state")

    def export_value_history(self) -> pd.DataFrame:
        """
        Export value history for analysis.

        Returns:
            DataFrame with complete value history
        """
        return self.get_value_timeline()

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive summary statistics.

        Returns:
            Dictionary with summary statistics
        """
        timeline = self.get_value_timeline()
        performance = self.calculate_performance_metrics()
        position_summary = self.get_position_summary()

        return {
            "initial_capital": self.initial_capital,
            "current_cash": self._cash_balance,
            "current_positions": len(self._positions),
            "total_value_points": len(self._value_history),
            "performance_metrics": performance,
            "position_summary": position_summary,
            "value_timeline_available": not timeline.empty,
            "first_value_date": timeline.index.min() if not timeline.empty else None,
            "last_value_date": timeline.index.max() if not timeline.empty else None,
        }
