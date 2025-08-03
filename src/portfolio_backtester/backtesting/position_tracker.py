"""
Position Tracking System for Daily Strategy Evaluation.

This module provides classes for tracking positions and generating trade records
during daily strategy evaluation. It's designed to work with the enhanced WFO
system to provide accurate trade duration calculations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open position.
    
    Attributes:
        ticker: Asset ticker symbol
        entry_date: Date when position was opened
        weight: Position weight (positive for long, negative for short)
        entry_price: Price at entry (optional, for P&L calculation)
    """
    ticker: str
    entry_date: pd.Timestamp
    weight: float
    entry_price: Optional[float] = None


@dataclass
class Trade:
    """Represents a completed trade.
    
    Attributes:
        ticker: Asset ticker symbol
        entry_date: Date when position was opened
        exit_date: Date when position was closed
        entry_weight: Position weight at entry
        exit_weight: Position weight at exit (usually 0)
        duration_days: Trade duration in business days
        pnl: Profit/loss if price data available (optional)
    """
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_weight: float
    exit_weight: float
    duration_days: int
    pnl: Optional[float] = None


class PositionTracker:
    """Tracks positions and generates trade records for daily evaluation.
    
    This class maintains the state of open positions and creates trade records
    when positions are closed. It's designed to work with daily strategy
    evaluation to provide accurate trade duration calculations.
    
    Attributes:
        current_positions: Dictionary of currently open positions
        completed_trades: List of completed trade records
        daily_weights: List of daily weight series for analysis
    """
    
    def __init__(self):
        """Initialize the position tracker."""
        self.current_positions: Dict[str, Position] = {}
        self.completed_trades: List[Trade] = []
        self.daily_weights: List[pd.Series] = []
        
        logger.debug("PositionTracker initialized")
    
    def update_positions(self, 
                        signals: pd.DataFrame, 
                        current_date: pd.Timestamp,
                        prices: Optional[pd.DataFrame] = None) -> pd.Series:
        """Update positions based on new signals and return current weights.
        
        This method processes new strategy signals and updates the position
        state accordingly. It handles opening new positions, closing existing
        positions, and adjusting position sizes.
        
        Args:
            signals: DataFrame with strategy signals (columns=tickers, index=[current_date])
            current_date: Current evaluation date
            prices: Optional price data for P&L calculation
            
        Returns:
            Series of current position weights indexed by ticker
        """
        if signals.empty or current_date not in signals.index:
            logger.warning(f"No signals found for date {current_date}")
            return pd.Series(dtype=float)
        
        current_weights = {}
        signals_for_date = signals.loc[current_date]
        
        # Process each ticker in the signals
        for ticker in signals.columns:
            target_weight = signals_for_date[ticker] if not pd.isna(signals_for_date[ticker]) else 0.0
            current_position = self.current_positions.get(ticker)
            
            # Handle position opening
            if current_position is None and abs(target_weight) > 1e-6:
                entry_price = self._get_price(prices, current_date, ticker) if prices is not None else None
                self.current_positions[ticker] = Position(
                    ticker=ticker,
                    entry_date=current_date,
                    weight=target_weight,
                    entry_price=entry_price
                )
                current_weights[ticker] = target_weight
                
                logger.debug(f"Opened position: {ticker} weight={target_weight:.4f} on {current_date.date()}")
                
            # Handle position closing
            elif current_position is not None and abs(target_weight) < 1e-6:
                exit_price = self._get_price(prices, current_date, ticker) if prices is not None else None
                trade = self._close_position(current_position, current_date, exit_price)
                self.completed_trades.append(trade)
                del self.current_positions[ticker]
                current_weights[ticker] = 0.0
                
                logger.debug(
                    f"Closed position: {ticker} duration={trade.duration_days} days on {current_date.date()}"
                )
                
            # Handle position adjustment
            elif current_position is not None:
                current_position.weight = target_weight
                current_weights[ticker] = target_weight
                
                if abs(target_weight - current_position.weight) > 1e-6:
                    logger.debug(
                        f"Adjusted position: {ticker} weight={target_weight:.4f} on {current_date.date()}"
                    )
            else:
                current_weights[ticker] = 0.0
        
        # Store daily weights for analysis
        weight_series = pd.Series(current_weights, name=current_date)
        self.daily_weights.append(weight_series)
        
        return weight_series
    
    def _close_position(self, position: Position, exit_date: pd.Timestamp, 
                       exit_price: Optional[float] = None) -> Trade:
        """Close a position and create a trade record.
        
        Args:
            position: Position to close
            exit_date: Date when position is closed
            exit_price: Price at exit (optional, for P&L calculation)
            
        Returns:
            Trade record for the completed trade
        """
        # Calculate duration in business days
        duration = len(pd.bdate_range(position.entry_date, exit_date)) - 1
        
        # Calculate P&L if prices are available
        pnl = None
        if position.entry_price is not None and exit_price is not None:
            price_return = (exit_price - position.entry_price) / position.entry_price
            pnl = price_return * position.weight
        
        return Trade(
            ticker=position.ticker,
            entry_date=position.entry_date,
            exit_date=exit_date,
            entry_weight=position.weight,
            exit_weight=0.0,
            duration_days=duration,
            pnl=pnl
        )
    
    def _get_price(self, prices: pd.DataFrame, date: pd.Timestamp, ticker: str) -> Optional[float]:
        """Get price for a specific ticker and date.
        
        Args:
            prices: Price DataFrame
            date: Date to get price for
            ticker: Ticker symbol
            
        Returns:
            Price value or None if not available
        """
        try:
            if isinstance(prices.columns, pd.MultiIndex):
                # Multi-level columns (Ticker, Field)
                if (ticker, 'Close') in prices.columns and date in prices.index:
                    return prices.loc[date, (ticker, 'Close')]
            else:
                # Single-level columns
                if ticker in prices.columns and date in prices.index:
                    return prices.loc[date, ticker]
        except (KeyError, IndexError):
            pass
        
        return None
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current position weights.
        
        Returns:
            Dictionary mapping ticker to current weight
        """
        return {ticker: pos.weight for ticker, pos in self.current_positions.items()}
    
    def get_completed_trades(self) -> List[Trade]:
        """Get all completed trades.
        
        Returns:
            List of completed Trade objects
        """
        return self.completed_trades.copy()
    
    def get_daily_weights_df(self) -> pd.DataFrame:
        """Get daily weights as DataFrame.
        
        Returns:
            DataFrame with dates as index and tickers as columns
        """
        if not self.daily_weights:
            return pd.DataFrame()
        
        # Combine all daily weight series
        all_tickers = set()
        for weights in self.daily_weights:
            all_tickers.update(weights.index)
        
        weights_df = pd.DataFrame(
            index=[w.name for w in self.daily_weights], 
            columns=sorted(all_tickers)
        )
        
        for weights in self.daily_weights:
            weights_df.loc[weights.name] = weights.reindex(weights_df.columns, fill_value=0.0)
        
        return weights_df.fillna(0.0)
    
    def get_trade_summary(self) -> Dict[str, float]:
        """Get summary statistics for completed trades.
        
        Returns:
            Dictionary with trade summary statistics
        """
        if not self.completed_trades:
            return {
                'total_trades': 0,
                'avg_duration': 0.0,
                'min_duration': 0,
                'max_duration': 0,
                'avg_pnl': 0.0
            }
        
        durations = [trade.duration_days for trade in self.completed_trades]
        pnls = [trade.pnl for trade in self.completed_trades if trade.pnl is not None]
        
        return {
            'total_trades': len(self.completed_trades),
            'avg_duration': np.mean(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'avg_pnl': np.mean(pnls) if pnls else 0.0
        }
    
    def reset(self):
        """Reset the tracker state."""
        self.current_positions.clear()
        self.completed_trades.clear()
        self.daily_weights.clear()
        
        logger.debug("PositionTracker reset")