"""Trade aggregation system for meta strategies."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from datetime import datetime

import pandas as pd
import numpy as np

from .trade_record import TradeRecord, PositionRecord, TradeSide

logger = logging.getLogger(__name__)


class TradeAggregator:
    """
    Aggregates and tracks trades from multiple sub-strategies within a meta strategy.
    
    This class is responsible for:
    - Recording all trades from sub-strategies
    - Maintaining current positions across all assets
    - Calculating portfolio value over time
    - Computing performance metrics based on actual trade execution
    """
    
    def __init__(self, initial_capital: float, allocation_mode: str = "reinvestment"):
        """
        Initialize the trade aggregator.
        
        Args:
            initial_capital: Starting capital for the meta strategy
            allocation_mode: Capital allocation mode ("reinvestment" or "fixed_fractional")
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.allocation_mode = allocation_mode
        
        # Validate allocation mode
        valid_modes = ["reinvestment", "compound", "fixed_fractional", "fixed_capital"]
        if allocation_mode not in valid_modes:
            raise ValueError(f"Invalid allocation_mode '{allocation_mode}'. Must be one of: {valid_modes}")
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"TradeAggregator initialized with allocation_mode='{allocation_mode}'")
        
        # Trade tracking
        self._trade_history: List[TradeRecord] = []
        self._trades_by_date: Dict[pd.Timestamp, List[TradeRecord]] = defaultdict(list)
        self._trades_by_strategy: Dict[str, List[TradeRecord]] = defaultdict(list)
        self._trades_by_asset: Dict[str, List[TradeRecord]] = defaultdict(list)
        
        # Position tracking
        self._current_positions: Dict[str, PositionRecord] = {}
        self._position_history: Dict[pd.Timestamp, Dict[str, PositionRecord]] = {}
        
        # Performance tracking
        self._portfolio_values: Dict[pd.Timestamp, float] = {}
        self._cash_balance = initial_capital
        
        # Strategy allocation tracking
        self._strategy_allocations: Dict[str, float] = {}
        self._strategy_capital_history: Dict[str, Dict[pd.Timestamp, float]] = defaultdict(dict)
    
    def track_sub_strategy_trade(self, trade: TradeRecord) -> None:
        """
        Record a trade from a sub-strategy.
        
        Args:
            trade: TradeRecord representing the trade to track
        """
        logger.debug(f"Tracking trade: {trade}")
        
        # Add to trade history
        self._trade_history.append(trade)
        self._trades_by_date[trade.date].append(trade)
        self._trades_by_strategy[trade.strategy_id].append(trade)
        self._trades_by_asset[trade.asset].append(trade)
        
        # Update positions
        self._update_position(trade)
        
        # Update cash balance
        self._update_cash_balance(trade)
        
        # Update portfolio value (using trade price as market price for this asset)
        market_prices = {trade.asset: trade.price}
        self._update_portfolio_value(trade.date, market_prices)
        
        # Update current capital to reflect the new portfolio value
        if self._portfolio_values:
            latest_date = max(self._portfolio_values.keys())
            self.current_capital = self._portfolio_values[latest_date]
        
        logger.debug(f"Trade tracked. Cash balance: ${self._cash_balance:.2f}, Portfolio value: ${self.current_capital:.2f}")
    
    def _update_position(self, trade: TradeRecord) -> None:
        """Update position records with a new trade."""
        if trade.asset not in self._current_positions:
            self._current_positions[trade.asset] = PositionRecord(
                asset=trade.asset,
                quantity=0.0,
                average_price=0.0,
                last_update=trade.date,
                strategy_contributions={}
            )
        
        self._current_positions[trade.asset].add_trade(trade)
        
        # Remove position if it's effectively zero
        if self._current_positions[trade.asset].is_flat:
            del self._current_positions[trade.asset]
    
    def _update_cash_balance(self, trade: TradeRecord) -> None:
        """Update cash balance based on trade execution."""
        if trade.is_buy:
            # Buying reduces cash
            self._cash_balance -= (trade.trade_value + trade.transaction_cost)
        else:
            # Selling increases cash
            self._cash_balance += (trade.trade_value - trade.transaction_cost)
    
    def _update_portfolio_value(self, date: pd.Timestamp, market_prices: Optional[Dict[str, float]] = None) -> None:
        """
        Update portfolio value for a given date.
        
        Args:
            date: Date to update portfolio value for
            market_prices: Optional dictionary of current market prices {asset: price}
        """
        position_value = 0.0
        
        for asset, pos in self._current_positions.items():
            if market_prices and asset in market_prices:
                # Use current market price if available
                position_value += pos.quantity * market_prices[asset]
            else:
                # Fall back to book value (average price)
                position_value += pos.market_value
        
        total_value = self._cash_balance + position_value
        
        self._portfolio_values[date] = total_value
        
        # Store position snapshot
        self._position_history[date] = {
            asset: PositionRecord(
                asset=pos.asset,
                quantity=pos.quantity,
                average_price=pos.average_price,
                last_update=pos.last_update,
                strategy_contributions=pos.strategy_contributions.copy()
            )
            for asset, pos in self._current_positions.items()
        }
    
    def get_aggregated_trades(self) -> List[TradeRecord]:
        """
        Get all trades in chronological order.
        
        Returns:
            List of all TradeRecord objects sorted by date
        """
        return sorted(self._trade_history, key=lambda t: t.date)
    
    def get_trades_by_strategy(self, strategy_id: str) -> List[TradeRecord]:
        """
        Get all trades for a specific sub-strategy.
        
        Args:
            strategy_id: ID of the sub-strategy
            
        Returns:
            List of TradeRecord objects for the specified strategy
        """
        return self._trades_by_strategy.get(strategy_id, [])
    
    def get_trades_by_asset(self, asset: str) -> List[TradeRecord]:
        """
        Get all trades for a specific asset.
        
        Args:
            asset: Asset symbol
            
        Returns:
            List of TradeRecord objects for the specified asset
        """
        return self._trades_by_asset.get(asset, [])
    
    def get_trades_by_date_range(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> List[TradeRecord]:
        """
        Get all trades within a date range.
        
        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            
        Returns:
            List of TradeRecord objects within the date range
        """
        return [
            trade for trade in self._trade_history
            if start_date <= trade.date <= end_date
        ]
    
    def get_current_positions(self) -> Dict[str, PositionRecord]:
        """
        Get current positions across all assets.
        
        Returns:
            Dictionary mapping asset symbols to PositionRecord objects
        """
        return self._current_positions.copy()
    
    def get_position_at_date(self, date: pd.Timestamp) -> Dict[str, PositionRecord]:
        """
        Get positions as of a specific date.
        
        Args:
            date: Date to get positions for
            
        Returns:
            Dictionary mapping asset symbols to PositionRecord objects
        """
        # Find the latest date <= requested date
        available_dates = [d for d in self._position_history.keys() if d <= date]
        if not available_dates:
            return {}
        
        latest_date = max(available_dates)
        return self._position_history[latest_date].copy()
    
    def calculate_portfolio_value(self, date: pd.Timestamp) -> float:
        """
        Calculate portfolio value at a specific date.
        
        Args:
            date: Date to calculate value for
            
        Returns:
            Portfolio value at the specified date
        """
        # Find the latest date <= requested date
        available_dates = [d for d in self._portfolio_values.keys() if d <= date]
        if not available_dates:
            return self.initial_capital
        
        latest_date = max(available_dates)
        return self._portfolio_values[latest_date]
    
    def calculate_weighted_performance(self) -> Dict[str, float]:
        """
        Calculate performance metrics based on aggregated trades.
        
        Returns:
            Dictionary containing various performance metrics
        """
        if not self._trade_history:
            return {
                'total_return': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'current_value': self.initial_capital
            }
        
        # Calculate basic metrics using the latest portfolio value
        total_trades = len(self._trade_history)
        
        # Get the latest portfolio value
        if self._portfolio_values:
            latest_date = max(self._portfolio_values.keys())
            current_value = self._portfolio_values[latest_date]
        else:
            current_value = self.initial_capital
        
        total_pnl = current_value - self.initial_capital
        total_return = total_pnl / self.initial_capital if self.initial_capital > 0 else 0.0
        
        # Calculate trade-level statistics
        # Note: This is simplified - in reality you'd need to match buy/sell pairs
        buy_trades = [t for t in self._trade_history if t.is_buy]
        sell_trades = [t for t in self._trade_history if t.is_sell]
        
        return {
            'total_return': total_return,
            'total_trades': total_trades,
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'total_pnl': total_pnl,
            'current_value': current_value,
            'cash_balance': self._cash_balance,
            'position_value': current_value - self._cash_balance,
            'initial_capital': self.initial_capital
        }
    
    def get_strategy_attribution(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate performance attribution by sub-strategy.
        
        Returns:
            Dictionary mapping strategy_id to performance metrics
        """
        attribution = {}
        
        for strategy_id in self._trades_by_strategy.keys():
            strategy_trades = self._trades_by_strategy[strategy_id]
            
            if not strategy_trades:
                continue
            
            # Calculate basic metrics for this strategy
            total_trades = len(strategy_trades)
            buy_trades = len([t for t in strategy_trades if t.is_buy])
            sell_trades = len([t for t in strategy_trades if t.is_sell])
            
            # Calculate total trade value for this strategy
            total_trade_value = sum(t.trade_value for t in strategy_trades)
            total_transaction_costs = sum(t.transaction_cost for t in strategy_trades)
            
            attribution[strategy_id] = {
                'total_trades': total_trades,
                'buy_trades': buy_trades,
                'sell_trades': sell_trades,
                'total_trade_value': total_trade_value,
                'total_transaction_costs': total_transaction_costs,
                'first_trade_date': min(t.date for t in strategy_trades),
                'last_trade_date': max(t.date for t in strategy_trades)
            }
        
        return attribution
    
    def update_portfolio_values_with_market_data(self, market_data: pd.DataFrame) -> None:
        """
        Update portfolio values using market data for all dates.
        
        Args:
            market_data: DataFrame with dates as index and assets as columns containing prices
        """
        if market_data.empty or not self._trade_history:
            return
        
        # Get all dates from market data
        all_dates = market_data.index
        
        # Track positions over time
        current_positions = {}
        cash_balance = self.initial_capital
        
        # Process all trades chronologically to rebuild position history
        sorted_trades = sorted(self._trade_history, key=lambda t: t.date)
        trade_idx = 0
        
        for date in all_dates:
            # Apply all trades up to this date
            while trade_idx < len(sorted_trades) and sorted_trades[trade_idx].date <= date:
                trade = sorted_trades[trade_idx]
                
                # Update positions
                if trade.asset not in current_positions:
                    current_positions[trade.asset] = 0.0
                
                if trade.is_buy:
                    current_positions[trade.asset] += trade.quantity
                    cash_balance -= (trade.trade_value + trade.transaction_cost)
                else:
                    current_positions[trade.asset] -= trade.quantity
                    cash_balance += (trade.trade_value - trade.transaction_cost)
                
                # Remove zero positions
                if abs(current_positions[trade.asset]) < 1e-8:
                    del current_positions[trade.asset]
                
                trade_idx += 1
            
            # Calculate portfolio value using market prices
            position_value = 0.0
            for asset, quantity in current_positions.items():
                if asset in market_data.columns and date in market_data.index:
                    price = market_data.loc[date, asset]
                    if not pd.isna(price):
                        position_value += quantity * price
            
            total_value = cash_balance + position_value
            self._portfolio_values[date] = total_value
    
    def get_portfolio_timeline(self) -> pd.DataFrame:
        """
        Get portfolio value timeline as a DataFrame.
        
        Returns:
            DataFrame with dates as index and portfolio values
        """
        if not self._portfolio_values:
            return pd.DataFrame()
        
        df = pd.DataFrame.from_dict(
            self._portfolio_values, 
            orient='index', 
            columns=['portfolio_value']
        )
        df.index.name = 'date'
        df = df.sort_index()
        
        # Add returns
        df['returns'] = df['portfolio_value'].pct_change().fillna(0.0)
        df['cumulative_return'] = (df['portfolio_value'] / self.initial_capital) - 1
        
        return df
    
    def export_trades_to_dataframe(self) -> pd.DataFrame:
        """
        Export all trades to a pandas DataFrame.
        
        Returns:
            DataFrame containing all trade records
        """
        if not self._trade_history:
            return pd.DataFrame()
        
        trades_data = [trade.to_dict() for trade in self._trade_history]
        df = pd.DataFrame(trades_data)
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def get_current_cash_balance(self) -> float:
        """
        Get the current cash balance.
        
        Returns:
            Current cash balance after all trades
        """
        return self._cash_balance
    
    def get_current_capital(self) -> float:
        """
        Get the current total capital (cash + positions).
        
        Returns:
            Current total capital
        """
        return self.current_capital
    
    def get_total_return(self) -> float:
        """
        Get the total return as a percentage.
        
        Returns:
            Total return as decimal (e.g., 0.05 for 5%)
        """
        if self.initial_capital == 0:
            return 0.0
        return (self.current_capital - self.initial_capital) / self.initial_capital
    
    def get_summary_statistics(self) -> Dict[str, any]:
        """
        Get comprehensive summary statistics.
        
        Returns:
            Dictionary containing summary statistics
        """
        performance = self.calculate_weighted_performance()
        attribution = self.get_strategy_attribution()
        
        return {
            'performance': performance,
            'attribution': attribution,
            'total_strategies': len(self._trades_by_strategy),
            'total_assets_traded': len(self._trades_by_asset),
            'current_positions': len(self._current_positions),
            'trading_period': {
                'start': min(t.date for t in self._trade_history) if self._trade_history else None,
                'end': max(t.date for t in self._trade_history) if self._trade_history else None
            }
        }