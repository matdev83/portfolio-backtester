"""Trade interceptor for capturing sub-strategy trading decisions."""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps

import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy
from .trade_record import TradeRecord, TradeSide
from ...trading.unified_commission_calculator import get_unified_commission_calculator

logger = logging.getLogger(__name__)


class MetaStrategyTradeInterceptor:
    """
    Wrapper that intercepts sub-strategy trading decisions and converts them to trade records.
    
    This class wraps a sub-strategy instance and intercepts its generate_signals method
    to capture trading decisions and convert them to actual trade records with proper
    capital scaling.
    """
    
    def __init__(
        self,
        sub_strategy: BaseStrategy,
        strategy_id: str,
        allocated_capital: float,
        trade_callback: Callable[[TradeRecord], None],
        transaction_cost_bps: float = 10.0,
        global_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the trade interceptor.
        
        Args:
            sub_strategy: The sub-strategy instance to wrap
            strategy_id: Unique identifier for this sub-strategy
            allocated_capital: Capital allocated to this sub-strategy
            trade_callback: Callback function to call when trades are detected
            transaction_cost_bps: Transaction cost in basis points
            global_config: Global configuration for unified commission calculation
        """
        self.sub_strategy = sub_strategy
        self.strategy_id = strategy_id
        self.allocated_capital = allocated_capital
        self.trade_callback = trade_callback
        
        # Initialize unified commission calculator
        if global_config is None:
            # Fallback configuration for backward compatibility
            global_config = {
                'default_transaction_cost_bps': 10.0,
                'commission_per_share': 0.005,
                'commission_min_per_order': 1.0,
                'commission_max_percent_of_trade': 0.005,
                'slippage_bps': 2.5
            }
        self.commission_calculator = get_unified_commission_calculator(global_config)
        
        # Track previous signals to detect changes
        self._previous_signals: Optional[pd.Series] = None
        
        # Wrap the generate_signals method
        self._wrap_generate_signals()
    
    def _wrap_generate_signals(self) -> None:
        """Wrap the sub-strategy's generate_signals method to intercept trades."""
        original_generate_signals = self.sub_strategy.generate_signals
        
        @wraps(original_generate_signals)
        def intercepted_generate_signals(*args, **kwargs):
            # Call the original method
            signals = original_generate_signals(*args, **kwargs)
            
            # Extract current_date from args/kwargs
            current_date = self._extract_current_date(*args, **kwargs)
            
            # Extract historical data for price lookup
            all_historical_data = self._extract_historical_data(*args, **kwargs)
            
            # Intercept and process the signals
            if current_date is not None and all_historical_data is not None:
                self._process_signals(signals, current_date, all_historical_data)
            
            return signals
        
        # Replace the method
        self.sub_strategy.generate_signals = intercepted_generate_signals
    
    def _extract_current_date(self, *args, **kwargs) -> Optional[pd.Timestamp]:
        """Extract current_date from generate_signals arguments."""
        try:
            # Try to get from kwargs first
            if 'current_date' in kwargs:
                return kwargs['current_date']
            
            # Try to get from positional args (4th argument)
            if len(args) >= 4:
                return args[3]
            
            return None
        except Exception as e:
            logger.warning(f"Could not extract current_date: {e}")
            return None
    
    def _extract_historical_data(self, *args, **kwargs) -> Optional[pd.DataFrame]:
        """Extract all_historical_data from generate_signals arguments."""
        try:
            # Try to get from kwargs first
            if 'all_historical_data' in kwargs:
                return kwargs['all_historical_data']
            
            # Try to get from positional args (1st argument)
            if len(args) >= 1:
                return args[0]
            
            return None
        except Exception as e:
            logger.warning(f"Could not extract historical_data: {e}")
            return None
    
    def _process_signals(
        self,
        signals: pd.DataFrame,
        current_date: pd.Timestamp,
        all_historical_data: pd.DataFrame
    ) -> None:
        """
        Process signals and generate trade records for changes.
        
        Args:
            signals: Signal DataFrame from sub-strategy
            current_date: Current date
            all_historical_data: Historical price data
        """
        if signals.empty or current_date not in signals.index:
            return
        
        current_signals = signals.loc[current_date]
        if isinstance(current_signals, pd.DataFrame):
            current_signals = current_signals.iloc[0]  # Take first row if DataFrame
        
        # Convert to Series if it's not already
        if not isinstance(current_signals, pd.Series):
            logger.warning(f"Unexpected signal format: {type(current_signals)}")
            return
        
        # Detect trades by comparing with previous signals
        trades = self._detect_trades(current_signals, current_date, all_historical_data)
        
        # Execute trade callback for each trade
        for trade in trades:
            try:
                self.trade_callback(trade)
            except Exception as e:
                logger.error(f"Error in trade callback for {trade}: {e}")
        
        # Update previous signals
        self._previous_signals = current_signals.copy()
    
    def _detect_trades(
        self,
        current_signals: pd.Series,
        current_date: pd.Timestamp,
        all_historical_data: pd.DataFrame
    ) -> list[TradeRecord]:
        """
        Detect trades by comparing current signals with previous signals.
        
        Args:
            current_signals: Current signal weights
            current_date: Current date
            all_historical_data: Historical price data
            
        Returns:
            List of TradeRecord objects representing detected trades
        """
        trades = []
        
        # If no previous signals, treat all non-zero current signals as new positions
        if self._previous_signals is None:
            previous_signals = pd.Series(0.0, index=current_signals.index)
        else:
            # Align previous signals with current signals
            previous_signals = self._previous_signals.reindex(current_signals.index, fill_value=0.0)
        
        # Calculate signal changes
        signal_changes = current_signals - previous_signals
        
        # Generate trades for non-zero changes
        for asset, change in signal_changes.items():
            if abs(change) < 1e-8:
                continue  # No significant change
            
            # Get price for this asset
            price = self._get_asset_price(asset, current_date, all_historical_data)
            if price is None or price <= 0:
                logger.warning(f"Could not get price for {asset} on {current_date}")
                continue
            
            # Create trade record
            trade = self._create_trade_record(
                asset=asset,
                signal_change=change,
                price=price,
                current_date=current_date
            )
            
            if trade is not None:
                trades.append(trade)
        
        return trades
    
    def _create_trade_record(
        self,
        asset: str,
        signal_change: float,
        price: float,
        current_date: pd.Timestamp
    ) -> Optional[TradeRecord]:
        """
        Create a trade record from a signal change.
        
        Args:
            asset: Asset symbol
            signal_change: Change in signal weight
            price: Asset price
            current_date: Trade date
            
        Returns:
            TradeRecord or None if trade is too small
        """
        if abs(signal_change) < 1e-8:
            return None
        
        # Calculate trade value and quantity
        trade_value = abs(signal_change * self.allocated_capital)
        quantity = trade_value / price if price > 0 else 0
        
        if quantity < 1e-8:
            return None  # Trade too small
        
        # Determine trade side
        side = TradeSide.BUY if signal_change > 0 else TradeSide.SELL
        
        # Calculate commission using unified calculator
        commission_info = self.commission_calculator.calculate_trade_commission(
            asset=asset,
            date=current_date,
            quantity=quantity if side == TradeSide.BUY else -quantity,
            price=price,
            transaction_costs_bps=self.transaction_cost_bps
        )
        
        # Create trade record with accurate commission
        trade = TradeRecord(
            date=current_date,
            asset=asset,
            quantity=quantity if side == TradeSide.BUY else -quantity,
            price=price,
            side=side,
            strategy_id=self.strategy_id,
            allocated_capital=self.allocated_capital,
            transaction_cost=commission_info.total_cost,
            trade_value=trade_value,
            metadata={
                'signal_change': signal_change,
                'interceptor_generated': True,
                'commission_info': commission_info.to_dict()
            }
        )
        
        logger.debug(f"Created trade from signal change with commission ${commission_info.total_cost:.2f}: {trade}")
        return trade
    
    def _get_asset_price(
        self,
        asset: str,
        date: pd.Timestamp,
        all_historical_data: pd.DataFrame
    ) -> Optional[float]:
        """
        Get the price of an asset on a specific date.
        
        Args:
            asset: Asset symbol
            date: Date to get price for
            all_historical_data: Historical data DataFrame
            
        Returns:
            Price of the asset, or None if not available
        """
        try:
            if isinstance(all_historical_data.columns, pd.MultiIndex):
                # MultiIndex columns (Ticker, Field)
                if (asset, 'Close') in all_historical_data.columns:
                    price_series = all_historical_data[(asset, 'Close')]
                    if date in price_series.index:
                        price = price_series.loc[date]
                        return float(price) if not pd.isna(price) else None
            else:
                # Simple columns
                if asset in all_historical_data.columns:
                    price_series = all_historical_data[asset]
                    if date in price_series.index:
                        price = price_series.loc[date]
                        return float(price) if not pd.isna(price) else None
            
            return None
            
        except Exception as e:
            logger.warning(f"Error getting price for {asset} on {date}: {e}")
            return None
    
    def update_allocated_capital(self, new_capital: float) -> None:
        """
        Update the allocated capital for this sub-strategy.
        
        Args:
            new_capital: New capital allocation
        """
        logger.debug(f"Updating allocated capital for {self.strategy_id}: "
                    f"${self.allocated_capital:,.2f} -> ${new_capital:,.2f}")
        self.allocated_capital = new_capital
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about the wrapped strategy.
        
        Returns:
            Dictionary containing strategy information
        """
        return {
            'strategy_id': self.strategy_id,
            'strategy_class': type(self.sub_strategy).__name__,
            'allocated_capital': self.allocated_capital,
            'transaction_cost_bps': self.transaction_cost_bps,
            'has_previous_signals': self._previous_signals is not None,
            'previous_signal_count': len(self._previous_signals) if self._previous_signals is not None else 0
        }
    
    def reset_state(self) -> None:
        """Reset the interceptor state (clear previous signals)."""
        self._previous_signals = None
        logger.debug(f"Reset state for interceptor {self.strategy_id}")
    
    def __getattr__(self, name: str):
        """Delegate attribute access to the wrapped sub-strategy."""
        return getattr(self.sub_strategy, name)
    
    def __str__(self) -> str:
        """String representation of the interceptor."""
        return f"MetaStrategyTradeInterceptor({self.strategy_id}, {type(self.sub_strategy).__name__})"
    
    def __repr__(self) -> str:
        """Detailed representation of the interceptor."""
        return (f"MetaStrategyTradeInterceptor(strategy_id='{self.strategy_id}', "
                f"strategy_class='{type(self.sub_strategy).__name__}', "
                f"allocated_capital={self.allocated_capital})")