"""
UVXY RSI Strategy

A strategy that trades UVXY (VIX volatility ETF) based on SPY RSI(2) signals.
- Universe: UVXY only
- Signal: RSI(2) on SPY daily data
- Entry: Short UVXY when SPY RSI(2) falls below threshold (default 30)
- Exit: Cover short on next trading day's close
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import ta

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class UvxyRsiStrategy(BaseStrategy):
    """
    UVXY RSI Strategy implementation.
    
    This strategy:
    1. Trades only UVXY (universe should contain only UVXY)
    2. Uses SPY daily data for RSI(2) signal generation
    3. Goes short UVXY when SPY RSI(2) < threshold (configurable, default 30)
    4. Covers short position on the next trading day's close (1-day holding period)
    """

    def __init__(self, strategy_config: Dict[str, Any]):
        # Set default parameters
        defaults = {
            "rsi_period": 2,
            "rsi_threshold": 30.0,
            "price_column_asset": "Close",
            "price_column_benchmark": "Close",
            "long_only": False,  # Must allow shorts
        }
        
        # Ensure nested dict exists and apply defaults
        strategy_params = strategy_config.setdefault("strategy_params", {})
        for k, v in defaults.items():
            strategy_params.setdefault(k, v)
            
        super().__init__(strategy_config)
        
        # Legacy state tracking (kept for backward compatibility)
        # The timing controller now handles most state management
        self._previous_signal = 0.0
        self._entry_date = None

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        """Return the set of parameters that can be optimized."""
        return {
            "rsi_period",
            "rsi_threshold",
        }



    def get_non_universe_data_requirements(self) -> list[str]:
        """Return list of non-universe tickers needed for signal generation."""
        return ["SPY"]

    @staticmethod
    def _calculate_rsi(price_series: pd.Series, period: int) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index) for a price series using ta library.
        
        Args:
            price_series: Series of prices
            period: RSI period (e.g., 2 for RSI(2))
            
        Returns:
            Series of RSI values
        """
        if len(price_series) < period + 1:
            return pd.Series(index=price_series.index, dtype=float)
        
        # Use ta library for RSI calculation
        rsi = ta.momentum.RSIIndicator(close=price_series, window=period).rsi()
        
        return rsi

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        non_universe_historical_data: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate trading signals for UVXY based on SPY RSI.
        
        This method generates signals for ALL trading days from start_date to end_date,
        not just rebalancing dates, to ensure proper daily signal logic.
        """
        params = self.strategy_config.get("strategy_params", {})
        rsi_period = int(params.get("rsi_period", 2))
        rsi_threshold = float(params.get("rsi_threshold", 30.0))
        
        # Get universe tickers (should be UVXY only)
        if isinstance(all_historical_data.columns, pd.MultiIndex):
            universe_tickers = all_historical_data.columns.get_level_values('Ticker').unique().tolist()
        else:
            universe_tickers = all_historical_data.columns.tolist()
        
        # Validate that we have SPY data
        if non_universe_historical_data is None or non_universe_historical_data.empty:
            logger.warning(f"No SPY data available for signal generation")
            return self._create_empty_signals_range(universe_tickers, current_date, current_date)
            
        # Extract SPY close prices up to current date
        spy_data = self._extract_spy_prices(non_universe_historical_data, current_date)
        if spy_data is None or len(spy_data) < rsi_period + 1:
            logger.warning(f"Insufficient SPY data for RSI calculation")
            return self._create_empty_signals_range(universe_tickers, current_date, current_date)
        
        # Generate signals for the full date range (or just current_date if no range specified)
        if start_date is not None and end_date is not None:
            # Generate signals for the full range
            return self._generate_signals_for_range(spy_data, universe_tickers, rsi_period, rsi_threshold, start_date, end_date)
        else:
            # Generate signal for current date only
            return self._generate_signal_for_date(spy_data, universe_tickers, rsi_period, rsi_threshold, current_date)
    
    def _generate_signals_for_range(self, spy_data: pd.Series, universe_tickers: list, 
                                   rsi_period: int, rsi_threshold: float,
                                   start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """Generate signals for a full date range with proper daily logic."""
        
        # Get all trading days in the range
        trading_days = spy_data.index[(spy_data.index >= start_date) & (spy_data.index <= end_date)]
        
        if len(trading_days) == 0:
            return self._create_empty_signals_range(universe_tickers, start_date, end_date)
        
        # Calculate RSI for the full SPY series
        spy_rsi = self._calculate_rsi(spy_data, rsi_period)
        
        # Initialize signals DataFrame
        signals = pd.DataFrame(
            index=trading_days,
            columns=universe_tickers,
            dtype=float
        ).fillna(0.0)
        
        # Track position state
        in_position = False
        entry_date = None
        
        for date in trading_days:
            if date not in spy_rsi.index:
                continue
                
            current_rsi = spy_rsi.loc[date]
            
            if np.isnan(current_rsi):
                continue
            
            # Signal logic
            if not in_position and current_rsi < rsi_threshold:
                # Enter short position
                for ticker in universe_tickers:
                    signals.loc[date, ticker] = -1.0 / len(universe_tickers)
                in_position = True
                entry_date = date
                logger.debug(f"Entering short UVXY on {date}: SPY RSI({rsi_period}) = {current_rsi:.2f} < {rsi_threshold}")
                
            elif in_position and entry_date is not None:
                # Check if we should exit (1-day holding period)
                if date > entry_date:
                    # Exit position
                    for ticker in universe_tickers:
                        signals.loc[date, ticker] = 0.0
                    in_position = False
                    entry_date = None
                    logger.debug(f"Closing UVXY position on {date}: 1-day holding period complete")
                else:
                    # Hold position
                    for ticker in universe_tickers:
                        signals.loc[date, ticker] = -1.0 / len(universe_tickers)
        
        return signals
    
    def _generate_signal_for_date(self, spy_data: pd.Series, universe_tickers: list,
                                 rsi_period: int, rsi_threshold: float, 
                                 current_date: pd.Timestamp) -> pd.DataFrame:
        """Generate signal for a single date with proper entry/exit logic."""
        
        # Calculate RSI on SPY up to current date
        spy_rsi = self._calculate_rsi(spy_data, rsi_period)
        current_rsi = spy_rsi.loc[current_date] if current_date in spy_rsi.index else np.nan
        
        # Create signals DataFrame
        signals = pd.DataFrame(
            index=[current_date],
            columns=universe_tickers,
            dtype=float
        ).fillna(0.0)
        
        # Check if we're currently in a position
        if self._entry_date is not None:
            # We're in a position, check if we should exit (1-day holding period)
            if current_date > self._entry_date:
                # Exit position (close short)
                for ticker in universe_tickers:
                    signals.loc[current_date, ticker] = 0.0
                logger.info(f"Closing UVXY short position on {current_date}: 1-day holding period complete")
                self._entry_date = None
                self._previous_signal = 0.0
            else:
                # Hold position
                for ticker in universe_tickers:
                    signals.loc[current_date, ticker] = -1.0 / len(universe_tickers)
                self._previous_signal = -1.0 / len(universe_tickers)
        else:
            # Not in position, check for entry signal
            if not np.isnan(current_rsi) and current_rsi < rsi_threshold:
                # Enter short position
                for ticker in universe_tickers:
                    signals.loc[current_date, ticker] = -1.0 / len(universe_tickers)
                logger.info(f"Short signal on {current_date}: SPY RSI({rsi_period}) = {current_rsi:.2f} < {rsi_threshold}")
                self._entry_date = current_date
                self._previous_signal = -1.0 / len(universe_tickers)
            
        return signals

    def _extract_spy_prices(self, non_universe_data: pd.DataFrame, current_date: pd.Timestamp) -> Optional[pd.Series]:
        """Extract SPY close prices from non-universe data."""
        try:
            # Handle MultiIndex columns
            if isinstance(non_universe_data.columns, pd.MultiIndex):
                if ('SPY', 'Close') in non_universe_data.columns:
                    spy_prices = non_universe_data[('SPY', 'Close')]
                elif 'SPY' in non_universe_data.columns.get_level_values('Ticker'):
                    # Get all SPY data and extract Close
                    spy_data = non_universe_data.xs('SPY', level='Ticker', axis=1)
                    if 'Close' in spy_data.columns:
                        spy_prices = spy_data['Close']
                    else:
                        logger.warning("No Close price found for SPY in non-universe data")
                        return None
                else:
                    logger.warning("SPY not found in non-universe data columns")
                    return None
            else:
                # Single level columns
                if 'SPY' in non_universe_data.columns:
                    spy_prices = non_universe_data['SPY']
                else:
                    logger.warning("SPY not found in non-universe data columns")
                    return None
                    
            # Filter data up to current date and remove NaN values
            spy_prices = spy_prices[spy_prices.index <= current_date].dropna()
            return spy_prices
            
        except Exception as e:
            logger.error(f"Error extracting SPY prices: {e}")
            return None

    def _create_empty_signals_range(self, universe_tickers: list, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """Create empty signals DataFrame for a date range."""
        return pd.DataFrame(
            index=[start_date] if start_date == end_date else pd.date_range(start_date, end_date, freq='D'),
            columns=universe_tickers,
            dtype=float
        ).fillna(0.0)