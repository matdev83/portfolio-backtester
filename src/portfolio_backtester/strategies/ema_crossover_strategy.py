"""
Simple EMA Crossover Strategy

This strategy uses exponential moving average crossovers to generate buy/sell signals.
- Long signal: Fast EMA crosses above Slow EMA
- Exit signal: Fast EMA crosses below Slow EMA
"""

from typing import Optional, Set

import pandas as pd

from .base_strategy import BaseStrategy
# Optional Numba optimisation
try:
    from ..numba_optimized import ema_fast
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


class EMAStrategy(BaseStrategy):
    """Simple EMA crossover strategy implementation."""
    
    def __init__(self, strategy_config: dict):
        super().__init__(strategy_config)
        self.fast_ema_days = strategy_config.get('fast_ema_days', 20)
        self.slow_ema_days = strategy_config.get('slow_ema_days', 64)
        self.leverage = strategy_config.get('leverage', 1.0)
        
    @staticmethod
    def tunable_parameters() -> Set[str]:
        """Return set of tunable parameters for this strategy."""
        return {'fast_ema_days', 'slow_ema_days', 'leverage'}
    
    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Generate EMA crossover signals.
        
        Args:
            all_historical_data: DataFrame with historical OHLCV data for all assets
            benchmark_historical_data: DataFrame with historical OHLCV data for benchmark
            current_date: The current date for signal generation
            start_date: Optional start date for WFO window
            end_date: Optional end date for WFO window
            
        Returns:
            DataFrame with signals (weights) for the current date
        """
        # Check if we should generate signals for this date
        if start_date is not None and current_date < start_date:
            return pd.DataFrame(index=[current_date], columns=all_historical_data.columns).fillna(0.0)
        if end_date is not None and current_date > end_date:
            return pd.DataFrame(index=[current_date], columns=all_historical_data.columns).fillna(0.0)
        
        # Extract close prices
        if isinstance(all_historical_data.columns, pd.MultiIndex):
            # Multi-index columns (Ticker, Field)
            close_prices = all_historical_data.xs('Close', level='Field', axis=1)
        else:
            # Assume single-level columns are already close prices
            close_prices = all_historical_data
        
        # Get universe tickers from available columns
        # Since we don't have access to global_config here, use available columns
        if isinstance(all_historical_data.columns, pd.MultiIndex):
            available_tickers = list(close_prices.columns)
        else:
            available_tickers = list(all_historical_data.columns)
        
        # Initialize weights
        weights = pd.Series(0.0, index=available_tickers)
        
        # Calculate EMA signals for each ticker
        for ticker in available_tickers:
            prices = close_prices[ticker].dropna()
            
            if len(prices) < max(self.fast_ema_days, self.slow_ema_days) + 10:
                # Not enough data for reliable signals
                continue
                
            # Calculate EMAs up to current date
            if NUMBA_AVAILABLE:
                fast_ema_values = ema_fast(prices.values, self.fast_ema_days)
                slow_ema_values = ema_fast(prices.values, self.slow_ema_days)
                fast_ema = pd.Series(fast_ema_values, index=prices.index)
                slow_ema = pd.Series(slow_ema_values, index=prices.index)
            else:
                fast_ema = prices.ewm(span=self.fast_ema_days).mean()
                slow_ema = prices.ewm(span=self.slow_ema_days).mean()
            
            # Get EMA values at current date
            if current_date in fast_ema.index and current_date in slow_ema.index:
                fast_value = fast_ema.loc[current_date]
                slow_value = slow_ema.loc[current_date]
                
                # Long signal when fast EMA > slow EMA
                if not pd.isna(fast_value) and not pd.isna(slow_value) and fast_value > slow_value:
                    weights[ticker] = 1.0
        
        # Equal weight allocation among selected stocks
        if weights.sum() > 0:
            weights = weights / weights.sum()
            # Apply leverage
            weights = weights * self.leverage
        
        # Return as DataFrame with current date as index
        return pd.DataFrame([weights], index=[current_date])
    
    def __str__(self):
        return f"EMA({self.fast_ema_days},{self.slow_ema_days})" 