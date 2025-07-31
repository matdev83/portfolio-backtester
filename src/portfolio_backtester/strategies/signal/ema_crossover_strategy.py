"""
Simple EMA Crossover Strategy

This strategy uses exponential moving average crossovers to generate buy/sell signals.
- Long signal: Fast EMA crosses above Slow EMA
- Exit signal: Fast EMA crosses below Slow EMA
"""

from typing import Optional, Set

import pandas as pd

from ..base.signal_strategy import SignalStrategy
# Optional Numba optimisation
try:
    from ...numba_optimized import ema_fast
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


class EMAStrategy(SignalStrategy):
    """Simple EMA crossover strategy implementation."""
    
    def __init__(self, strategy_config: dict):
        super().__init__(strategy_config)
        self.fast_ema_days = strategy_config.get('fast_ema_days', 20)
        self.slow_ema_days = strategy_config.get('slow_ema_days', 64)
        self.leverage = strategy_config.get('leverage', 1.0)
        
    @classmethod
    def tunable_parameters(cls) -> Set[str]:
        """Return set of tunable parameters for this strategy."""
        return {'fast_ema_days', 'slow_ema_days', 'leverage'}
    
    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: Optional[pd.DataFrame] = None,
        current_date: Optional[pd.Timestamp] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        **kwargs,
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
        # Check if current_date is provided
        # Handle None current_date gracefully - use the last date in the data
        if current_date is None:
            current_date = all_historical_data.index[-1]
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
        
        # --------------------------------------------------------------
        # Vectorised EMA calculation for all tickers.
        # Require only *exactly* max(fast, slow) periods of history instead
        # of +10; this prevents ‘no data’ on the very first WFO window.
        # --------------------------------------------------------------

        min_periods = max(self.fast_ema_days, self.slow_ema_days)
        valid_mask = close_prices.notna().sum() >= min_periods
        valid_tickers = close_prices.columns[valid_mask]
        fast_ema = close_prices[valid_tickers].ewm(span=self.fast_ema_days).mean()
        slow_ema = close_prices[valid_tickers].ewm(span=self.slow_ema_days).mean()
        # Get EMA values at current date (or closest available date)
        if current_date in fast_ema.index and current_date in slow_ema.index:
            fast_values = fast_ema.loc[current_date]
            slow_values = slow_ema.loc[current_date]
            signal_mask = (fast_values > slow_values) & (~fast_values.isna()) & (~slow_values.isna())
            weights = pd.Series(0.0, index=available_tickers)
            weights.loc[signal_mask.index[signal_mask]] = 1.0
        else:
            # If current_date is not available, use the last available date before current_date
            available_dates = fast_ema.index[fast_ema.index <= current_date]
            if len(available_dates) > 0:
                last_available_date = available_dates[-1]
                fast_values = fast_ema.loc[last_available_date]
                slow_values = slow_ema.loc[last_available_date]
                signal_mask = (fast_values > slow_values) & (~fast_values.isna()) & (~slow_values.isna())
                weights = pd.Series(0.0, index=available_tickers)
                weights.loc[signal_mask.index[signal_mask]] = 1.0
                import logging
                logging.getLogger(__name__).debug(f"EMAStrategy: using {last_available_date} instead of {current_date}")
            else:
                weights = pd.Series(0.0, index=available_tickers)
        # Equal-weight allocation among selected stocks
        if weights.sum() > 0:
            weights = weights / weights.sum()
            # Apply leverage
            weights = weights * self.leverage
        else:
            import logging
            logging.getLogger(__name__).debug("EMAStrategy: no positions selected on %s", current_date)
            # Return as DataFrame with current date as index
        return pd.DataFrame([weights], index=[current_date])
    
    def __str__(self):
        return f"EMA({self.fast_ema_days},{self.slow_ema_days})" 