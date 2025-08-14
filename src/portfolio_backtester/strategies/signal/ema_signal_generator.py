"""
EMA Crossover Signal Generator

Generic signal generator for EMA crossover strategies.
Implements ISignalGenerator interface for polymorphic signal generation.
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from ...interfaces.signal_generator_interface import (
    ISignalGenerator,
    signal_generator_factory,
)

logger = logging.getLogger(__name__)


class EmaCrossoverSignalGenerator(ISignalGenerator):
    """
    Generic EMA crossover signal generator.

    Generates signals based on fast/slow EMA crossovers with configurable
    parameters and optional filters.

    Implements ISignalGenerator interface for polymorphic signal generation.
    """

    def __init__(
        self,
        config=None,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ):
        """
        Initialize EMA crossover signal generator.

        WARNING: Direct instantiation is deprecated. Use signal_generator_factory.create_generator() instead.

        Args:
            config: Configuration dictionary (for factory pattern)
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal EMA period for MACD-style signals (default: 9)
        """
        import warnings
        import inspect

        # Check if called directly (not from factory)
        frame = inspect.currentframe()
        try:
            if frame is not None:
                caller_frame = frame.f_back
                if caller_frame is not None:
                    caller_name = caller_frame.f_code.co_name
                    caller_file = caller_frame.f_code.co_filename

                    # Allow factory creation and test creation
                    if "create_generator" not in caller_name and "test" not in caller_file.lower():
                        warnings.warn(
                            "Direct instantiation of EmaCrossoverSignalGenerator is deprecated. "
                            "Use signal_generator_factory.create_generator('ema_crossover', config) instead.",
                            DeprecationWarning,
                            stacklevel=2,
                        )
        finally:
            del frame
        # Handle both factory pattern (config dict) and direct instantiation
        if config is not None:
            self.fast_period = config.get("fast_period", fast_period)
            self.slow_period = config.get("slow_period", slow_period)
            self.signal_period = config.get("signal_period", signal_period)
        else:
            self.fast_period = fast_period
            self.slow_period = slow_period
            self.signal_period = signal_period

        # State tracking
        self._previous_signals: Optional[pd.Series] = None
        self._position_state: Dict[str, bool] = {}

    def generate_signals_for_range(
        self,
        data: Dict[str, Any],
        universe_tickers: List[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Generate signals for a full date range using vectorized operations.

        Args:
            data: Dictionary containing 'price_data' (pd.DataFrame)
            universe_tickers: List of universe ticker symbols
            start_date: Start date for signal generation
            end_date: End date for signal generation

        Returns:
            DataFrame with signals for each date and ticker
        """
        price_data = data.get("price_data")
        if price_data is None:
            raise ValueError("price_data required in data dictionary")

        # Filter date range
        price_range = price_data.loc[
            (price_data.index >= start_date) & (price_data.index <= end_date)
        ]

        if price_range.empty:
            return self._create_empty_signals(universe_tickers, start_date, end_date)

        # Calculate EMAs for all tickers
        signals_df = pd.DataFrame(0.0, index=price_range.index, columns=universe_tickers)

        for ticker in universe_tickers:
            if ticker in price_data.columns:
                ticker_prices = price_data[ticker]
                ticker_signals = self._calculate_ema_signals(ticker_prices, price_range.index)
                signals_df[ticker] = ticker_signals

        return signals_df

    def generate_signal_for_date(
        self,
        data: Dict[str, Any],
        universe_tickers: List[str],
        current_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Generate signal for a single date with state management.

        Args:
            data: Dictionary containing 'price_data' and optionally 'current_prices'
            universe_tickers: List of universe ticker symbols
            current_date: Current trading date

        Returns:
            DataFrame with signals for current date
        """
        price_data = data.get("price_data")
        if price_data is None:
            raise ValueError("price_data required in data dictionary")

        # Create signals DataFrame for current date
        signals = pd.DataFrame(0.0, index=[current_date], columns=universe_tickers)

        for ticker in universe_tickers:
            if ticker in price_data.columns:
                ticker_prices = price_data[ticker]
                # Get price data up to current date
                ticker_history = ticker_prices[ticker_prices.index <= current_date]

                if len(ticker_history) >= max(self.fast_period, self.slow_period):
                    signal = self._calculate_single_ema_signal(ticker_history)
                    signals.loc[current_date, ticker] = signal

                    # Update position state
                    self._position_state[ticker] = signal != 0

        current_signals = signals.loc[current_date]
        if isinstance(current_signals, pd.Series):
            self._previous_signals = current_signals
        else:
            # If it's a DataFrame, take the first row
            self._previous_signals = current_signals.iloc[0] if not current_signals.empty else None
        return signals

    def reset_state(self) -> None:
        """Reset internal state for new backtest runs."""
        self._previous_signals = None
        self._position_state.clear()

    def is_in_position(self) -> bool:
        """Check if currently in any positions."""
        return any(self._position_state.values()) if self._position_state else False

    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration parameters."""
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "signal_period": self.signal_period,
        }

    def _calculate_ema_signals(
        self, price_series: pd.Series, date_range: pd.DatetimeIndex
    ) -> pd.Series:
        """Calculate EMA crossover signals for a price series."""
        # Calculate EMAs
        fast_ema = price_series.ewm(span=self.fast_period).mean()
        slow_ema = price_series.ewm(span=self.slow_period).mean()

        # Generate crossover signals
        crossover = fast_ema - slow_ema
        crossover_signal = crossover.ewm(span=self.signal_period).mean()

        # Create binary signals: 1 for bullish, -1 for bearish, 0 for neutral
        signals = pd.Series(0.0, index=price_series.index)

        # Long when fast EMA > slow EMA and rising
        bullish_condition = (crossover > 0) & (crossover_signal > 0)
        bearish_condition = (crossover < 0) & (crossover_signal < 0)

        signals[bullish_condition] = 1.0
        signals[bearish_condition] = -1.0

        # Reindex to requested date range
        return signals.reindex(date_range).fillna(0.0)

    def _calculate_single_ema_signal(self, price_series: pd.Series) -> float:
        """Calculate EMA signal for the latest data point."""
        if len(price_series) < max(self.fast_period, self.slow_period):
            return 0.0

        # Calculate EMAs
        fast_ema = price_series.ewm(span=self.fast_period).mean().iloc[-1]
        slow_ema = price_series.ewm(span=self.slow_period).mean().iloc[-1]

        # Simple crossover logic
        if fast_ema > slow_ema:
            return 1.0
        elif fast_ema < slow_ema:
            return -1.0
        else:
            return 0.0

    def _create_empty_signals(
        self,
        universe_tickers: List[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Create empty signals DataFrame for a date range."""
        if start_date == end_date:
            index = [start_date]
        else:
            index = pd.date_range(start_date, end_date, freq="D").tolist()

        return pd.DataFrame(0.0, index=index, columns=universe_tickers)


# Register the EMA signal generator with the factory
signal_generator_factory.register_generator("ema_crossover", EmaCrossoverSignalGenerator)
