"""
Signal Generator

Handles trading signal generation logic for UVXY RSI strategy.
Separated from strategy logic to follow Single Responsibility Principle.

This implementation follows the ISignalGenerator interface to eliminate isinstance violations.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ...interfaces.signal_generator_interface import ISignalGenerator, signal_generator_factory

logger = logging.getLogger(__name__)


class UvxySignalGenerator(ISignalGenerator):
    """
    Signal generator for UVXY RSI trading strategy.

    This class is responsible solely for generating trading signals based on
    RSI values and strategy parameters. It handles both single-date and
    date-range signal generation with proper entry/exit logic.

    Implements ISignalGenerator interface for polymorphic signal generation.
    """

    def __init__(self, config=None, rsi_threshold: float = 30.0, holding_period_days: int = 1):
        """
        Initialize signal generator.

        WARNING: Direct instantiation is deprecated. Use signal_generator_factory.create_generator() instead.

        Args:
            config: Configuration dictionary (for factory pattern)
            rsi_threshold: RSI threshold for entry signals (default: 30.0)
            holding_period_days: Number of days to hold position (default: 1)
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
                            "Direct instantiation of UvxySignalGenerator is deprecated. "
                            "Use signal_generator_factory.create_generator('uvxy_rsi', config) instead.",
                            DeprecationWarning,
                            stacklevel=2,
                        )
        finally:
            del frame
        # Handle both factory pattern (config dict) and direct instantiation
        if config is not None:
            self.rsi_threshold = config.get("rsi_threshold", rsi_threshold)
            self.holding_period_days = config.get("holding_period_days", holding_period_days)
        else:
            self.rsi_threshold = rsi_threshold
            self.holding_period_days = holding_period_days

        # State tracking for single-date generation
        self._entry_date: Optional[pd.Timestamp] = None
        self._previous_signal: float = 0.0

    # Interface-compliant methods
    def generate_signals_for_range(
        self,
        data: Dict[str, Any],
        universe_tickers: List[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Interface-compliant method for generating signals over a date range."""
        rsi_series = data.get("rsi_series")
        if rsi_series is None:
            raise ValueError("rsi_series required in data dictionary")
        return self._generate_signals_for_range_impl(
            rsi_series, universe_tickers, start_date, end_date
        )

    def generate_signal_for_date(
        self, data: Dict[str, Any], universe_tickers: List[str], current_date: pd.Timestamp
    ) -> pd.DataFrame:
        """Interface-compliant method for generating signals for a single date."""
        current_rsi = data.get("current_rsi")
        return self._generate_signal_for_date_impl(current_rsi, universe_tickers, current_date)

    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration parameters."""
        return {
            "rsi_threshold": self.rsi_threshold,
            "holding_period_days": self.holding_period_days,
        }

    # Legacy methods for backward compatibility
    def _generate_signals_for_range_impl(
        self,
        rsi_series: pd.Series,
        universe_tickers: List[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Generate signals for a full date range using vectorized operations.

        Args:
            rsi_series: Series of RSI values indexed by date
            universe_tickers: List of universe ticker symbols
            start_date: Start date for signal generation
            end_date: End date for signal generation

        Returns:
            DataFrame with signals for each date and ticker
        """
        # Get trading days in the range
        trading_days = rsi_series.index[
            (rsi_series.index >= start_date) & (rsi_series.index <= end_date)
        ]
        if len(trading_days) == 0:
            return self._create_empty_signals(universe_tickers, start_date, end_date)

        # Reindex RSI to trading days
        rsi_values = rsi_series.reindex(trading_days)

        # Entry: short signal when RSI < threshold
        entry_signal = (rsi_values < self.rsi_threshold) & (~rsi_values.isna())
        entry_dates = trading_days[entry_signal]

        # Initialize signals DataFrame with zeros
        signals = pd.DataFrame(0.0, index=trading_days, columns=universe_tickers, dtype=float)

        # Vectorized signal assignment: For each entry, set signal for entry and holding period
        entry_indices = trading_days.get_indexer(entry_dates)

        # Calculate all holding indices (entry + holding period days)
        hold_indices = []
        for entry_idx in entry_indices:
            for day_offset in range(self.holding_period_days + 1):  # +1 to include entry day
                hold_idx = entry_idx + day_offset
                if 0 <= hold_idx < len(trading_days):
                    hold_indices.append(hold_idx)

        # Remove duplicates and apply signals
        hold_indices = np.unique(hold_indices)
        signal_value = -1.0 / len(universe_tickers)  # Short position equally weighted
        signals.iloc[hold_indices, :] = signal_value

        return signals

    def _generate_signal_for_date_impl(
        self, current_rsi: Optional[float], universe_tickers: List[str], current_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Generate signal for a single date with state management.

        Args:
            current_rsi: RSI value for current date (can be None/NaN)
            universe_tickers: List of universe ticker symbols
            current_date: Current trading date

        Returns:
            DataFrame with signals for current date
        """
        # Create signals DataFrame for current date
        signals = pd.DataFrame(index=[current_date], columns=universe_tickers, dtype=float).fillna(
            0.0
        )

        # Check if we're currently in a position
        if self._entry_date is not None and current_date > self._entry_date:
            # Check if holding period is complete
            days_held = (current_date - self._entry_date).days
            if days_held >= self.holding_period_days:
                # Exit position (close short)
                for ticker in universe_tickers:
                    signals.loc[current_date, ticker] = 0.0
                logger.info(
                    f"Closing UVXY short position on {current_date}: {self.holding_period_days}-day holding period complete"
                )
                self._entry_date = None
                self._previous_signal = 0.0
            else:
                # Continue holding position
                signal_value = -1.0 / len(universe_tickers)
                for ticker in universe_tickers:
                    signals.loc[current_date, ticker] = signal_value
                self._previous_signal = signal_value
        elif self._entry_date is not None:
            # Hold position (entry date)
            signal_value = -1.0 / len(universe_tickers)
            for ticker in universe_tickers:
                signals.loc[current_date, ticker] = signal_value
            self._previous_signal = signal_value
        else:
            # Not in position, check for entry signal
            if (
                current_rsi is not None
                and not np.isnan(current_rsi)
                and current_rsi < self.rsi_threshold
            ):
                # Enter short position
                signal_value = -1.0 / len(universe_tickers)
                for ticker in universe_tickers:
                    signals.loc[current_date, ticker] = signal_value
                logger.info(
                    f"Short signal on {current_date}: RSI = {current_rsi:.2f} < {self.rsi_threshold}"
                )
                self._entry_date = current_date
                self._previous_signal = signal_value

        return signals

    def reset_state(self):
        """Reset internal state for new backtest runs."""
        self._entry_date = None
        self._previous_signal = 0.0

    def is_in_position(self) -> bool:
        """Check if currently in a position."""
        return self._entry_date is not None

    def get_entry_date(self) -> Optional[pd.Timestamp]:
        """Get current entry date if in position."""
        return self._entry_date

    # Legacy methods removed - now using interface methods only

    def _create_empty_signals(
        self, universe_tickers: List[str], start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> pd.DataFrame:
        """Create empty signals DataFrame for a date range."""
        if start_date == end_date:
            index = [start_date]
        else:
            index = pd.date_range(start_date, end_date, freq="D").tolist()

        return pd.DataFrame(index=index, columns=universe_tickers, dtype=float).fillna(0.0)


# Register the UVXY signal generator with the factory
signal_generator_factory.register_generator("uvxy_rsi", UvxySignalGenerator)
