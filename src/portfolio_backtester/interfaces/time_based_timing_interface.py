"""
Interface for time-based timing operations to support dependency inversion principle.

This interface abstracts TimeBasedTiming dependencies, allowing adaptive and momentum
timing controllers to depend on abstractions rather than concrete implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING
import pandas as pd

from .timing_base_interface import ITimingBase

if TYPE_CHECKING:
    from ..strategies._core.base.base_strategy import BaseStrategy


class ITimeBasedTiming(ITimingBase):
    """Interface for time-based timing operations."""

    @abstractmethod
    def get_frequency(self) -> str:
        """Get the rebalance frequency."""
        pass

    @abstractmethod
    def get_offset(self) -> int:
        """Get the rebalance offset in days."""
        pass

    @abstractmethod
    def set_frequency(self, frequency: str) -> None:
        """Set the rebalance frequency."""
        pass

    @abstractmethod
    def set_offset(self, offset: int) -> None:
        """Set the rebalance offset in days."""
        pass

    @abstractmethod
    def generate_base_dates(
        self, start_date: pd.Timestamp, end_date: pd.Timestamp, frequency: str
    ) -> pd.DatetimeIndex:
        """Generate base rebalance dates based on frequency."""
        pass

    @abstractmethod
    def find_nearest_trading_dates(
        self, target_dates: pd.DatetimeIndex, available_dates: pd.DatetimeIndex
    ) -> pd.DatetimeIndex:
        """Find nearest trading dates for target dates."""
        pass


class ITimeBasedTimingFactory(ABC):
    """Interface for creating time-based timing instances."""

    @abstractmethod
    def create_time_based_timing(self, config: Dict[str, Any]) -> ITimeBasedTiming:
        """Create a time-based timing instance."""
        pass


def create_time_based_timing_factory() -> ITimeBasedTimingFactory:
    """Create a time-based timing factory instance."""
    return _TimeBasedTimingFactory()


def create_time_based_timing(config: Dict[str, Any]) -> ITimeBasedTiming:
    """Create a time-based timing instance directly."""
    factory = create_time_based_timing_factory()
    return factory.create_time_based_timing(config)


class _TimeBasedTimingFactory(ITimeBasedTimingFactory):
    """Default factory implementation for time-based timing."""

    def create_time_based_timing(self, config: Dict[str, Any]) -> ITimeBasedTiming:
        """Create a time-based timing instance."""
        from ..timing.time_based_timing import TimeBasedTiming
        from .timing_state_interface import create_timing_state

        # Create timing state using interface factory
        timing_state = create_timing_state()
        timing_controller = TimeBasedTiming(config, timing_state)

        return _TimeBasedTimingAdapter(timing_controller)


class _TimeBasedTimingAdapter(ITimeBasedTiming):
    """Adapter to make TimeBasedTiming compatible with ITimeBasedTiming interface."""

    def __init__(self, time_based_timing):
        """Initialize with a concrete TimeBasedTiming instance."""
        self._time_based_timing = time_based_timing

    def get_rebalance_dates(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        available_dates: pd.DatetimeIndex,
        strategy_context: "BaseStrategy",
    ) -> pd.DatetimeIndex:
        """Return dates when the strategy should generate signals."""
        return self._time_based_timing.get_rebalance_dates(  # type: ignore[no-any-return]
            start_date, end_date, available_dates, strategy_context
        )

    def should_generate_signal(
        self, current_date: pd.Timestamp, strategy_context: "BaseStrategy"
    ) -> bool:
        """Determine if a signal should be generated on the current date."""
        return self._time_based_timing.should_generate_signal(current_date, strategy_context)  # type: ignore[no-any-return]

    def reset_state(self) -> None:
        """Reset timing state for new backtest run."""
        self._time_based_timing.reset_state()

    def get_timing_state(self):
        """Get the current timing state."""
        return self._time_based_timing.get_timing_state()

    def update_signal_state(self, date: pd.Timestamp, weights: pd.Series) -> None:
        """Update timing state after signal generation."""
        self._time_based_timing.update_signal_state(date, weights)

    def update_position_state(
        self, date: pd.Timestamp, new_weights: pd.Series, prices: pd.Series
    ) -> None:
        """Update position tracking state."""
        self._time_based_timing.update_position_state(date, new_weights, prices)

    @property
    def config(self) -> Dict[str, Any]:
        """Get timing controller configuration."""
        return self._time_based_timing.config  # type: ignore[no-any-return]

    def get_frequency(self) -> str:
        """Get the rebalance frequency."""
        return self._time_based_timing.frequency  # type: ignore[no-any-return]

    def get_offset(self) -> int:
        """Get the rebalance offset in days."""
        return self._time_based_timing.offset  # type: ignore[no-any-return]

    def set_frequency(self, frequency: str) -> None:
        """Set the rebalance frequency."""
        self._time_based_timing.frequency = frequency

    def set_offset(self, offset: int) -> None:
        """Set the rebalance offset in days."""
        self._time_based_timing.offset = offset

    def generate_base_dates(
        self, start_date: pd.Timestamp, end_date: pd.Timestamp, frequency: str
    ) -> pd.DatetimeIndex:
        """Generate base rebalance dates based on frequency."""
        # Map legacy frequencies to new pandas format
        freq_mapping = {
            "M": "ME",  # Month end
            "Q": "QE",  # Quarter end
            "A": "YE",  # Year end
            "Y": "YE",  # Year end (alias)
            "W": "W",  # Weekly (unchanged)
            "D": "D",  # Daily (unchanged)
        }
        mapped_freq = freq_mapping.get(frequency.upper(), frequency)

        try:
            return pd.date_range(start_date, end_date, freq=mapped_freq)
        except ValueError as e:
            raise ValueError(f"Invalid frequency '{frequency}': {e}")

    def find_nearest_trading_dates(
        self, target_dates: pd.DatetimeIndex, available_dates: pd.DatetimeIndex
    ) -> pd.DatetimeIndex:
        """Find nearest trading dates for target dates."""
        result_dates = []

        for target_date in target_dates:
            if target_date in available_dates:
                result_dates.append(target_date)
            else:
                # Find nearest available trading date
                future_dates = available_dates[available_dates >= target_date]
                if len(future_dates) > 0:
                    result_dates.append(future_dates.min())
                else:
                    # If no future dates, use last available trading date
                    past_dates = available_dates[available_dates < target_date]
                    if len(past_dates) > 0:
                        result_dates.append(past_dates.max())

        return pd.DatetimeIndex(sorted(set(result_dates)))
