"""
Interface for timing controller base functionality to support dependency inversion principle.

This interface abstracts parent class dependencies, allowing timing controllers
to depend on abstractions rather than concrete parent class implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TYPE_CHECKING
import pandas as pd

from .timing_state_interface import ITimingState

if TYPE_CHECKING:
    from ..strategies._core.base.base_strategy import BaseStrategy


class ITimingBase(ABC):
    """Interface for base timing controller functionality."""

    @abstractmethod
    def __init__(self, config: Dict[str, Any], timing_state: Optional[ITimingState] = None):
        """Initialize timing controller with configuration and optional timing state."""
        pass

    @abstractmethod
    def get_rebalance_dates(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        available_dates: pd.DatetimeIndex,
        strategy_context: "BaseStrategy",
    ) -> pd.DatetimeIndex:
        """Return dates when the strategy should generate signals."""
        pass

    @abstractmethod
    def should_generate_signal(
        self, current_date: pd.Timestamp, strategy_context: "BaseStrategy"
    ) -> bool:
        """Determine if a signal should be generated on the current date."""
        pass

    @abstractmethod
    def reset_state(self) -> None:
        """Reset timing state for new backtest run."""
        pass

    @abstractmethod
    def get_timing_state(self) -> ITimingState:
        """Get the current timing state."""
        pass

    @abstractmethod
    def update_signal_state(self, date: pd.Timestamp, weights: pd.Series) -> None:
        """Update timing state after signal generation."""
        pass

    @abstractmethod
    def update_position_state(
        self, date: pd.Timestamp, new_weights: pd.Series, prices: pd.Series
    ) -> None:
        """Update position tracking state."""
        pass

    @property
    @abstractmethod
    def config(self) -> Dict[str, Any]:
        """Get timing controller configuration."""
        pass


class ITimingBaseFactory(ABC):
    """Interface for creating timing base instances."""

    @abstractmethod
    def create_timing_base(
        self, config: Dict[str, Any], timing_state: Optional[ITimingState] = None
    ) -> ITimingBase:
        """Create a timing base instance."""
        pass


def create_timing_base_factory() -> ITimingBaseFactory:
    """Create a timing base factory instance."""
    return _TimingBaseFactory()


def create_timing_base(
    config: Dict[str, Any], timing_state: Optional[ITimingState] = None
) -> ITimingBase:
    """Create a timing base instance directly."""
    factory = create_timing_base_factory()
    return factory.create_timing_base(config, timing_state)


class _TimingBaseFactory(ITimingBaseFactory):
    """Default factory implementation for timing base."""

    def create_timing_base(
        self, config: Dict[str, Any], timing_state: Optional[ITimingState] = None
    ) -> ITimingBase:
        """Create a timing base instance."""
        from .timing_state_interface import create_timing_state

        # Create adapter that wraps TimingController
        if timing_state is None:
            timing_state = create_timing_state()

        return _TimingBaseAdapter(config, timing_state)


class _TimingBaseAdapter(ITimingBase):
    """Adapter to provide ITimingBase interface for TimingController functionality."""

    def __init__(self, config: Dict[str, Any], timing_state: Optional[ITimingState] = None):
        """Initialize timing base adapter."""
        self._config = config
        self._timing_state = timing_state or create_timing_state()

    def get_rebalance_dates(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        available_dates: pd.DatetimeIndex,
        strategy_context: "BaseStrategy",
    ) -> pd.DatetimeIndex:
        """Return dates when the strategy should generate signals."""
        # This is abstract and should be implemented by subclasses
        # For the adapter, we provide a default implementation
        raise NotImplementedError("get_rebalance_dates must be implemented by subclass")

    def should_generate_signal(
        self, current_date: pd.Timestamp, strategy_context: "BaseStrategy"
    ) -> bool:
        """Determine if a signal should be generated on the current date."""
        # This is abstract and should be implemented by subclasses
        # For the adapter, we provide a default implementation
        raise NotImplementedError("should_generate_signal must be implemented by subclass")

    def reset_state(self) -> None:
        """Reset timing state for new backtest run."""
        self._timing_state.reset()

    def get_timing_state(self) -> ITimingState:
        """Get the current timing state."""
        return self._timing_state

    def update_signal_state(self, date: pd.Timestamp, weights: pd.Series) -> None:
        """Update timing state after signal generation."""
        self._timing_state.update_signal(date, weights)

    def update_position_state(
        self, date: pd.Timestamp, new_weights: pd.Series, prices: pd.Series
    ) -> None:
        """Update position tracking state."""
        self._timing_state.update_positions(date, new_weights, prices)

    @property
    def config(self) -> Dict[str, Any]:
        """Get timing controller configuration."""
        return self._config


def create_signal_based_timing(config: Dict[str, Any], timing_state: Optional[ITimingState] = None):
    """Create a SignalBasedTiming instance using interface factories."""
    from .timing_state_interface import create_timing_state
    from ..timing.signal_based_timing import SignalBasedTiming

    # Create timing state using interface factory if not provided
    if timing_state is None:
        timing_state = create_timing_state()

    # Create SignalBasedTiming with dependency injection
    return SignalBasedTiming(config, timing_state)


def create_timing_state():
    """Create a timing state instance - imported locally to avoid circular imports."""
    from .timing_state_interface import create_timing_state as _create_timing_state

    return _create_timing_state()
