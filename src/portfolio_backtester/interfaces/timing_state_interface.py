"""
Interface for timing state management to support dependency inversion principle.

This interface abstracts TimingState dependencies, allowing timing controllers
to depend on abstractions rather than concrete implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Set
import pandas as pd


class ITimingState(ABC):
    """Interface for timing state management operations."""

    @abstractmethod
    def reset(self) -> None:
        """Reset all state for new backtest run."""
        pass

    @abstractmethod
    def update_signal(self, date: pd.Timestamp, weights: pd.Series) -> None:
        """Update state after signal generation."""
        pass

    @abstractmethod
    def update_positions(
        self, date: pd.Timestamp, new_weights: pd.Series, prices: pd.Series
    ) -> None:
        """Update position tracking state."""
        pass

    @abstractmethod
    def get_position_holding_days(self, asset: str, current_date: pd.Timestamp) -> Optional[int]:
        """Get the number of days an asset has been held."""
        pass

    @abstractmethod
    def is_position_held(self, asset: str) -> bool:
        """Check if a position is currently held."""
        pass

    @abstractmethod
    def get_held_assets(self) -> Set[str]:
        """Get set of currently held assets."""
        pass

    @abstractmethod
    def get_consecutive_periods(self, asset: str) -> int:
        """Get consecutive periods for an asset."""
        pass

    @abstractmethod
    def get_position_return(
        self, asset: str, current_price: Optional[float] = None
    ) -> Optional[float]:
        """Calculate current return for a position."""
        pass

    @abstractmethod
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get summary of current portfolio state."""
        pass

    @abstractmethod
    def get_position_statistics(self) -> Dict[str, Any]:
        """Get statistics about position history."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary."""
        pass

    @abstractmethod
    def save_to_file(self, filepath: str) -> None:
        """Save state to JSON file."""
        pass

    # Properties that need to be accessible
    @property
    @abstractmethod
    def last_signal_date(self) -> Optional[pd.Timestamp]:
        """Get last signal date."""
        pass

    @last_signal_date.setter
    @abstractmethod
    def last_signal_date(self, value: Optional[pd.Timestamp]) -> None:
        """Set last signal date."""
        pass

    @property
    @abstractmethod
    def last_weights(self) -> Optional[pd.Series]:
        """Get last weights."""
        pass

    @last_weights.setter
    @abstractmethod
    def last_weights(self, value: Optional[pd.Series]) -> None:
        """Set last weights."""
        pass

    @property
    @abstractmethod
    def scheduled_dates(self) -> Set[pd.Timestamp]:
        """Get scheduled dates."""
        pass

    @scheduled_dates.setter
    @abstractmethod
    def scheduled_dates(self, value: Set[pd.Timestamp]) -> None:
        """Set scheduled dates."""
        pass


class ITimingStateFactory(ABC):
    """Interface for creating timing state instances."""

    @abstractmethod
    def create_timing_state(self, **kwargs) -> ITimingState:
        """Create a timing state instance."""
        pass


def create_timing_state_factory() -> ITimingStateFactory:
    """Create a timing state factory instance."""
    return _TimingStateFactory()


def create_timing_state(**kwargs) -> ITimingState:
    """Create a timing state instance directly."""
    factory = create_timing_state_factory()
    return factory.create_timing_state(**kwargs)


class _TimingStateFactory(ITimingStateFactory):
    """Default factory implementation for timing state."""

    def create_timing_state(self, **kwargs) -> ITimingState:
        """Create a timing state instance."""
        from ..timing.timing_state import TimingState

        return _TimingStateAdapter(TimingState(**kwargs))


class _TimingStateAdapter(ITimingState):
    """Adapter to make TimingState compatible with ITimingState interface."""

    def __init__(self, timing_state):
        """Initialize with a concrete TimingState instance."""
        self._timing_state = timing_state

    def reset(self) -> None:
        """Reset all state for new backtest run."""
        self._timing_state.reset()

    def update_signal(self, date: pd.Timestamp, weights: pd.Series) -> None:
        """Update state after signal generation."""
        self._timing_state.update_signal(date, weights)

    def update_positions(
        self, date: pd.Timestamp, new_weights: pd.Series, prices: pd.Series
    ) -> None:
        """Update position tracking state."""
        self._timing_state.update_positions(date, new_weights, prices)

    def get_position_holding_days(self, asset: str, current_date: pd.Timestamp) -> Optional[int]:
        """Get the number of days an asset has been held."""
        return self._timing_state.get_position_holding_days(asset, current_date)  # type: ignore[no-any-return]

    def is_position_held(self, asset: str) -> bool:
        """Check if a position is currently held."""
        return self._timing_state.is_position_held(asset)  # type: ignore[no-any-return]

    def get_held_assets(self) -> Set[str]:
        """Get set of currently held assets."""
        return self._timing_state.get_held_assets()  # type: ignore[no-any-return]

    def get_consecutive_periods(self, asset: str) -> int:
        """Get consecutive periods for an asset."""
        return self._timing_state.get_consecutive_periods(asset)  # type: ignore[no-any-return]

    def get_position_return(
        self, asset: str, current_price: Optional[float] = None
    ) -> Optional[float]:
        """Calculate current return for a position."""
        return self._timing_state.get_position_return(asset, current_price)  # type: ignore[no-any-return]

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get summary of current portfolio state."""
        return self._timing_state.get_portfolio_summary()  # type: ignore[no-any-return]

    def get_position_statistics(self) -> Dict[str, Any]:
        """Get statistics about position history."""
        return self._timing_state.get_position_statistics()  # type: ignore[no-any-return]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary."""
        return self._timing_state.to_dict()  # type: ignore[no-any-return]

    def save_to_file(self, filepath: str) -> None:
        """Save state to JSON file."""
        self._timing_state.save_to_file(filepath)

    @property
    def last_signal_date(self) -> Optional[pd.Timestamp]:
        """Get last signal date."""
        return self._timing_state.last_signal_date  # type: ignore[no-any-return]

    @last_signal_date.setter
    def last_signal_date(self, value: Optional[pd.Timestamp]) -> None:
        """Set last signal date."""
        self._timing_state.last_signal_date = value

    @property
    def last_weights(self) -> Optional[pd.Series]:
        """Get last weights."""
        return self._timing_state.last_weights  # type: ignore[no-any-return]

    @last_weights.setter
    def last_weights(self, value: Optional[pd.Series]) -> None:
        """Set last weights."""
        self._timing_state.last_weights = value

    @property
    def scheduled_dates(self) -> Set[pd.Timestamp]:
        """Get scheduled dates."""
        return self._timing_state.scheduled_dates  # type: ignore[no-any-return]

    @scheduled_dates.setter
    def scheduled_dates(self, value: Set[pd.Timestamp]) -> None:
        """Set scheduled dates."""
        self._timing_state.scheduled_dates = value

        # Additional backward compatibility properties

    @property
    def position_entry_dates(self):
        """Get position entry dates - backward compatibility."""
        return self._timing_state.position_entry_dates

    @property
    def position_entry_prices(self):
        """Get position entry prices - backward compatibility."""
        return self._timing_state.position_entry_prices

    @property
    def positions(self):
        """Get positions - backward compatibility."""
        return self._timing_state.positions

    @property
    def position_history(self):
        """Get position history - backward compatibility."""
        return self._timing_state.position_history

    @property
    def consecutive_periods(self):
        """Get consecutive periods - backward compatibility."""
        return self._timing_state.consecutive_periods

    @property
    def debug_log(self):
        """Get debug log - backward compatibility."""
        return self._timing_state.debug_log
    
    def add_test_position_history(self, position_entries):
        """Add test position history entries - for testing purposes only."""
        self._timing_state.add_test_position_history(position_entries)
    
    # Additional methods expected by tests
    def enable_debug(self, enabled=True):
        """Enable or disable debug logging."""
        self._timing_state.enable_debug(enabled)
    
    def _log_debug(self, message, data):
        """Internal debug logging method."""
        self._timing_state._log_debug(message, data)
    
    def print_state_summary(self):
        """Print detailed state summary for debugging."""
        self._timing_state.print_state_summary()
    
    def get_position_info(self, asset):
        """Get detailed position information for an asset."""
        return self._timing_state.get_position_info(asset)
    
    @property
    def debug_enabled(self):
        """Get debug enabled state."""
        return self._timing_state.debug_enabled
    
    @property
    def state_version(self):
        """Get state version."""
        return self._timing_state.state_version
    
    def get_debug_log(self, last_n=None):
        """Get debug log entries."""
        return self._timing_state.get_debug_log(last_n)
    
    def clear_debug_log(self):
        """Clear debug log."""
        self._timing_state.clear_debug_log()
