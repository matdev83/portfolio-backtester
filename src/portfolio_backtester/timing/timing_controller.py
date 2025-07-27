"""
Abstract base class for timing control strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING
import pandas as pd

from .timing_state import TimingState

if TYPE_CHECKING:
    from ..strategies.base_strategy import BaseStrategy


class TimingController(ABC):
    """Abstract base class for timing control strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.timing_state = TimingState()
    
    @abstractmethod
    def get_rebalance_dates(
        self, 
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        available_dates: pd.DatetimeIndex,
        strategy_context: 'BaseStrategy'
    ) -> pd.DatetimeIndex:
        """Return dates when the strategy should generate signals."""
        pass
    
    @abstractmethod
    def should_generate_signal(
        self, 
        current_date: pd.Timestamp,
        strategy_context: 'BaseStrategy'
    ) -> bool:
        """Determine if a signal should be generated on the current date."""
        pass
    
    def reset_state(self):
        """Reset timing state for new backtest run."""
        self.timing_state.reset()
    
    def get_timing_state(self) -> TimingState:
        """Get the current timing state."""
        return self.timing_state
    
    def update_signal_state(self, date: pd.Timestamp, weights: pd.Series):
        """Update timing state after signal generation."""
        self.timing_state.update_signal(date, weights)
    
    def update_position_state(self, date: pd.Timestamp, new_weights: pd.Series, prices: pd.Series):
        """Update position tracking state."""
        self.timing_state.update_positions(date, new_weights, prices)