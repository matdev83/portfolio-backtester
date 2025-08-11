"""
Traditional time-based rebalancing timing controller.
"""

from typing import Dict, Any, TYPE_CHECKING, Optional
import pandas as pd

from .timing_controller import TimingController
from ..interfaces.timing_state_interface import ITimingState

if TYPE_CHECKING:
    from ..strategies._core.base.base_strategy import BaseStrategy


class TimeBasedTiming(TimingController):
    """Traditional time-based rebalancing (monthly, quarterly, etc.)."""

    def __init__(self, config: Dict[str, Any], timing_state: Optional[ITimingState] = None):
        super().__init__(config, timing_state)
        self.frequency = config.get("rebalance_frequency", "M")
        self.offset = config.get("rebalance_offset", 0)  # Days offset from period end

    def get_rebalance_dates(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        available_dates: pd.DatetimeIndex,
        strategy_context: "BaseStrategy",
    ) -> pd.DatetimeIndex:
        """Generate rebalance dates based on frequency."""
        # Use pandas to generate the rebalance dates directly
        rebalance_dates = pd.bdate_range(start=start_date, end=end_date, freq=self.frequency)

        # Apply offset if specified
        if self.offset != 0:
            rebalance_dates = rebalance_dates + pd.DateOffset(days=self.offset)

        # Filter to only include available trading dates
        rebalance_dates = available_dates.intersection(rebalance_dates)

        # Store scheduled dates in timing state
        self.timing_state.scheduled_dates = set(rebalance_dates)

        return rebalance_dates

    def should_generate_signal(
        self, current_date: pd.Timestamp, strategy_context: "BaseStrategy"
    ) -> bool:
        """For time-based timing, signals are only generated on rebalance dates."""
        decision = current_date in self.timing_state.scheduled_dates

        # Optional logging
        if self.config.get("enable_logging", False):
            from .timing_logger import log_signal_generation

            reason = "scheduled rebalance date" if decision else "not on rebalance schedule"
            log_signal_generation(
                strategy_context.__class__.__name__,
                current_date,
                decision,
                reason,
                controller="TimeBasedTiming",
            )

        return decision
