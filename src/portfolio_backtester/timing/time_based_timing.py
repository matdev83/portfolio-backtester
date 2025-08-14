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
        self.offset = config.get("rebalance_offset", 0)

    def _convert_deprecated_frequency(self, freq: str) -> str:
        """Converts deprecated pandas frequency strings to their modern equivalents."""
        freq_upper = freq.upper()
        if freq_upper == "M":
            return "ME"
        if freq_upper == "Q":
            return "QE"
        if freq_upper == "A" or freq_upper == "Y":
            return "YE"
        if freq_upper == "W":
            return "W-MON"
        return freq

    def get_rebalance_dates(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        available_dates: pd.DatetimeIndex,
        strategy_context: "BaseStrategy",
    ) -> pd.DatetimeIndex:
        """Generate rebalance dates based on frequency."""
        # Use pandas to generate the rebalance dates directly
        try:
            freq = self._convert_deprecated_frequency(self.frequency)
            rebalance_dates = pd.bdate_range(start=start_date, end=end_date, freq=freq)
        except ValueError as e:
            # Re-raise with expected error message format for tests
            if "Invalid frequency" in str(e):
                raise ValueError(f"Invalid frequency '{self.frequency}'") from e
            raise

        # Filter out the start date to avoid rebalancing on the first day
        rebalance_dates = rebalance_dates[rebalance_dates > start_date]

        # Apply offset if specified
        if self.offset != 0:
            rebalance_dates = rebalance_dates + pd.DateOffset(days=self.offset)

        # For non-trading days, roll back to the previous available trading day.
        rolled_dates = []
        # Convert to DatetimeIndex if it's a list
        if isinstance(available_dates, list):
            available_dates_idx = pd.DatetimeIndex(available_dates)
        else:
            available_dates_idx = available_dates
            
        for date in rebalance_dates:
            if date in available_dates_idx:
                rolled_dates.append(date)
            else:
                # Find the index for the insertion point to maintain order.
                loc = available_dates_idx.searchsorted(date, side="left")
                if loc > 0:
                    rolled_dates.append(available_dates_idx[loc - 1])

        # Ensure the dates are unique and sorted
        if rolled_dates:
            rebalance_dates = pd.DatetimeIndex(rolled_dates).unique()
        else:
            rebalance_dates = pd.DatetimeIndex([])

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
