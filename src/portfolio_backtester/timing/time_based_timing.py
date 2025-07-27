"""
Traditional time-based rebalancing timing controller.
"""

from typing import Dict, Any, TYPE_CHECKING
import pandas as pd

from .timing_controller import TimingController

if TYPE_CHECKING:
    from ..strategies.base_strategy import BaseStrategy


class TimeBasedTiming(TimingController):
    """Traditional time-based rebalancing (monthly, quarterly, etc.)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.frequency = config.get('rebalance_frequency', 'M')
        self.offset = config.get('rebalance_offset', 0)  # Days offset from period end
    
    def get_rebalance_dates(
        self, 
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        available_dates: pd.DatetimeIndex,
        strategy_context: 'BaseStrategy'
    ) -> pd.DatetimeIndex:
        """Generate rebalance dates based on frequency."""
        # Convert legacy frequencies to new pandas format
        freq_mapping = {
            'M': 'ME',  # Month end
            'Q': 'QE',  # Quarter end  
            'A': 'YE',  # Year end
            'Y': 'YE',  # Year end (alias)
            'W': 'W',   # Weekly (unchanged)
            'D': 'D'    # Daily (unchanged)
        }
        freq = freq_mapping.get(self.frequency.upper(), self.frequency)
        
        # Generate base dates
        try:
            base_dates = pd.date_range(start_date, end_date, freq=freq)
        except ValueError as e:
            raise ValueError(f"Invalid frequency '{self.frequency}': {e}")
        
        # Apply offset if specified
        if self.offset != 0:
            base_dates = base_dates + pd.Timedelta(days=self.offset)
        
        # Filter to only include available trading dates
        rebalance_dates = []
        for date in base_dates:
            # Find nearest available trading date
            # Check if the target date is already a business day
            if date in available_dates:
                nearest_date = date
            else:
                # For month-end frequencies (ME), prefer last business day of period
                # For other frequencies, prefer next business day if target is not available
                if freq in ['ME', 'M']:
                    # Month-end: prefer next business day if within end_date range, otherwise last business day
                    future_dates = available_dates[available_dates >= date]
                    if len(future_dates) > 0 and future_dates.min() <= end_date:
                        nearest_date = future_dates.min()
                    else:
                        # If no future dates within range, use last business day
                        past_dates = available_dates[available_dates <= date]
                        if len(past_dates) > 0:
                            nearest_date = past_dates.max()
                        else:
                            continue
                else:
                    # Other frequencies: prefer next business day on or after target
                    future_dates = available_dates[available_dates >= date]
                    if len(future_dates) > 0:
                        nearest_date = future_dates.min()
                    else:
                        # If no future dates, find last available trading date
                        past_dates = available_dates[available_dates < date]
                        if len(past_dates) > 0:
                            nearest_date = past_dates.max()
                        else:
                            continue
            
            # Only include if the nearest date is within reasonable range
            if start_date <= nearest_date <= end_date:
                rebalance_dates.append(nearest_date)
        
        # Remove duplicates and sort
        rebalance_dates = sorted(set(rebalance_dates))
        
        # Store scheduled dates in timing state
        scheduled_dates_set = set(rebalance_dates)
        self.timing_state.scheduled_dates = scheduled_dates_set
        
        return pd.DatetimeIndex(rebalance_dates)
    
    def should_generate_signal(
        self, 
        current_date: pd.Timestamp,
        strategy_context: 'BaseStrategy'
    ) -> bool:
        """For time-based timing, signals are only generated on rebalance dates."""
        return current_date in self.timing_state.scheduled_dates