"""
Signal-based timing controller for custom timing based on market conditions.
"""

from typing import Dict, Any, TYPE_CHECKING
import pandas as pd

from .timing_controller import TimingController

if TYPE_CHECKING:
    from ..strategies.base_strategy import BaseStrategy


class SignalBasedTiming(TimingController):
    """Custom timing based on market conditions and strategy signals."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.scan_frequency = config.get('scan_frequency', 'D')  # How often to check for signals
        self.max_holding_period = config.get('max_holding_period', None)  # Max days to hold
        self.min_holding_period = config.get('min_holding_period', 1)  # Min days to hold
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate signal-based timing configuration."""
        # Validate scan frequency
        valid_frequencies = ['D', 'W', 'M']
        if self.scan_frequency not in valid_frequencies:
            raise ValueError(f"Invalid scan_frequency '{self.scan_frequency}'. Must be one of {valid_frequencies}")
        
        # Validate holding periods
        if not isinstance(self.min_holding_period, int) or self.min_holding_period < 1:
            raise ValueError(f"min_holding_period must be a positive integer, got {self.min_holding_period}")
        
        if self.max_holding_period is not None:
            if not isinstance(self.max_holding_period, int) or self.max_holding_period < 1:
                raise ValueError(f"max_holding_period must be a positive integer, got {self.max_holding_period}")
            
            if self.min_holding_period > self.max_holding_period:
                raise ValueError(f"min_holding_period ({self.min_holding_period}) cannot exceed max_holding_period ({self.max_holding_period})")
    
    def get_rebalance_dates(
        self, 
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        available_dates: pd.DatetimeIndex,
        strategy_context: 'BaseStrategy'
    ) -> pd.DatetimeIndex:
        """Scan all available dates for potential signals based on scan frequency."""
        # Filter available dates to the specified range
        date_range = available_dates[(available_dates >= start_date) & (available_dates <= end_date)]
        
        if self.scan_frequency == 'D':
            # Daily scanning - return all available dates
            scan_dates = date_range
        else:
            # For other frequencies, generate scan dates and find nearest trading dates
            freq_mapping = {
                'W': 'W',   # Weekly
                'M': 'ME'   # Month end
            }
            freq = freq_mapping.get(self.scan_frequency, self.scan_frequency)
            
            try:
                base_scan_dates = pd.date_range(start_date, end_date, freq=freq)
            except ValueError as e:
                raise ValueError(f"Invalid scan_frequency '{self.scan_frequency}': {e}")
            
            # Find nearest available trading dates for each scan date
            scan_dates = []
            for scan_date in base_scan_dates:
                # Find nearest available trading date on or after the scan date
                future_dates = available_dates[available_dates >= scan_date]
                if len(future_dates) > 0:
                    nearest_date = future_dates.min()
                    if nearest_date <= end_date:
                        scan_dates.append(nearest_date)
            
            scan_dates = pd.DatetimeIndex(sorted(set(scan_dates)))
        
        # Store scheduled dates in timing state for reference
        self.timing_state.scheduled_dates = set(scan_dates)
        
        return scan_dates
    
    def should_generate_signal(
        self, 
        current_date: pd.Timestamp,
        strategy_context: 'BaseStrategy'
    ) -> bool:
        """Check if conditions are met for signal generation."""
        # For daily scan frequency, if no scheduled dates are set, assume all dates are valid
        if self.scan_frequency == 'D' and not self.timing_state.scheduled_dates:
            # All dates are potential scan dates for daily frequency
            pass
        else:
            # First check if this is a scheduled scan date
            if current_date not in self.timing_state.scheduled_dates:
                return False
        
        # Check minimum holding period constraint
        if self._is_within_min_holding_period(current_date):
            return False
        
        # Check if maximum holding period forces a rebalance
        if self._should_force_rebalance(current_date):
            return True
        
        # Otherwise, allow signal generation (strategy will make final decision)
        decision = True

        if self.config.get('enable_logging', False):
            from .timing_logger import log_signal_generation
            if self._is_within_min_holding_period(current_date):
                reason = 'within min_holding_period'
            elif self._should_force_rebalance(current_date):
                reason = 'forced by max_holding_period'
            else:
                reason = 'scheduled scan date'
            log_signal_generation(strategy_context.__class__.__name__, current_date, decision, reason,
                                  controller='SignalBasedTiming')

        return decision
    
    def _is_within_min_holding_period(self, current_date: pd.Timestamp) -> bool:
        """Check if we're within the minimum holding period since last signal."""
        if self.timing_state.last_signal_date is None:
            return False
        
        days_since_last = (current_date - self.timing_state.last_signal_date).days
        return days_since_last < self.min_holding_period
    
    def _should_force_rebalance(self, current_date: pd.Timestamp) -> bool:
        """Check if maximum holding period forces a rebalance."""
        if self.max_holding_period is None or self.timing_state.last_signal_date is None:
            return False
        
        days_held = (current_date - self.timing_state.last_signal_date).days
        return days_held >= self.max_holding_period
    
    def get_days_since_last_signal(self, current_date: pd.Timestamp) -> int:
        """Get the number of days since the last signal was generated."""
        if self.timing_state.last_signal_date is None:
            return 0
        return (current_date - self.timing_state.last_signal_date).days
    
    def get_position_holding_days(self, asset: str, current_date: pd.Timestamp) -> int:
        """Get the number of days an asset position has been held."""
        return self.timing_state.get_position_holding_days(asset, current_date) or 0
    
    def is_position_held(self, asset: str) -> bool:
        """Check if a position is currently held."""
        return self.timing_state.is_position_held(asset)
    
    def get_held_assets(self) -> set:
        """Get set of currently held assets."""
        return self.timing_state.get_held_assets()
    
    def can_enter_position(self, current_date: pd.Timestamp) -> bool:
        """Check if new positions can be entered based on timing constraints."""
        return not self._is_within_min_holding_period(current_date)
    
    def can_exit_position(self, asset: str, current_date: pd.Timestamp) -> bool:
        """Check if a specific position can be exited based on timing constraints."""
        if not self.is_position_held(asset):
            return False
        
        holding_days = self.get_position_holding_days(asset, current_date)
        return holding_days >= self.min_holding_period
    
    def must_exit_position(self, asset: str, current_date: pd.Timestamp) -> bool:
        """Check if a position must be exited due to maximum holding period."""
        if not self.is_position_held(asset) or self.max_holding_period is None:
            return False
        
        holding_days = self.get_position_holding_days(asset, current_date)
        return holding_days >= self.max_holding_period