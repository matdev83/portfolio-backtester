"""
TimingState dataclass for managing timing-related state across rebalancing periods.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Set
import pandas as pd


@dataclass
class TimingState:
    """Manages timing-related state across rebalancing periods."""
    
    last_signal_date: Optional[pd.Timestamp] = None
    last_weights: Optional[pd.Series] = None
    position_entry_dates: Dict[str, pd.Timestamp] = field(default_factory=dict)
    position_entry_prices: Dict[str, float] = field(default_factory=dict)
    scheduled_dates: Set[pd.Timestamp] = field(default_factory=set)
    consecutive_periods: Dict[str, int] = field(default_factory=dict)  # For stateful strategies
    
    def reset(self):
        """Reset all state for new backtest run."""
        self.last_signal_date = None
        self.last_weights = None
        self.position_entry_dates.clear()
        self.position_entry_prices.clear()
        self.scheduled_dates.clear()
        self.consecutive_periods.clear()
    
    def update_signal(self, date: pd.Timestamp, weights: pd.Series):
        """Update state after signal generation."""
        self.last_signal_date = date
        self.last_weights = weights.copy() if weights is not None else None
    
    def update_positions(self, date: pd.Timestamp, new_weights: pd.Series, prices: pd.Series):
        """Update position tracking when weights change."""
        # Get previous weights before they were updated by update_signal
        # We need to check what the weights were before this signal
        prev_weights = pd.Series(0.0, index=new_weights.index)
        
        # Check current position status to determine previous weights
        for asset in new_weights.index:
            if asset in self.position_entry_dates:
                # Asset was already in a position, so previous weight was non-zero
                # We'll use a placeholder value since we don't track exact previous weights
                prev_weights[asset] = 1.0 if asset in self.position_entry_dates else 0.0
            else:
                prev_weights[asset] = 0.0
        
        # Track new positions
        for asset in new_weights.index:
            prev_weight = prev_weights.get(asset, 0.0)
            new_weight = new_weights.get(asset, 0.0)
            
            # New position entered (any non-zero weight when previous was zero)
            if abs(prev_weight) < 1e-10 and abs(new_weight) > 1e-10:
                self.position_entry_dates[asset] = date
                if asset in prices.index and pd.notna(prices[asset]):
                    self.position_entry_prices[asset] = prices[asset]
            
            # Position closed (weight becomes zero when it was non-zero)
            elif abs(prev_weight) > 1e-10 and abs(new_weight) < 1e-10:
                self.position_entry_dates.pop(asset, None)
                self.position_entry_prices.pop(asset, None)
    
    def get_position_holding_days(self, asset: str, current_date: pd.Timestamp) -> Optional[int]:
        """Get the number of days an asset has been held."""
        if asset not in self.position_entry_dates:
            return None
        
        entry_date = self.position_entry_dates[asset]
        return (current_date - entry_date).days
    
    def is_position_held(self, asset: str) -> bool:
        """Check if a position is currently held."""
        return asset in self.position_entry_dates
    
    def get_held_assets(self) -> Set[str]:
        """Get set of currently held assets."""
        return set(self.position_entry_dates.keys())