"""
Position tracking functionality for timing framework.
Handles entry, exit, weight changes, and position lifecycle management.
"""

import pandas as pd
from typing import Dict, Optional, Set, Any, List
from dataclasses import dataclass


@dataclass
class PositionInfo:
    """Detailed information about a position."""

    entry_date: pd.Timestamp
    entry_price: float
    entry_weight: float
    current_weight: float
    consecutive_periods: int = 0
    max_weight: float = 0.0
    min_weight: float = 0.0
    total_return: float = 0.0
    unrealized_pnl: float = 0.0

    def update_weight(self, new_weight: float, current_price: Optional[float] = None):
        """Update position weight and related metrics."""
        self.current_weight = new_weight
        self.max_weight = max(self.max_weight, new_weight)
        if self.min_weight == 0.0:  # First update
            self.min_weight = new_weight
        else:
            self.min_weight = min(self.min_weight, new_weight)

        if current_price is not None and self.entry_price > 0:
            self.total_return = (current_price - self.entry_price) / self.entry_price
            self.unrealized_pnl = self.current_weight * self.total_return


class PositionTracker:
    """Manages position tracking with detailed state management."""

    def __init__(self):
        """Initialize position tracker."""
        # Enhanced position tracking
        self.positions: Dict[str, PositionInfo] = {}
        self.position_history: List[Dict[str, Any]] = []

        # Legacy compatibility (maintained for backward compatibility)
        self.position_entry_dates: Dict[str, pd.Timestamp] = {}
        self.position_entry_prices: Dict[str, float] = {}
        self.consecutive_periods: Dict[str, int] = {}

    def reset(self):
        """Reset all position tracking state."""
        # Enhanced state
        self.positions.clear()
        self.position_history.clear()

        # Legacy compatibility
        self.position_entry_dates.clear()
        self.position_entry_prices.clear()
        self.consecutive_periods.clear()

    def update_positions(
        self, date: pd.Timestamp, new_weights: pd.Series, prices: pd.Series
    ) -> List[Dict[str, Any]]:
        """
        Update position tracking with new weights and prices.

        Args:
            date: Current date
            new_weights: New position weights
            prices: Current asset prices

        Returns:
            List of position changes for logging
        """
        position_changes = []

        # Process each asset in new weights
        for asset in new_weights.index:
            new_weight = new_weights.get(asset, 0.0)
            current_price = prices.get(asset) if asset in prices.index else None

            if abs(new_weight) > 1e-10:  # New or existing position
                if asset not in self.positions:
                    # New position entry
                    position_info = PositionInfo(
                        entry_date=date,
                        entry_price=current_price if current_price is not None else 0.0,
                        entry_weight=new_weight,
                        current_weight=new_weight,
                        consecutive_periods=1,
                        max_weight=new_weight,
                        min_weight=new_weight,
                    )
                    self.positions[asset] = position_info

                    # Legacy compatibility
                    self.position_entry_dates[asset] = date
                    if current_price is not None:
                        self.position_entry_prices[asset] = current_price
                    self.consecutive_periods[asset] = 1

                    position_changes.append(
                        {
                            "asset": asset,
                            "action": "entry",
                            "weight": new_weight,
                            "price": current_price,
                        }
                    )

                else:
                    # Update existing position
                    position_info = self.positions[asset]
                    old_weight = position_info.current_weight
                    position_info.update_weight(new_weight, current_price)
                    position_info.consecutive_periods += 1

                    # Update legacy compatibility
                    self.consecutive_periods[asset] = position_info.consecutive_periods

                    if abs(old_weight - new_weight) > 1e-10:
                        position_changes.append(
                            {
                                "asset": asset,
                                "action": "weight_change",
                                "old_weight": old_weight,
                                "new_weight": new_weight,
                                "price": current_price,
                            }
                        )

            else:
                # Position exit
                if asset in self.positions:
                    position_info = self.positions[asset]
                    holding_days = (date - position_info.entry_date).days

                    # Record position history before removal
                    self.position_history.append(
                        {
                            "asset": asset,
                            "entry_date": position_info.entry_date,
                            "exit_date": date,
                            "holding_days": holding_days,
                            "entry_price": position_info.entry_price,
                            "exit_price": current_price,
                            "entry_weight": position_info.entry_weight,
                            "max_weight": position_info.max_weight,
                            "min_weight": position_info.min_weight,
                            "consecutive_periods": position_info.consecutive_periods,
                            "total_return": position_info.total_return,
                            "final_pnl": position_info.unrealized_pnl,
                        }
                    )

                    # Remove from active positions
                    del self.positions[asset]

                    # Legacy compatibility
                    self.position_entry_dates.pop(asset, None)
                    self.position_entry_prices.pop(asset, None)
                    self.consecutive_periods.pop(asset, None)

                    position_changes.append(
                        {
                            "asset": asset,
                            "action": "exit",
                            "holding_days": holding_days,
                            "price": current_price,
                        }
                    )

        return position_changes

    def get_position_holding_days(self, asset: str, current_date: pd.Timestamp) -> Optional[int]:
        """Get the number of days an asset has been held."""
        if asset in self.positions:
            return (current_date - self.positions[asset].entry_date).days
        elif asset in self.position_entry_dates:  # Legacy compatibility
            return (current_date - self.position_entry_dates[asset]).days
        return None

    def is_position_held(self, asset: str) -> bool:
        """Check if a position is currently held."""
        return asset in self.positions or asset in self.position_entry_dates

    def get_held_assets(self) -> Set[str]:
        """Get set of currently held assets."""
        return set(self.positions.keys()) | set(self.position_entry_dates.keys())

    def get_position_info(self, asset: str) -> Optional[PositionInfo]:
        """Get detailed position information for an asset."""
        return self.positions.get(asset)

    def get_consecutive_periods(self, asset: str) -> int:
        """Get consecutive periods for an asset (enhanced version)."""
        if asset in self.positions:
            return self.positions[asset].consecutive_periods
        return self.consecutive_periods.get(asset, 0)

    def get_position_return(
        self, asset: str, current_price: Optional[float] = None
    ) -> Optional[float]:
        """Calculate current return for a position."""
        if asset not in self.positions:
            return None

        position_info = self.positions[asset]
        if position_info.entry_price <= 0:
            return None

        if current_price is not None:
            return (current_price - position_info.entry_price) / position_info.entry_price

        return position_info.total_return

    def get_active_positions(self) -> Dict[str, PositionInfo]:
        """Get all active positions."""
        return self.positions.copy()

    def get_position_history(self) -> List[Dict[str, Any]]:
        """Get position history."""
        return self.position_history.copy()

    def get_position_count(self) -> int:
        """Get count of active positions."""
        return len(self.positions)

    def get_total_weight(self) -> float:
        """Get total weight of all active positions."""
        return sum(pos.current_weight for pos in self.positions.values())

    def get_assets_list(self) -> List[str]:
        """Get list of assets in active positions."""
        return list(self.positions.keys())
