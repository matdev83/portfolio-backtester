"""
Position tracking functionality for timing framework.
Manages position information, updates, and lifecycle tracking.
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
    """Manages position tracking with enhanced position information."""

    def __init__(self, debug_enabled: bool = False):
        """
        Initialize position tracker.

        Args:
            debug_enabled: Enable debug logging
        """
        # Enhanced position tracking
        self.positions: Dict[str, PositionInfo] = {}
        self.position_history: List[Dict[str, Any]] = []

        # Legacy compatibility (maintained for backward compatibility)
        self.position_entry_dates: Dict[str, pd.Timestamp] = {}
        self.position_entry_prices: Dict[str, float] = {}
        self.consecutive_periods: Dict[str, int] = {}

        # Debug support
        self.debug_enabled = debug_enabled
        self.debug_log: List[Dict[str, Any]] = []

    def reset(self):
        """Reset all position tracking state."""
        # Enhanced state
        self.positions.clear()
        self.position_history.clear()

        # Legacy compatibility
        self.position_entry_dates.clear()
        self.position_entry_prices.clear()
        self.consecutive_periods.clear()

        self._log_debug(
            "Position tracking reset",
            {"action": "reset", "timestamp": pd.Timestamp.now()},
        )

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
                    position_changes.extend(
                        self._add_new_position(asset, date, new_weight, current_price)
                    )
                else:
                    # Update existing position
                    position_changes.extend(
                        self._update_existing_position(asset, new_weight, current_price)
                    )
            else:
                # Position exit
                if asset in self.positions:
                    position_changes.extend(self._remove_position(asset, date, current_price))

        # Log position changes
        if position_changes:
            self._log_debug(
                "Position changes",
                {
                    "date": date,
                    "changes": position_changes,
                    "active_positions": len(self.positions),
                },
            )

        return position_changes

    def _add_new_position(
        self, asset: str, date: pd.Timestamp, weight: float, price: Optional[float]
    ) -> List[Dict[str, Any]]:
        """Add a new position to tracking."""
        position_info = PositionInfo(
            entry_date=date,
            entry_price=price if price is not None else 0.0,
            entry_weight=weight,
            current_weight=weight,
            consecutive_periods=1,
            max_weight=weight,
            min_weight=weight,
        )
        self.positions[asset] = position_info

        # Legacy compatibility
        self.position_entry_dates[asset] = date
        if price is not None:
            self.position_entry_prices[asset] = price
        self.consecutive_periods[asset] = 1

        return [{"asset": asset, "action": "entry", "weight": weight, "price": price}]

    def _update_existing_position(
        self, asset: str, new_weight: float, current_price: Optional[float]
    ) -> List[Dict[str, Any]]:
        """Update an existing position."""
        position_info = self.positions[asset]
        old_weight = position_info.current_weight
        position_info.update_weight(new_weight, current_price)
        position_info.consecutive_periods += 1

        # Update legacy compatibility
        self.consecutive_periods[asset] = position_info.consecutive_periods

        changes = []
        if abs(old_weight - new_weight) > 1e-10:
            changes.append(
                {
                    "asset": asset,
                    "action": "weight_change",
                    "old_weight": old_weight,
                    "new_weight": new_weight,
                    "price": current_price,
                }
            )

        return changes

    def _remove_position(
        self, asset: str, date: pd.Timestamp, current_price: Optional[float]
    ) -> List[Dict[str, Any]]:
        """Remove a position from tracking."""
        position_info = self.positions[asset]
        holding_days = (date - position_info.entry_date).days

        # Calculate final return if we have exit price
        final_return = position_info.total_return
        if current_price is not None and position_info.entry_price > 0:
            final_return = (current_price - position_info.entry_price) / position_info.entry_price

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
                "total_return": final_return,
                "final_pnl": position_info.unrealized_pnl,
            }
        )

        # Remove from active positions
        del self.positions[asset]

        # Legacy compatibility
        self.position_entry_dates.pop(asset, None)
        self.position_entry_prices.pop(asset, None)
        self.consecutive_periods.pop(asset, None)

        return [
            {
                "asset": asset,
                "action": "exit",
                "holding_days": holding_days,
                "price": current_price,
            }
        ]

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

    def get_active_positions_count(self) -> int:
        """Get count of active positions."""
        return len(self.positions)

    def get_historical_positions_count(self) -> int:
        """Get count of historical positions."""
        return len(self.position_history)

    def get_position_history(self) -> List[Dict[str, Any]]:
        """Get copy of position history."""
        return self.position_history.copy()

    def enable_debug(self, enabled: bool = True):
        """Enable or disable debug logging."""
        self.debug_enabled = enabled
        if enabled:
            self._log_debug("Debug enabled", {"timestamp": pd.Timestamp.now()})

    def _log_debug(self, message: str, data: Dict[str, Any]):
        """Internal debug logging method."""
        if self.debug_enabled:
            log_entry = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "message": message,
                "data": data,
            }
            self.debug_log.append(log_entry)

            # Keep debug log size manageable
            if len(self.debug_log) > 1000:
                self.debug_log = self.debug_log[-500:]  # Keep last 500 entries

    def get_debug_log(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get debug log entries."""
        if last_n is not None:
            return self.debug_log[-last_n:]
        return self.debug_log.copy()

    def clear_debug_log(self):
        """Clear debug log."""
        self.debug_log.clear()
        self._log_debug("Debug log cleared", {"timestamp": pd.Timestamp.now()})

    def add_test_position_history(self, position_entries: List[Dict[str, Any]]):
        """Add test position history entries - for testing purposes only.

        Args:
            position_entries: List of position history dictionaries to add
        """
        self.position_history.extend(position_entries)
        self._log_debug(
            "Test position history added",
            {"entries_count": len(position_entries), "timestamp": pd.Timestamp.now()},
        )
