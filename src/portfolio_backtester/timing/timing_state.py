"""
TimingState dataclass for managing timing-related state across rebalancing periods.
Refactored version using SOLID principles with facade pattern for backward compatibility.
"""

import pandas as pd
from typing import Dict, Optional, Set, Any, List
from dataclasses import dataclass, field

from .state_management import PositionTracker, StateStatistics, StateSerializer, PositionInfo


@dataclass
class TimingState:
    """
    Enhanced timing state management with advanced position tracking.

    Refactored facade that delegates to specialized classes following SOLID principles:
    - PositionTracker: Manages position information and lifecycle tracking
    - StateStatistics: Provides statistical analysis and portfolio summaries
    - StateSerializer: Handles state persistence and serialization
    """

    # Basic timing state
    last_signal_date: Optional[pd.Timestamp] = None
    last_weights: Optional[pd.Series] = None
    scheduled_dates: Set[pd.Timestamp] = field(default_factory=set)

    # State management metadata
    state_version: str = "1.0"
    last_updated: Optional[pd.Timestamp] = None
    debug_enabled: bool = False

    def __post_init__(self):
        """Initialize specialized components after dataclass initialization."""
        # Initialize specialized components
        self.position_tracker = PositionTracker(debug_enabled=self.debug_enabled)
        self.statistics = StateStatistics(self.position_tracker)
        self.serializer = StateSerializer(self.position_tracker)

        # Log debug enabled message if debug was enabled during initialization
        if self.debug_enabled:
            self._log_debug("Debug enabled", {"timestamp": pd.Timestamp.now()})

        # Backward compatibility properties - delegate to position_tracker
        self._legacy_compatibility_setup()

    def _legacy_compatibility_setup(self):
        """Set up backward compatibility properties."""
        # These properties delegate to the position tracker for backward compatibility
        pass

    # Backward compatibility properties
    @property
    def positions(self) -> Dict[str, PositionInfo]:
        """Backward compatibility - delegate to position tracker."""
        return self.position_tracker.positions

    @property
    def position_history(self) -> List[Dict[str, Any]]:
        """Backward compatibility - delegate to position tracker."""
        return self.position_tracker.position_history

    @property
    def position_entry_dates(self) -> Dict[str, pd.Timestamp]:
        """Backward compatibility - delegate to position tracker."""
        return self.position_tracker.position_entry_dates

    @property
    def position_entry_prices(self) -> Dict[str, float]:
        """Backward compatibility - delegate to position tracker."""
        return self.position_tracker.position_entry_prices

    @property
    def consecutive_periods(self) -> Dict[str, int]:
        """Backward compatibility - delegate to position tracker."""
        return self.position_tracker.consecutive_periods

    @property
    def debug_log(self) -> List[Dict[str, Any]]:
        """Backward compatibility - delegate to position tracker."""
        return self.position_tracker.debug_log

    def reset(self):
        """Reset all state for new backtest run."""
        self.last_signal_date = None
        self.last_weights = None
        self.scheduled_dates.clear()
        self.last_updated = None

        # Reset specialized components
        self.position_tracker.reset()

        self._log_debug("State reset", {"action": "reset", "timestamp": pd.Timestamp.now()})

    def update_signal(self, date: pd.Timestamp, weights: pd.Series):
        """Update state after signal generation."""
        self.last_signal_date = date
        self.last_weights = weights.copy() if weights is not None else None
        self.last_updated = date

        self._log_debug(
            "Signal updated",
            {
                "date": date,
                "num_assets": len(weights) if weights is not None else 0,
                "total_weight": weights.sum() if weights is not None else 0.0,
            },
        )

    def update_positions(self, date: pd.Timestamp, new_weights: pd.Series, prices: pd.Series):
        """Enhanced position tracking with detailed state management - delegates to PositionTracker."""
        self.last_updated = date
        return self.position_tracker.update_positions(date, new_weights, prices)

    # Delegate position-related methods to PositionTracker
    def get_position_holding_days(self, asset: str, current_date: pd.Timestamp) -> Optional[int]:
        """Get the number of days an asset has been held - delegates to PositionTracker."""
        return self.position_tracker.get_position_holding_days(asset, current_date)

    def is_position_held(self, asset: str) -> bool:
        """Check if a position is currently held - delegates to PositionTracker."""
        return self.position_tracker.is_position_held(asset)

    def get_held_assets(self) -> Set[str]:
        """Get set of currently held assets - delegates to PositionTracker."""
        return self.position_tracker.get_held_assets()

    def get_position_info(self, asset: str) -> Optional[PositionInfo]:
        """Get detailed position information for an asset - delegates to PositionTracker."""
        return self.position_tracker.get_position_info(asset)

    def get_consecutive_periods(self, asset: str) -> int:
        """Get consecutive periods for an asset - delegates to PositionTracker."""
        return self.position_tracker.get_consecutive_periods(asset)

    def get_position_return(
        self, asset: str, current_price: Optional[float] = None
    ) -> Optional[float]:
        """Calculate current return for a position - delegates to PositionTracker."""
        return self.position_tracker.get_position_return(asset, current_price)

    # Delegate statistics methods to StateStatistics
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get summary of current portfolio state - delegates to StateStatistics."""
        return self.statistics.get_portfolio_summary(self.last_updated)

    def get_position_statistics(self) -> Dict[str, Any]:
        """Get statistics about position history - delegates to StateStatistics."""
        return self.statistics.get_position_statistics()

    def get_position_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of positions - delegates to StateStatistics."""
        return self.statistics.get_position_performance_summary()

    def get_asset_analysis(self) -> Dict[str, Any]:
        """Get analysis by asset - delegates to StateStatistics."""
        return self.statistics.get_asset_analysis()

    def get_weight_distribution_analysis(self) -> Dict[str, Any]:
        """Analyze weight distribution - delegates to StateStatistics."""
        return self.statistics.get_weight_distribution_analysis()

    def get_holding_period_analysis(self) -> Dict[str, Any]:
        """Analyze holding periods - delegates to StateStatistics."""
        return self.statistics.get_holding_period_analysis(self.last_updated)

    # Delegate serialization methods to StateSerializer
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary - delegates to StateSerializer."""
        return self.serializer.serialize_state(
            last_signal_date=self.last_signal_date,
            last_weights=self.last_weights,
            scheduled_dates=self.scheduled_dates,
            last_updated=self.last_updated,
            state_version=self.state_version,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimingState":
        """Deserialize state from dictionary - delegates to StateSerializer."""
        # Create new instance
        state = cls()

        # Deserialize using StateSerializer
        deserialized_data = state.serializer.deserialize_state(data)

        # Update basic timing state
        state.last_signal_date = deserialized_data["last_signal_date"]
        state.last_weights = deserialized_data["last_weights"]
        state.scheduled_dates = deserialized_data["scheduled_dates"]
        state.last_updated = deserialized_data["last_updated"]
        state.state_version = deserialized_data["state_version"]

        # Restore debug setting
        state.debug_enabled = data.get("debug_enabled", False)

        return state

    def save_to_file(self, filepath: str):
        """Save state to JSON file - delegates to StateSerializer."""
        self.serializer.save_to_file(
            filepath,
            last_signal_date=self.last_signal_date,
            last_weights=self.last_weights,
            scheduled_dates=self.scheduled_dates,
            last_updated=self.last_updated,
            state_version=self.state_version,
        )

    @classmethod
    def load_from_file(cls, filepath: str) -> "TimingState":
        """Load state from JSON file - delegates to StateSerializer."""
        # Create temporary instance to access serializer
        temp_state = cls()
        deserialized_data = temp_state.serializer.load_from_file(filepath)

        # Create final state with deserialized data
        state = cls()
        # Use the same serializer that already has the deserialized position data
        state.serializer = temp_state.serializer
        state.position_tracker = temp_state.position_tracker  # Already deserialized
        state.statistics = StateStatistics(state.position_tracker)

        state.last_signal_date = deserialized_data["last_signal_date"]
        state.last_weights = deserialized_data["last_weights"]
        state.scheduled_dates = deserialized_data["scheduled_dates"]
        state.last_updated = deserialized_data["last_updated"]
        state.state_version = deserialized_data["state_version"]

        # Preserve debug setting
        state.debug_enabled = temp_state.position_tracker.debug_enabled

        return state

    # Enhanced analysis methods (new functionality)
    def get_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive analysis combining all statistics.

        Returns:
            Dictionary with complete state analysis
        """
        return {
            "portfolio_summary": self.get_portfolio_summary(),
            "position_statistics": self.get_position_statistics(),
            "performance_summary": self.get_position_performance_summary(),
            "asset_analysis": self.get_asset_analysis(),
            "weight_distribution": self.get_weight_distribution_analysis(),
            "holding_period_analysis": self.get_holding_period_analysis(),
            "state_info": self.serializer.get_state_size_info(),
        }

    def validate_state_integrity(self) -> List[str]:
        """
        Validate integrity of current state.

        Returns:
            List of validation error messages (empty if valid)
        """
        state_data = self.to_dict()
        return self.serializer.validate_state_integrity(state_data)

    # Debug functionality - delegates to PositionTracker
    def enable_debug(self, enabled: bool = True):
        """Enable or disable debug logging - delegates to PositionTracker."""
        self.debug_enabled = enabled
        self.position_tracker.enable_debug(enabled)
        if enabled:
            self._log_debug("Debug enabled", {"timestamp": pd.Timestamp.now()})

    def _log_debug(self, message: str, data: Dict[str, Any]):
        """Internal debug logging method - delegates to PositionTracker."""
        self.position_tracker._log_debug(message, data)

    def get_debug_log(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get debug log entries - delegates to PositionTracker."""
        return self.position_tracker.get_debug_log(last_n)

    def clear_debug_log(self):
        """Clear debug log - delegates to PositionTracker."""
        self.position_tracker.clear_debug_log()

    def print_state_summary(self):
        """Print a human-readable summary of the current state."""
        summary = self.get_portfolio_summary()
        stats = self.get_position_statistics()

        print("=" * 50)
        print("TIMING STATE SUMMARY")
        print("=" * 50)
        print(f"Last Updated: {summary['last_updated']}")
        print(f"Active Positions: {summary['total_positions']}")
        print(f"Total Weight: {summary['total_weight']:.4f}")
        print(f"Average Holding Days: {summary['avg_holding_days']:.1f}")

        if summary["assets"]:
            print(f"Assets: {', '.join(summary['assets'])}")

        print("\nHistorical Statistics:")
        print(f"Total Trades: {stats['total_trades']}")
        if stats["total_trades"] > 0:
            print(f"Average Holding Days: {stats.get('avg_holding_days', 0):.1f}")
            if "win_rate" in stats:
                print(f"Win Rate: {stats['win_rate']:.2%}")
                print(f"Average Return: {stats['avg_return']:.2%}")

        if self.debug_enabled:
            print(f"\nDebug Log Entries: {len(self.debug_log)}")

        print("=" * 50)

    def add_test_position_history(self, position_entries: List[Dict[str, Any]]):
        """Add test position history entries - for testing purposes only.

        Args:
            position_entries: List of position history dictionaries to add
        """
        self.position_tracker.add_test_position_history(position_entries)
