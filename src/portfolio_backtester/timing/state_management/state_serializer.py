"""
State serialization functionality for timing framework.
Handles serialization and deserialization of timing state for persistence.
"""

import json
import pandas as pd
from typing import Dict, Any, Optional, Set, List
from pathlib import Path
from .position_tracker import PositionTracker, PositionInfo


class StateSerializer:
    """Handles serialization and deserialization of timing state."""

    def __init__(self, position_tracker: PositionTracker):
        """
        Initialize state serializer.

        Args:
            position_tracker: Position tracker instance to serialize
        """
        self.position_tracker = position_tracker

    def serialize_state(
        self,
        last_signal_date: Optional[pd.Timestamp] = None,
        last_weights: Optional[pd.Series] = None,
        scheduled_dates: Optional[Set[pd.Timestamp]] = None,
        last_updated: Optional[pd.Timestamp] = None,
        state_version: str = "1.0",
    ) -> Dict[str, Any]:
        """
        Serialize complete timing state to dictionary.

        Args:
            last_signal_date: Last signal generation date
            last_weights: Last signal weights
            scheduled_dates: Set of scheduled dates
            last_updated: Last update timestamp
            state_version: State version for compatibility

        Returns:
            Serialized state dictionary
        """
        return {
            "state_version": state_version,
            "last_signal_date": self._serialize_timestamp(last_signal_date),
            "last_updated": self._serialize_timestamp(last_updated),
            "last_weights": self._serialize_series(last_weights),
            "scheduled_dates": [ts.isoformat() for ts in (scheduled_dates or set())],
            "positions": self._serialize_positions(),
            "position_history": self.position_tracker.get_position_history(),
            "debug_enabled": self.position_tracker.debug_enabled,
            "debug_log": (
                self.position_tracker.get_debug_log() if self.position_tracker.debug_enabled else []
            ),
        }

    def deserialize_state(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deserialize timing state from dictionary.

        Args:
            data: Serialized state dictionary

        Returns:
            Dictionary with deserialized state components
        """
        # Clear existing state
        self.position_tracker.reset()

        # Restore basic timing state
        state = {
            "state_version": data.get("state_version", "1.0"),
            "last_signal_date": self._parse_timestamp(data.get("last_signal_date")),
            "last_updated": self._parse_timestamp(data.get("last_updated")),
            "last_weights": self._parse_series(data.get("last_weights")),
            "scheduled_dates": {pd.Timestamp(ts) for ts in data.get("scheduled_dates", [])},
        }

        # Restore position tracker state
        self.position_tracker.debug_enabled = data.get("debug_enabled", False)
        self.position_tracker.debug_log = data.get("debug_log", [])
        self.position_tracker.position_history = data.get("position_history", [])

        # Restore positions
        self._deserialize_positions(data.get("positions", {}))

        return state

    def save_to_file(self, filepath: str, **state_args) -> None:
        """
        Save timing state to JSON file.

        Args:
            filepath: File path to save to
            **state_args: Additional state arguments for serialization
        """
        state_data = self.serialize_state(**state_args)

        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(state_data, f, indent=2, default=str)

    def load_from_file(self, filepath: str) -> Dict[str, Any]:
        """
        Load timing state from JSON file.

        Args:
            filepath: File path to load from

        Returns:
            Dictionary with deserialized state components
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        return self.deserialize_state(data)

    def _serialize_positions(self) -> Dict[str, Any]:
        """Serialize active positions to dictionary format."""
        return {
            asset: {
                "entry_date": pos.entry_date.isoformat(),
                "entry_price": pos.entry_price,
                "entry_weight": pos.entry_weight,
                "current_weight": pos.current_weight,
                "consecutive_periods": pos.consecutive_periods,
                "max_weight": pos.max_weight,
                "min_weight": pos.min_weight,
                "total_return": pos.total_return,
                "unrealized_pnl": pos.unrealized_pnl,
            }
            for asset, pos in self.position_tracker.positions.items()
        }

    def _deserialize_positions(self, positions_data: Dict[str, Any]) -> None:
        """Deserialize positions from dictionary format."""
        for asset, pos_data in positions_data.items():
            position_info = PositionInfo(
                entry_date=pd.Timestamp(pos_data["entry_date"]),
                entry_price=pos_data["entry_price"],
                entry_weight=pos_data["entry_weight"],
                current_weight=pos_data["current_weight"],
                consecutive_periods=pos_data["consecutive_periods"],
                max_weight=pos_data["max_weight"],
                min_weight=pos_data["min_weight"],
                total_return=pos_data["total_return"],
                unrealized_pnl=pos_data["unrealized_pnl"],
            )
            self.position_tracker.positions[asset] = position_info

            # Maintain legacy compatibility
            self.position_tracker.position_entry_dates[asset] = position_info.entry_date
            self.position_tracker.position_entry_prices[asset] = position_info.entry_price
            self.position_tracker.consecutive_periods[asset] = position_info.consecutive_periods

    def _serialize_timestamp(self, timestamp: Optional[pd.Timestamp]) -> Optional[str]:
        """Serialize timestamp to ISO format string."""
        return timestamp.isoformat() if timestamp is not None else None

    def _parse_timestamp(self, timestamp_str: Optional[str]) -> Optional[pd.Timestamp]:
        """Parse timestamp from ISO format string."""
        return pd.Timestamp(timestamp_str) if timestamp_str is not None else None

    def _serialize_series(self, series: Optional[pd.Series]) -> Optional[Dict[str, float]]:
        """Serialize pandas Series to dictionary."""
        return series.to_dict() if series is not None else None

    def _parse_series(self, series_dict: Optional[Dict[str, float]]) -> Optional[pd.Series]:
        """Parse pandas Series from dictionary."""
        return pd.Series(series_dict) if series_dict is not None else None

    def get_state_size_info(self) -> Dict[str, Any]:
        """
        Get information about state size for monitoring.

        Returns:
            Dictionary with state size metrics
        """
        positions_count = len(self.position_tracker.positions)
        history_count = len(self.position_tracker.position_history)
        debug_log_count = len(self.position_tracker.debug_log)

        return {
            "active_positions": positions_count,
            "historical_positions": history_count,
            "debug_log_entries": debug_log_count,
            "estimated_memory_kb": self._estimate_memory_usage(),
            "serialization_complexity": self._get_complexity_score(),
        }

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in KB (rough approximation)."""
        # Rough estimation based on typical sizes
        positions_size = len(self.position_tracker.positions) * 0.5  # ~0.5KB per position
        history_size = (
            len(self.position_tracker.position_history) * 0.3
        )  # ~0.3KB per historical position
        debug_size = len(self.position_tracker.debug_log) * 0.2  # ~0.2KB per debug entry

        return positions_size + history_size + debug_size

    def _get_complexity_score(self) -> str:
        """Get complexity score for serialization."""
        total_items = (
            len(self.position_tracker.positions)
            + len(self.position_tracker.position_history)
            + len(self.position_tracker.debug_log)
        )

        if total_items < 100:
            return "low"
        elif total_items < 1000:
            return "medium"
        else:
            return "high"

    def validate_state_integrity(self, state_data: Dict[str, Any]) -> List[str]:
        """
        Validate integrity of serialized state data.

        Args:
            state_data: Serialized state dictionary

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check required fields
        required_fields = ["state_version", "positions", "position_history"]
        for field in required_fields:
            if field not in state_data:
                errors.append(f"Missing required field: {field}")

        # Validate positions structure
        if "positions" in state_data:
            for asset, pos_data in state_data["positions"].items():
                required_pos_fields = ["entry_date", "entry_price", "current_weight"]
                for field in required_pos_fields:
                    if field not in pos_data:
                        errors.append(f"Position {asset} missing field: {field}")

        # Validate timestamps
        timestamp_fields = ["last_signal_date", "last_updated"]
        for field in timestamp_fields:
            if field in state_data and state_data[field] is not None:
                try:
                    pd.Timestamp(state_data[field])
                except Exception as e:
                    errors.append(f"Invalid timestamp format for {field}: {e}")

        return errors
