"""
State serialization functionality for timing framework.
Handles conversion to/from dictionaries, JSON files, and other formats.
"""

import json
import pandas as pd
from typing import Dict, Any, Optional, Set, List
from .position_tracker import PositionTracker, PositionInfo


class StateSerializer:
    """Handles serialization and deserialization of timing state data."""

    def __init__(self, position_tracker: PositionTracker):
        """
        Initialize state serializer.

        Args:
            position_tracker: Reference to the position tracker instance
        """
        self.position_tracker = position_tracker

    def serialize_state_to_dict(
        self,
        state_version: str = "1.0",
        last_signal_date: Optional[pd.Timestamp] = None,
        last_updated: Optional[pd.Timestamp] = None,
        last_weights: Optional[pd.Series] = None,
        scheduled_dates: Optional[Set[pd.Timestamp]] = None,
        debug_enabled: bool = False,
        debug_log: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Serialize state to dictionary for persistence.

        Args:
            state_version: State version string
            last_signal_date: Last signal generation date
            last_updated: Last state update timestamp
            last_weights: Last signal weights
            scheduled_dates: Set of scheduled dates
            debug_enabled: Whether debug logging is enabled
            debug_log: Debug log entries

        Returns:
            Dictionary representation of the state
        """

        def serialize_timestamp(ts):
            return ts.isoformat() if ts is not None else None

        def serialize_series(series):
            return series.to_dict() if series is not None else None

        return {
            "state_version": state_version,
            "last_signal_date": serialize_timestamp(last_signal_date),
            "last_updated": serialize_timestamp(last_updated),
            "last_weights": serialize_series(last_weights),
            "scheduled_dates": [ts.isoformat() for ts in (scheduled_dates or set())],
            "positions": {
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
            },
            "position_history": self.position_tracker.position_history.copy(),
            "debug_enabled": debug_enabled,
            "debug_log": (debug_log or []).copy() if debug_enabled else [],
        }

    def deserialize_state_from_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deserialize state from dictionary.

        Args:
            data: Dictionary containing serialized state

        Returns:
            Dictionary with deserialized state components
        """

        def parse_timestamp(ts_str):
            return pd.Timestamp(ts_str) if ts_str is not None else None

        def parse_series(series_dict):
            return pd.Series(series_dict) if series_dict is not None else None

        # Clear current state
        self.position_tracker.reset()

        # Parse basic state
        result = {
            "state_version": data.get("state_version", "1.0"),
            "last_signal_date": parse_timestamp(data.get("last_signal_date")),
            "last_updated": parse_timestamp(data.get("last_updated")),
            "last_weights": parse_series(data.get("last_weights")),
            "scheduled_dates": {pd.Timestamp(ts) for ts in data.get("scheduled_dates", [])},
            "debug_enabled": data.get("debug_enabled", False),
            "debug_log": data.get("debug_log", []),
        }

        # Restore position history
        self.position_tracker.position_history = data.get("position_history", [])

        # Restore positions
        for asset, pos_data in data.get("positions", {}).items():
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

        return result

    def save_to_file(
        self,
        filepath: str,
        state_version: str = "1.0",
        last_signal_date: Optional[pd.Timestamp] = None,
        last_updated: Optional[pd.Timestamp] = None,
        last_weights: Optional[pd.Series] = None,
        scheduled_dates: Optional[Set[pd.Timestamp]] = None,
        debug_enabled: bool = False,
        debug_log: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Save state to JSON file.

        Args:
            filepath: Path to save file
            state_version: State version string
            last_signal_date: Last signal generation date
            last_updated: Last state update timestamp
            last_weights: Last signal weights
            scheduled_dates: Set of scheduled dates
            debug_enabled: Whether debug logging is enabled
            debug_log: Debug log entries
        """
        state_dict = self.serialize_state_to_dict(
            state_version=state_version,
            last_signal_date=last_signal_date,
            last_updated=last_updated,
            last_weights=last_weights,
            scheduled_dates=scheduled_dates,
            debug_enabled=debug_enabled,
            debug_log=debug_log,
        )

        with open(filepath, "w") as f:
            json.dump(state_dict, f, indent=2, default=str)

    def load_from_file(self, filepath: str) -> Dict[str, Any]:
        """
        Load state from JSON file.

        Args:
            filepath: Path to load file

        Returns:
            Dictionary with deserialized state components
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        return self.deserialize_state_from_dict(data)

    def export_positions_to_csv(self, filepath: str, include_history: bool = True):
        """
        Export position data to CSV format.

        Args:
            filepath: Path to save CSV file
            include_history: Whether to include historical positions
        """
        import csv

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(
                [
                    "asset",
                    "status",
                    "entry_date",
                    "exit_date",
                    "entry_price",
                    "exit_price",
                    "entry_weight",
                    "current_weight",
                    "max_weight",
                    "min_weight",
                    "consecutive_periods",
                    "holding_days",
                    "total_return",
                    "unrealized_pnl",
                ]
            )

            # Write active positions
            for asset, pos in self.position_tracker.positions.items():
                writer.writerow(
                    [
                        asset,
                        "active",
                        pos.entry_date.strftime("%Y-%m-%d"),
                        "",  # no exit date for active positions
                        pos.entry_price,
                        "",  # no exit price for active positions
                        pos.entry_weight,
                        pos.current_weight,
                        pos.max_weight,
                        pos.min_weight,
                        pos.consecutive_periods,
                        "",  # holding days calculated dynamically
                        pos.total_return,
                        pos.unrealized_pnl,
                    ]
                )

            # Write historical positions if requested
            if include_history:
                position_history = self.position_tracker.get_position_history()
                for hist_pos in position_history:
                    writer.writerow(
                        [
                            hist_pos["asset"],
                            "closed",
                            (
                                hist_pos["entry_date"].strftime("%Y-%m-%d")
                                if isinstance(hist_pos["entry_date"], pd.Timestamp)
                                else hist_pos["entry_date"]
                            ),
                            (
                                hist_pos["exit_date"].strftime("%Y-%m-%d")
                                if isinstance(hist_pos["exit_date"], pd.Timestamp)
                                else hist_pos["exit_date"]
                            ),
                            hist_pos["entry_price"],
                            hist_pos.get("exit_price", ""),
                            hist_pos["entry_weight"],
                            "",  # no current weight for closed positions
                            hist_pos["max_weight"],
                            hist_pos["min_weight"],
                            hist_pos["consecutive_periods"],
                            hist_pos["holding_days"],
                            hist_pos["total_return"],
                            hist_pos.get("final_pnl", ""),
                        ]
                    )

    def export_summary_to_json(
        self,
        filepath: str,
        last_updated: Optional[pd.Timestamp] = None,
        include_statistics: bool = True,
    ):
        """
        Export a summary of the state to JSON format.

        Args:
            filepath: Path to save JSON file
            last_updated: Last update timestamp
            include_statistics: Whether to include detailed statistics
        """
        from .state_statistics import StateStatistics

        statistics = StateStatistics(self.position_tracker)

        summary = {
            "export_timestamp": pd.Timestamp.now().isoformat(),
            "portfolio_summary": statistics.get_portfolio_summary(last_updated),
            "active_positions": {
                asset: {
                    "entry_date": pos.entry_date.isoformat(),
                    "current_weight": pos.current_weight,
                    "consecutive_periods": pos.consecutive_periods,
                    "total_return": pos.total_return,
                    "unrealized_pnl": pos.unrealized_pnl,
                }
                for asset, pos in self.position_tracker.positions.items()
            },
        }

        if include_statistics:
            summary["statistics"] = statistics.get_position_statistics()
            summary["performance_metrics"] = statistics.get_performance_metrics()
            summary["turnover_statistics"] = statistics.get_turnover_statistics()

        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2, default=str)
