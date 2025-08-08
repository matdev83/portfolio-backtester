"""
Log analysis functionality for timing framework.
Provides statistical analysis and insights from timing log entries.
"""

from typing import Dict, Any, List, Optional
from collections import Counter
from .log_entry_manager import TimingLogEntry
from ..interfaces.attribute_accessor_interface import (
    IObjectFieldAccessor,
    create_object_field_accessor,
)


class LogAnalyzer:
    """Analyzes timing log entries to provide statistics and insights."""

    def __init__(self, field_accessor: Optional[IObjectFieldAccessor] = None):
        """
        Initialize log analyzer.

        Args:
            field_accessor: Injected accessor for object field access (DIP)
        """
        # Dependency injection for field access (DIP)
        self._field_accessor = field_accessor or create_object_field_accessor()

    def get_summary(
        self, entries: List[TimingLogEntry], strategy_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive summary of log entries.

        Args:
            entries: List of TimingLogEntry objects to analyze
            strategy_name: Optional strategy filter (for backwards compatibility)

        Returns:
            Summary statistics dictionary
        """
        # Apply strategy filter if specified
        if strategy_name:
            entries = [e for e in entries if e.strategy_name == strategy_name]

        if not entries:
            return {"total_entries": 0}

        # Basic counts
        event_counts = self._count_by_field(entries, "event_type")
        level_counts = self._count_by_field(entries, "level")
        strategy_counts = self._count_by_field(entries, "strategy_name")

        # Date analysis
        date_range = self._analyze_date_range(entries)

        # Performance insights
        performance_insights = self._analyze_performance_patterns(entries)

        return {
            "total_entries": len(entries),
            "event_counts": event_counts,
            "level_counts": level_counts,
            "strategy_counts": strategy_counts,
            "date_range": date_range,
            "strategies": list(strategy_counts.keys()),
            "performance_insights": performance_insights,
        }

    def get_event_timeline(self, entries: List[TimingLogEntry]) -> List[Dict[str, Any]]:
        """
        Get chronological timeline of events.

        Args:
            entries: List of TimingLogEntry objects to analyze

        Returns:
            List of timeline events sorted by timestamp
        """
        timeline = []
        for entry in sorted(entries, key=lambda x: x.timestamp):
            timeline.append(
                {
                    "timestamp": entry.timestamp.isoformat(),
                    "strategy": entry.strategy_name,
                    "event_type": entry.event_type,
                    "message": entry.message,
                    "level": entry.level,
                }
            )
        return timeline

    def get_error_analysis(self, entries: List[TimingLogEntry]) -> Dict[str, Any]:
        """
        Analyze error patterns in log entries.

        Args:
            entries: List of TimingLogEntry objects to analyze

        Returns:
            Error analysis dictionary
        """
        error_entries = [e for e in entries if e.level == "ERROR"]

        if not error_entries:
            return {"error_count": 0, "has_errors": False}

        error_types: Counter[str] = Counter()
        error_strategies: Counter[str] = Counter()

        for entry in error_entries:
            # Extract error type from data if available
            error_type = entry.data.get("error_type", "Unknown")
            error_types[error_type] += 1
            error_strategies[entry.strategy_name] += 1

        return {
            "error_count": len(error_entries),
            "has_errors": True,
            "error_types": dict(error_types),
            "strategies_with_errors": dict(error_strategies),
            "first_error": error_entries[0].timestamp.isoformat(),
            "last_error": error_entries[-1].timestamp.isoformat(),
        }

    def get_strategy_performance_summary(
        self, entries: List[TimingLogEntry]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get performance summary by strategy.

        Args:
            entries: List of TimingLogEntry objects to analyze

        Returns:
            Dictionary with strategy names as keys and performance data as values
        """
        strategies: Dict[str, Dict[str, Any]] = {}

        for entry in entries:
            strategy = entry.strategy_name
            if strategy not in strategies:
                strategies[strategy] = {
                    "total_events": 0,
                    "event_types": Counter(),
                    "level_counts": Counter(),
                    "date_range": {"start": None, "end": None},
                    "has_errors": False,
                }

            strategy_data = strategies[strategy]
            strategy_data["total_events"] = strategy_data["total_events"] + 1

            # Update counters safely
            event_counter = strategy_data["event_types"]
            level_counter = strategy_data["level_counts"]
            date_range = strategy_data["date_range"]

            if isinstance(event_counter, Counter):
                event_counter[entry.event_type] += 1
            if isinstance(level_counter, Counter):
                level_counter[entry.level] += 1

            # Update date range
            if date_range["start"] is None or entry.current_date < date_range["start"]:
                date_range["start"] = entry.current_date
            if date_range["end"] is None or entry.current_date > date_range["end"]:
                date_range["end"] = entry.current_date

            # Check for errors
            if entry.level == "ERROR":
                strategy_data["has_errors"] = True

        # Convert counters to regular dicts and format dates
        for strategy, data in strategies.items():
            event_counter = data["event_types"]
            level_counter = data["level_counts"]
            date_range = data["date_range"]

            if isinstance(event_counter, Counter):
                data["event_types"] = dict(event_counter)
            if isinstance(level_counter, Counter):
                data["level_counts"] = dict(level_counter)

            # Format dates
            start_date = date_range["start"]
            end_date = date_range["end"]
            if start_date is not None:
                date_range["start"] = start_date.isoformat()
            if end_date is not None:
                date_range["end"] = end_date.isoformat()

        return strategies

    def _count_by_field(self, entries: List[TimingLogEntry], field: str) -> Dict[str, int]:
        """Count entries by a specific field using injected dependency instead of direct getattr."""
        counter: Counter[str] = Counter()
        for entry in entries:
            field_value = self._field_accessor.get_field_value(entry, field)
            counter[field_value] += 1
        return dict(counter)

    def _analyze_date_range(self, entries: List[TimingLogEntry]) -> Dict[str, Any]:
        """Analyze the date range of entries."""
        if not entries:
            return {}

        dates = [entry.current_date for entry in entries]
        return {
            "start_date": min(dates).isoformat(),
            "end_date": max(dates).isoformat(),
            "span_days": (max(dates) - min(dates)).days,
        }

    def _analyze_performance_patterns(self, entries: List[TimingLogEntry]) -> Dict[str, Any]:
        """Analyze performance-related patterns in the logs."""
        performance_entries = [e for e in entries if e.event_type == "performance_metric"]
        signal_entries = [e for e in entries if e.event_type == "signal_generation"]
        position_entries = [e for e in entries if e.event_type == "position_update"]
        rebalance_entries = [e for e in entries if e.event_type == "rebalance"]

        return {
            "performance_metrics_logged": len(performance_entries),
            "signals_generated": len(signal_entries),
            "position_updates": len(position_entries),
            "rebalances": len(rebalance_entries),
            "activity_ratio": {
                "signals_per_rebalance": len(signal_entries) / max(len(rebalance_entries), 1),
                "positions_per_rebalance": len(position_entries) / max(len(rebalance_entries), 1),
            },
        }
