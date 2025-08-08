"""
Log entry management for timing framework.
Handles creation, storage, and filtering of timing log entries.
"""

import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class TimingLogEntry:
    """Represents a single timing log entry."""

    timestamp: datetime
    event_type: str  # 'signal_generation', 'state_change', 'rebalance', 'position_update', 'error'
    strategy_name: str
    current_date: pd.Timestamp
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    level: str = "INFO"


class LogEntryManager:
    """Manages timing log entries with efficient storage and filtering."""

    def __init__(self, max_entries: int = 1000, cleanup_threshold: int = 500):
        """
        Initialize log entry manager.

        Args:
            max_entries: Maximum number of entries to keep in memory
            cleanup_threshold: Number of entries to keep after cleanup
        """
        self.log_entries: List[TimingLogEntry] = []
        self.max_entries = max_entries
        self.cleanup_threshold = cleanup_threshold

    def add_entry(
        self,
        event_type: str,
        strategy_name: str,
        current_date: pd.Timestamp,
        message: str,
        data: Dict[str, Any],
        level: str = "INFO",
    ) -> TimingLogEntry:
        """
        Add a new log entry.

        Args:
            event_type: Type of event
            strategy_name: Name of the strategy
            current_date: Current date
            message: Log message
            data: Additional data
            level: Log level

        Returns:
            Created log entry
        """
        entry = TimingLogEntry(
            timestamp=datetime.now(),
            event_type=event_type,
            strategy_name=strategy_name,
            current_date=current_date,
            message=message,
            data=data,
            level=level,
        )

        self.log_entries.append(entry)
        self._cleanup_if_needed()

        return entry

    def _cleanup_if_needed(self):
        """Clean up old entries if max limit is exceeded."""
        if len(self.log_entries) > self.max_entries:
            self.log_entries = self.log_entries[-self.cleanup_threshold :]

    def get_entries(
        self,
        strategy_name: Optional[str] = None,
        event_type: Optional[str] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        last_n: Optional[int] = None,
        level: Optional[str] = None,
    ) -> List[TimingLogEntry]:
        """
        Get filtered log entries.

        Args:
            strategy_name: Filter by strategy name
            event_type: Filter by event type
            start_date: Filter by start date
            end_date: Filter by end date
            last_n: Return last N entries
            level: Filter by log level

        Returns:
            List of filtered log entries
        """
        entries = self.log_entries.copy()

        # Apply filters
        if strategy_name:
            entries = [e for e in entries if e.strategy_name == strategy_name]

        if event_type:
            entries = [e for e in entries if e.event_type == event_type]

        if level:
            entries = [e for e in entries if e.level == level]

        if start_date:
            entries = [e for e in entries if e.current_date >= start_date]

        if end_date:
            entries = [e for e in entries if e.current_date <= end_date]

        # Sort by timestamp
        entries = sorted(entries, key=lambda x: x.timestamp)

        # Return last N if specified
        if last_n:
            entries = entries[-last_n:]

        return entries

    def get_entry_count(self) -> int:
        """Get total number of entries."""
        return len(self.log_entries)

    def clear_entries(self):
        """Clear all log entries."""
        self.log_entries.clear()

    def get_unique_strategies(self) -> List[str]:
        """Get list of unique strategy names."""
        return list(set(entry.strategy_name for entry in self.log_entries))

    def get_unique_event_types(self) -> List[str]:
        """Get list of unique event types."""
        return list(set(entry.event_type for entry in self.log_entries))

    def get_date_range(self) -> Optional[Dict[str, pd.Timestamp]]:
        """
        Get date range of entries.

        Returns:
            Dictionary with start_date and end_date, or None if no entries
        """
        if not self.log_entries:
            return None

        dates = [entry.current_date for entry in self.log_entries]
        return {"start_date": min(dates), "end_date": max(dates)}
