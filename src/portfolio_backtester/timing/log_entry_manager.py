"""
Log entry management for timing framework.
Handles storage, retrieval, and filtering of timing log entries.
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
    """Manages storage and retrieval of timing log entries."""

    def __init__(self, max_entries: int = 1000, retention_size: int = 500):
        """
        Initialize log entry manager.

        Args:
            max_entries: Maximum number of entries to keep before cleanup
            retention_size: Number of entries to keep after cleanup
        """
        self.log_entries: List[TimingLogEntry] = []
        self.max_entries = max_entries
        self.retention_size = retention_size

    def add_entry(
        self,
        event_type: str,
        strategy_name: str,
        current_date: pd.Timestamp,
        message: str,
        data: Dict[str, Any],
        level: str,
    ) -> TimingLogEntry:
        """
        Create and store a new log entry.

        Args:
            event_type: Type of event being logged
            strategy_name: Name of the strategy
            current_date: Current date for the event
            message: Log message
            data: Additional data dictionary
            level: Log level

        Returns:
            Created TimingLogEntry
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
        self._manage_storage_size()

        return entry

    def get_entries(
        self,
        strategy_name: Optional[str] = None,
        event_type: Optional[str] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        last_n: Optional[int] = None,
    ) -> List[TimingLogEntry]:
        """
        Get filtered log entries.

        Args:
            strategy_name: Filter by strategy name
            event_type: Filter by event type
            start_date: Filter by start date
            end_date: Filter by end date
            last_n: Return last N entries

        Returns:
            List of filtered log entries
        """
        entries = self.log_entries

        # Apply filters
        if strategy_name:
            entries = [e for e in entries if e.strategy_name == strategy_name]

        if event_type:
            entries = [e for e in entries if e.event_type == event_type]

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

    def get_all_entries(self) -> List[TimingLogEntry]:
        """
        Get all log entries.

        Returns:
            List of all log entries
        """
        return self.log_entries.copy()

    def clear_entries(self) -> int:
        """
        Clear all log entries.

        Returns:
            Number of entries that were cleared
        """
        count = len(self.log_entries)
        self.log_entries.clear()
        return count

    def get_entry_count(self) -> int:
        """
        Get the current number of stored entries.

        Returns:
            Number of stored entries
        """
        return len(self.log_entries)

    def _manage_storage_size(self):
        """Manage storage size by keeping entries under the maximum limit."""
        if len(self.log_entries) > self.max_entries:
            # Keep the most recent entries
            self.log_entries = self.log_entries[-self.retention_size :]
