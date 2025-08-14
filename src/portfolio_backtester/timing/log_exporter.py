"""
Log export functionality for timing framework.
Handles exporting timing log entries to various formats.
"""

import json
import csv
from typing import List, Optional
from .log_entry_manager import TimingLogEntry


class LogExporter:
    """Handles exporting timing log entries to various formats."""

    def __init__(self, enable_detailed_logging: bool = False):
        """
        Initialize log exporter.

        Args:
            enable_detailed_logging: Whether to include detailed data in exports
        """
        self.enable_detailed_logging = enable_detailed_logging

    def export_logs(self, entries: List[TimingLogEntry], file_path: str, format: str = "json"):
        """
        Export log entries to file.

        Args:
            entries: List of TimingLogEntry objects to export
            file_path: Output file path
            format: Export format ('json', 'csv')

        Raises:
            ValueError: If format is not supported
        """
        if format.lower() == "json":
            self.export_json(entries, file_path)
        elif format.lower() == "csv":
            self.export_csv(entries, file_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def export_json(self, entries: List[TimingLogEntry], file_path: str):
        """
        Export entries to JSON format.

        Args:
            entries: List of TimingLogEntry objects to export
            file_path: Output JSON file path
        """
        data = []
        for entry in entries:
            data.append(
                {
                    "timestamp": entry.timestamp.isoformat(),
                    "event_type": entry.event_type,
                    "strategy_name": entry.strategy_name,
                    "current_date": entry.current_date.isoformat(),
                    "message": entry.message,
                    "data": entry.data,
                    "level": entry.level,
                }
            )

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    def export_csv(self, entries: List[TimingLogEntry], file_path: str):
        """
        Export entries to CSV format.

        Args:
            entries: List of TimingLogEntry objects to export
            file_path: Output CSV file path
        """
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(
                [
                    "timestamp",
                    "event_type",
                    "strategy_name",
                    "current_date",
                    "message",
                    "level",
                    "data",
                ]
            )

            # Write entries
            for entry in entries:
                writer.writerow(
                    [
                        entry.timestamp.isoformat(),
                        entry.event_type,
                        entry.strategy_name,
                        entry.current_date.isoformat(),
                        entry.message,
                        entry.level,
                        json.dumps(entry.data),
                    ]
                )

    def print_recent_logs(
        self, entries: List[TimingLogEntry], title: str = "RECENT TIMING LOG ENTRIES"
    ):
        """
        Print log entries in a readable format.

        Args:
            entries: List of TimingLogEntry objects to print
            title: Title for the log output
        """
        if not entries:
            print("No log entries found")
            return

        print(f"\n{'=' * 80}")
        print(f"{title} (Last {len(entries)})")
        print(f"{'=' * 80}")

        for i, entry in enumerate(entries, 1):
            print(f"\n{i}. [{entry.level}] {entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Strategy: {entry.strategy_name}")
            print(f"   Date: {entry.current_date.date()}")
            print(f"   Event: {entry.event_type}")
            print(f"   Message: {entry.message}")

            if entry.data and self.enable_detailed_logging:
                print(f"   Data: {entry.data}")

        print(f"\n{'=' * 80}")

    def format_entry_as_string(
        self, entry: TimingLogEntry, include_data: Optional[bool] = None
    ) -> str:
        """
        Format a single log entry as a string.

        Args:
            entry: TimingLogEntry to format
            include_data: Whether to include data field (uses class setting if None)

        Returns:
            Formatted string representation of the log entry
        """
        if include_data is None:
            include_data = self.enable_detailed_logging

        base_info = (
            f"[{entry.level}] {entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')} "
            f"- {entry.strategy_name} - {entry.event_type}: {entry.message}"
        )

        if include_data and entry.data:
            base_info += f" | Data: {entry.data}"

        return base_info
