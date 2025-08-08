"""
Log export functionality for timing framework.
Handles exporting log entries to various formats.
"""

import json
import logging
from typing import List
from .log_entry_manager import TimingLogEntry


logger = logging.getLogger(__name__)


class LogExporter:
    """Exports timing log entries to various formats."""

    @staticmethod
    def export_to_json(entries: List[TimingLogEntry], file_path: str) -> None:
        """
        Export entries to JSON format.

        Args:
            entries: List of log entries to export
            file_path: Output file path
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

        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Exported {len(entries)} entries to JSON: {file_path}")
        except Exception as e:
            logger.error(f"Failed to export to JSON: {e}")
            raise

    @staticmethod
    def export_to_csv(entries: List[TimingLogEntry], file_path: str) -> None:
        """
        Export entries to CSV format.

        Args:
            entries: List of log entries to export
            file_path: Output file path
        """
        import csv

        try:
            with open(file_path, "w", newline="", encoding="utf-8") as f:
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
                            json.dumps(entry.data) if entry.data else "",
                        ]
                    )

            logger.info(f"Exported {len(entries)} entries to CSV: {file_path}")
        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
            raise

    @staticmethod
    def export_to_text(
        entries: List[TimingLogEntry], file_path: str, include_data: bool = False
    ) -> None:
        """
        Export entries to human-readable text format.

        Args:
            entries: List of log entries to export
            file_path: Output file path
            include_data: Whether to include detailed data in output
        """
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("Timing Framework Log Export\n")
                f.write("=" * 50 + "\n\n")

                for i, entry in enumerate(entries, 1):
                    f.write(
                        f"{i}. [{entry.level}] {entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    )
                    f.write(f"   Strategy: {entry.strategy_name}\n")
                    f.write(f"   Date: {entry.current_date.date()}\n")
                    f.write(f"   Event: {entry.event_type}\n")
                    f.write(f"   Message: {entry.message}\n")

                    if include_data and entry.data:
                        f.write(f"   Data: {entry.data}\n")

                    f.write("\n")

            logger.info(f"Exported {len(entries)} entries to text: {file_path}")
        except Exception as e:
            logger.error(f"Failed to export to text: {e}")
            raise

    @classmethod
    def export(
        cls, entries: List[TimingLogEntry], file_path: str, format_type: str = "json", **kwargs
    ) -> None:
        """
        Export entries using specified format.

        Args:
            entries: List of log entries to export
            file_path: Output file path
            format_type: Export format ('json', 'csv', 'text')
            **kwargs: Additional format-specific options
        """
        format_type = format_type.lower()

        if format_type == "json":
            cls.export_to_json(entries, file_path)
        elif format_type == "csv":
            cls.export_to_csv(entries, file_path)
        elif format_type == "text":
            include_data = kwargs.get("include_data", False)
            cls.export_to_text(entries, file_path, include_data)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

    @staticmethod
    def get_supported_formats() -> List[str]:
        """Get list of supported export formats."""
        return ["json", "csv", "text"]
