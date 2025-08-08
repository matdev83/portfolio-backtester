"""Validation report formatting for configuration validation.

This module handles formatting of validation errors into human-readable reports
with proper severity indicators and suggestions.
"""

from __future__ import annotations

from typing import List

from .validation_error import ValidationError


class ValidationReporter:
    """Formats validation errors into human-readable reports."""

    @staticmethod
    def format_report(errors: List[ValidationError]) -> str:
        """Format a list of validation errors into a human-readable report.

        Args:
            errors: List of ValidationError objects to format

        Returns:
            Formatted report string
        """
        if not errors:
            return "✓ Configuration is valid"

        parts = ["Configuration Validation Report", "=" * 35, ""]

        for idx, err in enumerate(errors, 1):
            symbol = ValidationReporter._get_severity_symbol(err.severity)
            parts.append(f"{idx}. {symbol} {err.field}: {err.message}")

            if err.suggestion:
                parts.append(f"   Suggestion: {err.suggestion}")
            parts.append("")

        return "\n".join(parts)

    @staticmethod
    def _get_severity_symbol(severity: str) -> str:
        """Get the appropriate symbol for a given severity level.

        Args:
            severity: Severity level ("error", "warning", "info")

        Returns:
            Symbol character for the severity
        """
        severity_symbols = {"error": "✗", "warning": "⚠", "info": "ℹ"}
        return severity_symbols.get(severity, "•")
