"""
Validation error reporting and formatting.
Provides human-readable error reports and formatting utilities.
"""

from typing import List

from .types import ValidationError


class ValidationReporter:
    """Formats and reports validation errors in human-readable format."""

    @staticmethod
    def format_validation_report(errors: List[ValidationError]) -> str:
        """
        Format validation errors into a human-readable report.

        Args:
            errors: List of validation errors

        Returns:
            Formatted error report
        """
        if not errors:
            return "✓ Configuration is valid"

        report = ["Configuration Validation Report", "=" * 35, ""]

        # Group errors by severity
        error_count = sum(1 for e in errors if e.severity == "error")
        warning_count = sum(1 for e in errors if e.severity == "warning")

        report.append(f"Found {error_count} error(s) and {warning_count} warning(s)")
        report.append("")

        # Format each error
        for i, error in enumerate(errors, 1):
            severity_symbol = ValidationReporter._get_severity_symbol(error.severity)
            report.append(f"{i}. {severity_symbol} {error.field}: {error.message}")

            if error.value is not None:
                report.append(f"   Current value: {error.value}")

            if error.suggestion:
                report.append(f"   Suggestion: {error.suggestion}")

            report.append("")

        return "\n".join(report)

    @staticmethod
    def _get_severity_symbol(severity: str) -> str:
        """Get symbol for severity level."""
        symbols = {"error": "✗", "warning": "⚠", "info": "ℹ"}
        return symbols.get(severity, "?")

    @staticmethod
    def format_brief_summary(errors: List[ValidationError]) -> str:
        """
        Format a brief summary of validation results.

        Args:
            errors: List of validation errors

        Returns:
            Brief summary string
        """
        if not errors:
            return "✓ Valid"

        error_count = sum(1 for e in errors if e.severity == "error")
        warning_count = sum(1 for e in errors if e.severity == "warning")

        parts = []
        if error_count > 0:
            parts.append(f"{error_count} error{'s' if error_count != 1 else ''}")
        if warning_count > 0:
            parts.append(f"{warning_count} warning{'s' if warning_count != 1 else ''}")

        return "✗ " + ", ".join(parts) if parts else "✓ Valid"

    @staticmethod
    def has_errors(errors: List[ValidationError]) -> bool:
        """Check if there are any errors (not warnings) in the list."""
        return any(e.severity == "error" for e in errors)

    @staticmethod
    def filter_by_severity(errors: List[ValidationError], severity: str) -> List[ValidationError]:
        """Filter errors by severity level."""
        return [e for e in errors if e.severity == severity]

    @staticmethod
    def get_error_summary(errors: List[ValidationError]) -> dict[str, int]:
        """
        Get a summary of errors grouped by severity.

        Args:
            errors: List of validation errors

        Returns:
            Dictionary with error counts by severity
        """
        summary = {"error": 0, "warning": 0, "info": 0}

        for error in errors:
            if error.severity in summary:
                summary[error.severity] += 1

        return summary
