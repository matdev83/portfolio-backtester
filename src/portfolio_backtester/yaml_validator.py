"""
YAML Configuration Validator and Error Handler

This module provides comprehensive YAML validation with detailed error reporting,
including syntax checking, schema validation, and helpful error messages.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum


class YamlErrorType(Enum):
    """Types of YAML errors that can occur."""

    SYNTAX_ERROR = "syntax_error"
    SCHEMA_ERROR = "schema_error"
    FILE_NOT_FOUND = "file_not_found"
    PERMISSION_ERROR = "permission_error"
    ENCODING_ERROR = "encoding_error"
    VALIDATION_ERROR = "validation_error"


@dataclass
class YamlError:
    """Detailed information about a YAML error."""

    error_type: YamlErrorType
    message: str
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    context: Optional[str] = None
    suggestion: Optional[str] = None
    file_path: Optional[str] = None


class YamlValidator:
    """
    Comprehensive YAML validator with detailed error reporting and suggestions.
    """

    def __init__(self):
        self.errors: List[YamlError] = []

    def validate_file(
        self, file_path: Union[str, Path]
    ) -> Tuple[bool, Optional[Dict[str, Any]], List[YamlError]]:
        """
        Validate a YAML file and return detailed error information.

        Args:
            file_path: Path to the YAML file to validate

        Returns:
            Tuple of (is_valid, parsed_data, errors)
        """
        self.errors = []
        file_path = Path(file_path)

        # Check if file exists
        if not file_path.exists():
            error = YamlError(
                error_type=YamlErrorType.FILE_NOT_FOUND,
                message=f"Configuration file not found: {file_path}",
                file_path=str(file_path),
                suggestion="Please ensure the file exists and the path is correct.",
            )
            self.errors.append(error)
            return False, None, self.errors

        # Check file permissions
        if not file_path.is_file():
            error = YamlError(
                error_type=YamlErrorType.FILE_NOT_FOUND,
                message=f"Path exists but is not a file: {file_path}",
                file_path=str(file_path),
                suggestion="Please ensure the path points to a valid file.",
            )
            self.errors.append(error)
            return False, None, self.errors

        try:
            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except PermissionError:
            error = YamlError(
                error_type=YamlErrorType.PERMISSION_ERROR,
                message=f"Permission denied reading file: {file_path}",
                file_path=str(file_path),
                suggestion="Please check file permissions and ensure you have read access.",
            )
            self.errors.append(error)
            return False, None, self.errors
        except UnicodeDecodeError as e:
            error = YamlError(
                error_type=YamlErrorType.ENCODING_ERROR,
                message=f"File encoding error: {str(e)}",
                file_path=str(file_path),
                suggestion="Please ensure the file is saved with UTF-8 encoding.",
            )
            self.errors.append(error)
            return False, None, self.errors

        # Validate YAML syntax
        return self._validate_yaml_content(content, str(file_path))

    def _validate_yaml_content(
        self, content: str, file_path: str
    ) -> Tuple[bool, Optional[Dict[str, Any]], List[YamlError]]:
        """
        Validate YAML content and provide detailed error information.

        Args:
            content: YAML content as string
            file_path: Path to the file (for error reporting)

        Returns:
            Tuple of (is_valid, parsed_data, errors)
        """
        try:
            # Parse YAML
            data = yaml.safe_load(content)

            # Check for empty file
            if data is None:
                error = YamlError(
                    error_type=YamlErrorType.VALIDATION_ERROR,
                    message="YAML file is empty or contains only comments",
                    file_path=file_path,
                    suggestion="Please add valid YAML content to the file.",
                )
                self.errors.append(error)
                return False, None, self.errors

            return True, data, self.errors

        except yaml.YAMLError as e:
            # Parse YAML error details
            error = self._parse_yaml_error(e, content, file_path)
            self.errors.append(error)
            return False, None, self.errors
        except Exception as e:
            # Catch any other unexpected errors
            error = YamlError(
                error_type=YamlErrorType.SYNTAX_ERROR,
                message=f"Unexpected error parsing YAML: {str(e)}",
                file_path=file_path,
                suggestion="Please check the YAML syntax and structure.",
            )
            self.errors.append(error)
            return False, None, self.errors

    def _parse_yaml_error(
        self, yaml_error: yaml.YAMLError, content: str, file_path: str
    ) -> YamlError:
        """
        Parse PyYAML error and create detailed error information with suggestions.

        Args:
            yaml_error: The PyYAML error object
            content: Original YAML content
            file_path: Path to the file

        Returns:
            YamlError with detailed information and suggestions
        """
        lines = content.split("\n")

        # Extract line and column information if available
        line_number = getattr(yaml_error, "problem_mark", None)
        if line_number:
            line_num = line_number.line + 1  # Convert to 1-based indexing
            col_num = line_number.column + 1

            # Get context around the error
            context_lines = []
            start_line = max(0, line_num - 3)
            end_line = min(len(lines), line_num + 2)

            for i in range(start_line, end_line):
                prefix = ">>> " if i == line_num - 1 else "    "
                context_lines.append(f"{prefix}{i+1:3d}: {lines[i]}")

            context = "\n".join(context_lines)

            # Generate specific suggestions based on error type
            suggestion = self._generate_suggestion(yaml_error, lines, line_num - 1, col_num - 1)

            return YamlError(
                error_type=YamlErrorType.SYNTAX_ERROR,
                message=str(yaml_error),
                line_number=line_num,
                column_number=col_num,
                context=context,
                suggestion=suggestion,
                file_path=file_path,
            )
        else:
            # No line information available
            suggestion = self._generate_general_suggestion(yaml_error)

            return YamlError(
                error_type=YamlErrorType.SYNTAX_ERROR,
                message=str(yaml_error),
                suggestion=suggestion,
                file_path=file_path,
            )

    def _generate_suggestion(
        self, yaml_error: yaml.YAMLError, lines: List[str], line_idx: int, col_idx: int
    ) -> str:
        """
        Generate specific suggestions based on the YAML error and context.

        Args:
            yaml_error: The PyYAML error
            lines: Lines of the YAML file
            line_idx: 0-based line index where error occurred
            col_idx: 0-based column index where error occurred

        Returns:
            Helpful suggestion string
        """
        error_str = str(yaml_error).lower()
        _ = col_idx

        if line_idx < len(lines):
            error_line = lines[line_idx]
        else:
            error_line = ""

        # Common YAML syntax errors and suggestions
        if "mapping values are not allowed here" in error_str:
            return (
                "This usually means there's an indentation problem or missing colon. "
                "Check that all dictionary keys are properly indented and followed by a colon."
            )

        elif "could not find expected" in error_str and ":" in error_str:
            return "Missing colon after a dictionary key. Make sure each key is followed by a colon (:)."

        elif "found unexpected end of stream" in error_str:
            return (
                "The YAML file ended unexpectedly. Check for unclosed brackets, braces, or quotes."
            )

        elif "expected <block end>" in error_str:
            return "Indentation error. Make sure all nested items are properly indented with spaces (not tabs)."

        elif "found character '\\t'" in error_str:
            return "YAML doesn't allow tab characters for indentation. Please use spaces instead of tabs."

        elif "duplicate key" in error_str:
            return "Duplicate key found. Each key in a YAML dictionary must be unique."

        elif "could not determine a constructor" in error_str:
            return (
                "Invalid YAML value. Check for unquoted special characters or malformed data types."
            )

        elif "scanner error" in error_str:
            if "'" in error_line or '"' in error_line:
                return (
                    "Quote character issue. Make sure all quotes are properly closed and escaped."
                )
            else:
                return (
                    "Character encoding or special character issue. Check for invalid characters."
                )

        elif "parser error" in error_str:
            return "YAML structure error. Check brackets, braces, indentation, and overall file structure."

        # Check for common indentation issues
        if error_line.strip() and not error_line.startswith(" ") and line_idx > 0:
            prev_line = lines[line_idx - 1] if line_idx > 0 else ""
            if prev_line.strip().endswith(":"):
                return "The line after a dictionary key (ending with :) should be indented."

        # Check for missing quotes around strings with special characters
        if any(
            char in error_line
            for char in [":", "[", "]", "{", "}", "&", "*", "#", "|", ">", "%", "@"]
        ):
            return "Special characters in YAML values should be quoted. Try wrapping the value in quotes."

        return (
            "Check the YAML syntax around this line. Common issues include: "
            "incorrect indentation, missing colons, unmatched quotes, or special characters."
        )

    def _generate_general_suggestion(self, yaml_error: yaml.YAMLError) -> str:
        """
        Generate general suggestions when specific line information is not available.

        Args:
            yaml_error: The PyYAML error

        Returns:
            General suggestion string
        """
        error_str = str(yaml_error).lower()

        if "scanner error" in error_str:
            return (
                "Scanner error usually indicates invalid characters or encoding issues. "
                "Check for special characters, ensure UTF-8 encoding, and verify quote matching."
            )

        elif "parser error" in error_str:
            return (
                "Parser error indicates structural YAML issues. "
                "Check overall file structure, indentation consistency, and bracket/brace matching."
            )

        else:
            return (
                "Please check the YAML file for common syntax issues: "
                "proper indentation (spaces, not tabs), colons after keys, "
                "matched quotes and brackets, and valid YAML structure."
            )

    def format_errors(self, errors: List[YamlError]) -> str:
        """
        Format errors into a human-readable string.

        Args:
            errors: List of YamlError objects

        Returns:
            Formatted error message string
        """
        if not errors:
            return "No errors found."

        formatted_lines = []
        formatted_lines.append("=" * 80)
        formatted_lines.append("YAML CONFIGURATION ERROR(S) DETECTED")
        formatted_lines.append("=" * 80)

        for i, error in enumerate(errors, 1):
            formatted_lines.append(f"\nError #{i}:")
            formatted_lines.append(f"  Type: {error.error_type.value}")
            formatted_lines.append(f"  File: {error.file_path}")

            if error.line_number:
                formatted_lines.append(
                    f"  Location: Line {error.line_number}, Column {error.column_number}"
                )

            formatted_lines.append(f"  Message: {error.message}")

            if error.context:
                formatted_lines.append("\n  Context:")
                for line in error.context.split("\n"):
                    formatted_lines.append(f"    {line}")

            if error.suggestion:
                formatted_lines.append(f"\n  Suggestion: {error.suggestion}")

            formatted_lines.append("-" * 60)

        formatted_lines.append("\nPlease fix the above errors and try again.")
        formatted_lines.append("=" * 80)

        return "\n".join(formatted_lines)


def validate_yaml_file(file_path: Union[str, Path]) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Convenience function to validate a YAML file and get formatted error output.

    Args:
        file_path: Path to the YAML file to validate

    Returns:
        Tuple of (is_valid, parsed_data, error_message)
    """
    validator = YamlValidator()
    is_valid, data, errors = validator.validate_file(file_path)
    error_message = validator.format_errors(errors) if errors else ""

    return is_valid, data, error_message


def lint_yaml_file(file_path: Union[str, Path]) -> str:
    """
    Lint a YAML file and return detailed error information.

    Args:
        file_path: Path to the YAML file to lint

    Returns:
        Formatted lint results
    """
    is_valid, data, error_message = validate_yaml_file(file_path)

    if is_valid:
        return f"[OK] YAML file is valid: {file_path}"
    else:
        return error_message


if __name__ == "__main__":
    # Example usage for testing
    import sys

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(lint_yaml_file(file_path))
    else:
        print("Usage: python yaml_validator.py <yaml_file_path>")
