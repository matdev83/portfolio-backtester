"""YAML file handling operations for strategy configuration validation.

This module handles all YAML file operations using the existing advanced
YamlValidator from the codebase. It acts as an adapter between the advanced
validator and the ValidationError format expected by config schema validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

from ..yaml_validator import YamlValidator, YamlError
from .validation_error import ValidationError


class YamlFileHandler:
    """Handles YAML file operations for configuration validation using the existing advanced validator."""

    @staticmethod
    def load_yaml_file(
        file_path: Union[str, Path],
    ) -> tuple[Dict[str, Any], List[ValidationError]]:
        """Load and parse a YAML file using the advanced validator.

        Args:
            file_path: Path to the YAML file to load

        Returns:
            Tuple of (loaded_config, validation_errors)
            If there are validation errors, loaded_config will be an empty dict
        """
        validator = YamlValidator()
        is_valid, data, yaml_errors = validator.validate_file(file_path)

        if not is_valid or yaml_errors:
            validation_errors = YamlFileHandler._convert_yaml_errors(yaml_errors)
            return {}, validation_errors

        # Additional validation for config schema requirements
        if not isinstance(data, dict):
            validation_errors = [
                ValidationError(
                    field="yaml", value="root", message="Root of YAML must be a mapping"
                )
            ]
            return {}, validation_errors

        return data, []

    @staticmethod
    def _convert_yaml_errors(yaml_errors: List[YamlError]) -> List[ValidationError]:
        """Convert YamlError objects to ValidationError objects.

        Args:
            yaml_errors: List of YamlError objects from the advanced validator

        Returns:
            List of ValidationError objects for config schema validation
        """
        validation_errors = []

        for yaml_error in yaml_errors:
            # Map YamlError to ValidationError
            field = "yaml"
            if yaml_error.line_number:
                field = f"yaml:line_{yaml_error.line_number}"
            elif yaml_error.error_type.value == "file_not_found":
                field = "file"

            # Create enhanced message with context if available
            message = yaml_error.message
            if yaml_error.context:
                message = f"{yaml_error.message}\n\nContext:\n{yaml_error.context}"

            validation_error = ValidationError(
                field=field,
                value=yaml_error.file_path,
                message=message,
                suggestion=yaml_error.suggestion,
                severity="error",
            )
            validation_errors.append(validation_error)

        return validation_errors
