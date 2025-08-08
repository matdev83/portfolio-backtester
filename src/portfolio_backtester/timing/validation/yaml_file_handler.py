"""
YAML file handling for timing configuration validation.
Handles file I/O operations and YAML parsing.
"""

import yaml
import logging
from typing import Dict, List, Any, Union
from pathlib import Path

from .types import ValidationError


logger = logging.getLogger(__name__)


class YamlFileHandler:
    """Handles YAML file operations for timing configuration validation."""

    @staticmethod
    def load_config_from_file(
        file_path: Union[str, Path],
    ) -> tuple[Dict[str, Any], List[ValidationError]]:
        """
        Load configuration from YAML file.

        Args:
            file_path: Path to YAML file

        Returns:
            Tuple of (config_dict, validation_errors)
        """
        errors = []
        file_path = Path(file_path)

        if not file_path.exists():
            errors.append(
                ValidationError(
                    field="file",
                    value=str(file_path),
                    message=f"Configuration file not found: {file_path}",
                    suggestion="Check file path and ensure file exists",
                )
            )
            return {}, errors

        try:
            with open(file_path, "r") as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            errors.append(
                ValidationError(
                    field="yaml",
                    value=str(e),
                    message=f"Invalid YAML syntax: {e}",
                    suggestion="Check YAML syntax and fix formatting errors",
                )
            )
            return {}, errors
        except Exception as e:
            errors.append(
                ValidationError(
                    field="file",
                    value=str(e),
                    message=f"Error reading file: {e}",
                    suggestion="Check file permissions and format",
                )
            )
            return {}, errors

        if not isinstance(config, dict):
            errors.append(
                ValidationError(
                    field="config",
                    value=type(config).__name__,
                    message="Configuration must be a dictionary/object",
                    suggestion="Ensure YAML file contains key-value pairs",
                )
            )
            return {}, errors

        return config, errors

    @staticmethod
    def validate_file_structure(config: Dict[str, Any]) -> List[ValidationError]:
        """
        Validate basic file structure requirements.

        Args:
            config: Configuration dictionary

        Returns:
            List of validation errors
        """
        errors = []

        # Check if timing_config exists
        if "timing_config" not in config:
            errors.append(
                ValidationError(
                    field="timing_config",
                    value=None,
                    message="Missing timing_config section",
                    suggestion="Add timing_config section with at least mode specified",
                )
            )

        return errors
