"""
YAML configuration schema validation for timing framework.
Refactored version using SOLID principles with facade pattern for backward compatibility.
"""

import logging
from typing import Dict, List, Any, Union
from pathlib import Path

from .validation import (
    YamlFileHandler,
    ConfigDefaults,
    ModeValidatorFactory,
    ValidationReporter,
    ValidationError,
)

logger = logging.getLogger(__name__)


class TimingConfigSchema:
    """
    YAML configuration schema validator for timing framework.

    Refactored facade that delegates to specialized classes following SOLID principles:
    - YamlFileHandler: Handles YAML file I/O operations
    - ModeValidatorFactory: Creates mode-specific validators
    - ConfigDefaults: Provides default configurations
    - ValidationReporter: Formats validation reports
    """

    # Define valid values for each configuration option (maintained for compatibility)
    VALID_MODES = ["time_based", "signal_based", "custom"]
    VALID_TIME_FREQUENCIES = ["D", "W", "M", "ME", "Q", "QE", "A", "Y", "YE"]
    VALID_SCAN_FREQUENCIES = ["D", "W", "M"]

    # Configuration schema definition (maintained for compatibility)
    SCHEMA = {
        "timing_config": {
            "type": "object",
            "required": ["mode"],
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": VALID_MODES,
                    "description": "Timing mode: time_based for scheduled rebalancing, signal_based for market-driven timing, custom for user-defined timing",
                },
                "rebalance_frequency": {
                    "type": "string",
                    "enum": VALID_TIME_FREQUENCIES,
                    "description": "Rebalancing frequency for time_based mode (D=Daily, W=Weekly, M=Monthly, Q=Quarterly, A=Annual)",
                },
                "rebalance_offset": {
                    "type": "integer",
                    "minimum": -30,
                    "maximum": 30,
                    "description": "Days offset from standard rebalancing date (-30 to +30)",
                },
                "scan_frequency": {
                    "type": "string",
                    "enum": VALID_SCAN_FREQUENCIES,
                    "description": "Signal scanning frequency for signal_based mode (D=Daily, W=Weekly, M=Monthly)",
                },
                "min_holding_period": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Minimum days to hold a position before allowing exit",
                },
                "max_holding_period": {
                    "type": ["integer", "null"],
                    "minimum": 1,
                    "description": "Maximum days to hold a position (null for unlimited)",
                },
                "custom_controller_class": {
                    "type": "string",
                    "description": "Fully qualified class name for custom timing controller",
                },
                "custom_controller_params": {
                    "type": "object",
                    "description": "Parameters to pass to custom timing controller constructor",
                },
                "enable_logging": {
                    "type": "boolean",
                    "default": False,
                    "description": "Enable detailed logging of timing decisions",
                },
                "log_level": {
                    "type": "string",
                    "enum": ["DEBUG", "INFO", "WARNING", "ERROR"],
                    "default": "INFO",
                    "description": "Logging level for timing operations",
                },
            },
        }
    }

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> List[ValidationError]:
        """
        Validate timing configuration against schema.

        Args:
            config: Configuration dictionary to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Validate basic file structure
        structure_errors = YamlFileHandler.validate_file_structure(config)
        if structure_errors:
            return structure_errors

        timing_config = config["timing_config"]

        # Validate mode (required)
        errors.extend(cls._validate_mode(timing_config))

        # Mode-specific validation using specialized validators
        mode = timing_config.get("mode", "time_based")
        if mode in ModeValidatorFactory.get_supported_modes():
            try:
                mode_validator = ModeValidatorFactory.get_validator(mode)
                errors.extend(mode_validator.validate(timing_config))
            except ValueError as e:
                errors.append(
                    ValidationError(
                        field="mode",
                        value=mode,
                        message=str(e),
                        suggestion=f"Use one of: {', '.join(cls.VALID_MODES)}",
                    )
                )

        # Validate common optional fields
        errors.extend(cls._validate_logging_config(timing_config))

        return errors

    @classmethod
    def _validate_mode(cls, config: Dict[str, Any]) -> List[ValidationError]:
        """Validate timing mode."""
        errors = []

        if "mode" not in config:
            errors.append(
                ValidationError(
                    field="mode",
                    value=None,
                    message="Missing required field: mode",
                    suggestion=f"Add mode field with one of: {', '.join(cls.VALID_MODES)}",
                )
            )
        else:
            mode = config["mode"]
            if mode not in cls.VALID_MODES:
                errors.append(
                    ValidationError(
                        field="mode",
                        value=mode,
                        message=f"Invalid timing mode: {mode}",
                        suggestion=f"Use one of: {', '.join(cls.VALID_MODES)}",
                    )
                )

        return errors

    @classmethod
    def _validate_logging_config(cls, config: Dict[str, Any]) -> List[ValidationError]:
        """Validate logging configuration."""
        errors = []

        # Validate enable_logging
        if "enable_logging" in config:
            enable_logging = config["enable_logging"]
            if not isinstance(enable_logging, bool):
                errors.append(
                    ValidationError(
                        field="enable_logging",
                        value=enable_logging,
                        message="enable_logging must be a boolean",
                        suggestion="Use true or false",
                    )
                )

        # Validate log_level
        if "log_level" in config:
            log_level = config["log_level"]
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
            if log_level not in valid_levels:
                errors.append(
                    ValidationError(
                        field="log_level",
                        value=log_level,
                        message=f"Invalid log level: {log_level}",
                        suggestion=f"Use one of: {', '.join(valid_levels)}",
                    )
                )

        return errors

    @classmethod
    def get_default_config(cls, mode: str = "time_based") -> Dict[str, Any]:
        """
        Get default configuration for a timing mode.

        Delegates to ConfigDefaults class.

        Args:
            mode: Timing mode ('time_based', 'signal_based', 'custom')

        Returns:
            Default configuration dictionary
        """
        return ConfigDefaults.get_default_config(mode)

    @classmethod
    def validate_yaml_file(cls, file_path: Union[str, Path]) -> List[ValidationError]:
        """
        Validate a YAML configuration file.

        Delegates to YamlFileHandler class.

        Args:
            file_path: Path to YAML file

        Returns:
            List of validation errors
        """
        config, file_errors = YamlFileHandler.load_config_from_file(file_path)
        if file_errors:
            return file_errors

        # Validate the loaded configuration
        validation_errors = cls.validate_config(config)
        return validation_errors

    @classmethod
    def format_validation_report(cls, errors: List[ValidationError]) -> str:
        """
        Format validation errors into a human-readable report.

        Delegates to ValidationReporter class.

        Args:
            errors: List of validation errors

        Returns:
            Formatted error report
        """
        return ValidationReporter.format_validation_report(errors)


# Convenience functions for backward compatibility
def validate_timing_config(
    config: Dict[str, Any], raise_on_error: bool = True
) -> List[ValidationError]:
    """
    Convenience function to validate timing configuration.

    Args:
        config: Configuration dictionary
        raise_on_error: Whether to raise exception on validation errors

    Returns:
        List of validation errors

    Raises:
        ValueError: If raise_on_error is True and validation fails
    """
    errors = TimingConfigSchema.validate_config(config)

    if raise_on_error and ValidationReporter.has_errors(errors):
        error_report = TimingConfigSchema.format_validation_report(errors)
        raise ValueError(f"Configuration validation failed:\n{error_report}")

    return errors


def validate_timing_config_file(
    file_path: Union[str, Path], raise_on_error: bool = True
) -> List[ValidationError]:
    """
    Convenience function to validate timing configuration file.

    Args:
        file_path: Path to YAML configuration file
        raise_on_error: Whether to raise exception on validation errors

    Returns:
        List of validation errors

    Raises:
        ValueError: If raise_on_error is True and validation fails
    """
    errors = TimingConfigSchema.validate_yaml_file(file_path)

    if raise_on_error and ValidationReporter.has_errors(errors):
        error_report = TimingConfigSchema.format_validation_report(errors)
        raise ValueError(f"Configuration file validation failed:\n{error_report}")

    return errors
