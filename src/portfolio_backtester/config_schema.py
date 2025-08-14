"""Schema validation for top-level strategy scenario YAML files.

This lightweight validator focuses on the `strategy` and `strategy_params` blocks
and enforces the new *namespaced* parameter convention introduced during the
parameter-prefix refactor (e.g. `momentum.lookback_months`).

The file purposefully limits scope: timing-related settings are validated by
`portfolio_backtester.timing.config_schema.TimingConfigSchema`, while optimizer
specifications are validated inside the optimization subsystem.  Here we only
check structural correctness of scenario files and the prefix rule.

This class now acts as a facade to maintain backward compatibility while using
SOLID-compliant specialized classes for each responsibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union
import logging

from .config_validation import (
    ValidationError,
    YamlFileHandler,
    ConfigValidator,
    ValidationReporter,
)

logger = logging.getLogger(__name__)

# Re-export ValidationError for backward compatibility
__all__ = ["ValidationError", "StrategyConfigSchema"]


class StrategyConfigSchema:
    """Facade validator enforcing namespaced `strategy_params` keys.

    This class maintains backward compatibility while delegating to specialized
    SOLID-compliant classes for each responsibility.
    """

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> List[ValidationError]:
        """Validate a strategy configuration dictionary.

        Args:
            config: Dictionary containing strategy configuration

        Returns:
            List of validation errors found
        """
        return ConfigValidator.validate_config(config)

    @classmethod
    def validate_yaml_file(cls, file_path: Union[str, Path]) -> List[ValidationError]:
        """Load and validate a YAML configuration file.

        Args:
            file_path: Path to the YAML file to validate

        Returns:
            List of validation errors found
        """
        config, file_errors = YamlFileHandler.load_yaml_file(file_path)

        if file_errors:
            return file_errors

        validation_errors = cls.validate_config(config)
        return validation_errors

    @classmethod
    def format_report(cls, errors: List[ValidationError]) -> str:
        """Format validation errors into a human-readable report.

        Args:
            errors: List of ValidationError objects to format

        Returns:
            Formatted report string
        """
        return ValidationReporter.format_report(errors)
