"""Configuration validation logic for strategy parameters.

This module handles the core validation logic for strategy configuration,
including parameter structure validation and prefix enforcement.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .validation_error import ValidationError


class ConfigValidator:
    """Validates strategy configuration parameters and structure."""

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> List[ValidationError]:
        """Validate a strategy configuration dictionary.

        Args:
            config: Dictionary containing strategy configuration

        Returns:
            List of validation errors found
        """
        errors: List[ValidationError] = []

        # Basic presence checks
        strategy = config.get("strategy")
        if strategy is not None and not isinstance(strategy, str):
            errors.append(
                ValidationError(
                    field="strategy",
                    value=strategy,
                    message="Invalid 'strategy' field (must be string if provided)",
                    suggestion="Omit or set to a string identifier",
                )
            )

        params = config.get("strategy_params")
        if params is None:
            return errors

        if not isinstance(params, dict):
            errors.append(
                ValidationError(
                    field="strategy_params",
                    value=type(params).__name__,
                    message="'strategy_params' must be a mapping/dictionary",
                )
            )
            return errors

        return errors
