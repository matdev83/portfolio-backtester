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

    @staticmethod
    def _validate_parameter_prefixes(
        strategy: str, params: Dict[str, Any]
    ) -> List[ValidationError]:
        """Validate that parameter keys follow the required prefix convention.

        Args:
            strategy: Strategy name to check prefixes against
            params: Dictionary of strategy parameters

        Returns:
            List of validation errors for prefix violations
        """
        errors: List[ValidationError] = []
        prefix = f"{strategy}."

        for key in params.keys():
            if key.startswith(prefix):
                continue  # Correctly prefixed

            if "." in key:
                # Namespaced for some other purpose – allow but warn
                errors.append(
                    ValidationError(
                        field=f"strategy_params.{key}",
                        value=None,
                        message=(
                            "Parameter key contains a dot but does not start with the "
                            f"strategy prefix '{prefix}'. This may be unintended."
                        ),
                        severity="warning",
                    )
                )
            else:
                # Plain key – invalid after refactor
                errors.append(
                    ValidationError(
                        field=f"strategy_params.{key}",
                        value=None,
                        message="Parameter key missing required '<strategy>.' prefix",
                        suggestion=f"Rename to '{prefix}{key}'",
                    )
                )

        return errors
