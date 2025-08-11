"""
Signal validator interface for signal strategy validation.

This module provides interfaces for validating signal strategy configuration
without using isinstance checks.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from portfolio_backtester.yaml_validator import YamlError, YamlErrorType


class ISignalValidator(ABC):
    """Interface for validating signal strategy configuration."""

    @abstractmethod
    def validate_signal_strategy_logic(
        self, scenario_data: Dict[str, Any], strategy_class: Any, file_path_str: Optional[str]
    ) -> List[Any]:
        """Validate signal strategy specific configuration."""
        pass

    @abstractmethod
    def validate_timing_config(
        self, timing_config: Dict[str, Any], file_path_str: Optional[str]
    ) -> List[Any]:
        """Validate timing configuration structure."""
        pass


class DefaultSignalValidator(ISignalValidator):
    """Default implementation of signal validator."""

    def validate_signal_strategy_logic(
        self, scenario_data: Dict[str, Any], strategy_class: Any, file_path_str: Optional[str]
    ) -> List[Any]:
        """Validate signal strategy specific configuration."""
        # YamlError and YamlErrorType imported at module level
        errors = []

        # Check for timing_config in strategy_params
        strategy_params = scenario_data.get("strategy_params", {})
        if isinstance(strategy_params, dict):
            timing_config = None
            for key, value in strategy_params.items():
                if key.endswith(".timing_config") or key == "timing_config":
                    timing_config = value
                    break

            if timing_config is not None:
                # Validate timing_config structure
                errors.extend(self.validate_timing_config(timing_config, file_path_str))

        # Check for EMA parameters if relevant
        strategy_name = scenario_data.get("strategy", "")
        if "ema" in strategy_name.lower() and isinstance(strategy_params, dict):
            fast_ema = None
            slow_ema = None

            for key, value in strategy_params.items():
                if "fast_ema" in key.lower():
                    fast_ema = value
                elif "slow_ema" in key.lower():
                    slow_ema = value

            if fast_ema is not None and slow_ema is not None and fast_ema >= slow_ema:
                errors.append(
                    YamlError(
                        error_type=YamlErrorType.VALIDATION_ERROR,
                        message=(
                            f"Fast EMA period ({fast_ema}) should be less than slow EMA period ({slow_ema}). "
                            "Fast EMA should respond more quickly to price changes."
                        ),
                        file_path=file_path_str,
                    )
                )

        return errors

    def validate_timing_config(
        self, timing_config: Dict[str, Any], file_path_str: Optional[str]
    ) -> List[Any]:
        """Validate timing configuration structure."""
        # YamlError and YamlErrorType imported at module level
        errors = []

        if not isinstance(timing_config, dict):
            errors.append(
                YamlError(
                    error_type=YamlErrorType.VALIDATION_ERROR,
                    message=(
                        "timing_config must be a dictionary. "
                        f"Got {type(timing_config).__name__} instead."
                    ),
                    file_path=file_path_str,
                )
            )
            return errors

        # Check for mode
        mode = timing_config.get("mode")
        if mode is not None:
            valid_modes = {"time_based", "signal_based", "combined"}
            if mode not in valid_modes:
                errors.append(
                    YamlError(
                        error_type=YamlErrorType.VALIDATION_ERROR,
                        message=(
                            f"Invalid timing mode '{mode}'. "
                            f"Valid modes are: {', '.join(valid_modes)}."
                        ),
                        file_path=file_path_str,
                    )
                )

        return errors


class SignalValidatorFactory:
    """Factory for creating signal validators."""

    @staticmethod
    def create() -> ISignalValidator:
        """Create a new signal validator instance."""
        return DefaultSignalValidator()
