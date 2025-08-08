"""
Mode-specific validators for timing configuration.
Each validator handles validation logic for a specific timing mode.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Type

from .types import ValidationError


class ModeValidator(ABC):
    """Abstract base class for mode-specific validators."""

    @abstractmethod
    def validate(self, config: Dict[str, Any]) -> List[ValidationError]:
        """Validate configuration for this mode."""
        pass

    @abstractmethod
    def get_mode_name(self) -> str:
        """Get the mode name this validator handles."""
        pass


class TimeBasedValidator(ModeValidator):
    """Validator for time-based timing configuration."""

    VALID_TIME_FREQUENCIES = ["D", "W", "M", "ME", "Q", "QE", "A", "Y", "YE"]

    def get_mode_name(self) -> str:
        return "time_based"

    def validate(self, config: Dict[str, Any]) -> List[ValidationError]:
        """Validate time-based timing configuration."""
        errors = []

        # Validate rebalance_frequency
        frequency = config.get("rebalance_frequency", "M")
        if frequency not in self.VALID_TIME_FREQUENCIES:
            errors.append(
                ValidationError(
                    field="rebalance_frequency",
                    value=frequency,
                    message=f"Invalid rebalance frequency: {frequency}",
                    suggestion=f'Use one of: {", ".join(self.VALID_TIME_FREQUENCIES)}',
                )
            )

        # Validate rebalance_offset
        offset = config.get("rebalance_offset", 0)
        if not isinstance(offset, int):
            errors.append(
                ValidationError(
                    field="rebalance_offset",
                    value=offset,
                    message=f"rebalance_offset must be an integer, got {type(offset).__name__}",
                    suggestion="Use integer value between -30 and 30",
                )
            )
        elif abs(offset) > 30:
            errors.append(
                ValidationError(
                    field="rebalance_offset",
                    value=offset,
                    message=f"rebalance_offset must be between -30 and 30, got {offset}",
                    suggestion="Use value between -30 and 30 days",
                )
            )

        # Warn about signal-based fields in time-based mode
        signal_fields = ["scan_frequency", "min_holding_period", "max_holding_period"]
        for field in signal_fields:
            if field in config:
                errors.append(
                    ValidationError(
                        field=field,
                        value=config[field],
                        message=f"Field {field} is not used in time_based mode",
                        suggestion=f"Remove {field} or change mode to signal_based",
                        severity="warning",
                    )
                )

        return errors


class SignalBasedValidator(ModeValidator):
    """Validator for signal-based timing configuration."""

    VALID_SCAN_FREQUENCIES = ["D", "W", "M"]

    def get_mode_name(self) -> str:
        return "signal_based"

    def validate(self, config: Dict[str, Any]) -> List[ValidationError]:
        """Validate signal-based timing configuration."""
        errors = []

        # Validate scan_frequency
        scan_freq = config.get("scan_frequency", "D")
        if scan_freq not in self.VALID_SCAN_FREQUENCIES:
            errors.append(
                ValidationError(
                    field="scan_frequency",
                    value=scan_freq,
                    message=f"Invalid scan frequency: {scan_freq}",
                    suggestion=f'Use one of: {", ".join(self.VALID_SCAN_FREQUENCIES)}',
                )
            )

        # Validate holding periods
        min_holding = config.get("min_holding_period", 1)
        max_holding = config.get("max_holding_period")

        if not isinstance(min_holding, int) or min_holding < 1:
            errors.append(
                ValidationError(
                    field="min_holding_period",
                    value=min_holding,
                    message=f"min_holding_period must be a positive integer, got {min_holding}",
                    suggestion="Use positive integer (e.g., 1 for minimum 1 day holding)",
                )
            )

        if max_holding is not None:
            if not isinstance(max_holding, int) or max_holding < 1:
                errors.append(
                    ValidationError(
                        field="max_holding_period",
                        value=max_holding,
                        message=f"max_holding_period must be a positive integer or null, got {max_holding}",
                        suggestion="Use positive integer or null for unlimited holding",
                    )
                )
            elif isinstance(min_holding, int) and min_holding > max_holding:
                errors.append(
                    ValidationError(
                        field="max_holding_period",
                        value=max_holding,
                        message=f"max_holding_period ({max_holding}) cannot exceed min_holding_period ({min_holding})",
                        suggestion=f"Set max_holding_period to at least {min_holding} or null",
                    )
                )

        # Warn about time-based fields in signal-based mode
        time_fields = ["rebalance_frequency", "rebalance_offset"]
        for field in time_fields:
            if field in config:
                errors.append(
                    ValidationError(
                        field=field,
                        value=config[field],
                        message=f"Field {field} is not used in signal_based mode",
                        suggestion=f"Remove {field} or change mode to time_based",
                        severity="warning",
                    )
                )

        return errors


class CustomValidator(ModeValidator):
    """Validator for custom timing configuration."""

    def get_mode_name(self) -> str:
        return "custom"

    def validate(self, config: Dict[str, Any]) -> List[ValidationError]:
        """Validate custom timing configuration."""
        errors = []

        # Require custom_controller_class
        if "custom_controller_class" not in config:
            errors.append(
                ValidationError(
                    field="custom_controller_class",
                    value=None,
                    message="custom_controller_class is required for custom mode",
                    suggestion='Specify fully qualified class name (e.g., "mymodule.MyTimingController")',
                )
            )
        else:
            controller_class = config["custom_controller_class"]
            # Allow short names for built-in controllers registered in CustomTimingRegistry
            try:
                from ..custom_timing_registry import (
                    CustomTimingRegistry,
                )  # local import to avoid cycle

                if not isinstance(controller_class, str):
                    errors.append(
                        ValidationError(
                            field="custom_controller_class",
                            value=controller_class,
                            message="custom_controller_class must be a string",
                            suggestion="Provide controller class name or fully qualified path",
                        )
                    )
                else:
                    has_dot = "." in controller_class
                    is_registered = CustomTimingRegistry.get(controller_class) is not None
                    if not has_dot and not is_registered:
                        errors.append(
                            ValidationError(
                                field="custom_controller_class",
                                value=controller_class,
                                message="custom_controller_class must be a fully qualified class name or a built-in registered controller",
                                suggestion='Use format "module.submodule.ClassName" or one of: '
                                + ", ".join(CustomTimingRegistry.list_registered().keys()),
                            )
                        )
            except ImportError:
                # If CustomTimingRegistry is not available, just validate the basic format
                if not isinstance(controller_class, str):
                    errors.append(
                        ValidationError(
                            field="custom_controller_class",
                            value=controller_class,
                            message="custom_controller_class must be a string",
                            suggestion="Provide controller class name or fully qualified path",
                        )
                    )

        # Validate custom_controller_params if present
        if "custom_controller_params" in config:
            params = config["custom_controller_params"]
            if not isinstance(params, dict):
                errors.append(
                    ValidationError(
                        field="custom_controller_params",
                        value=params,
                        message="custom_controller_params must be a dictionary",
                        suggestion="Use key-value pairs for controller parameters",
                    )
                )

        return errors


class ModeValidatorFactory:
    """Factory for creating mode-specific validators."""

    _validators: Dict[str, Type[ModeValidator]] = {
        "time_based": TimeBasedValidator,
        "signal_based": SignalBasedValidator,
        "custom": CustomValidator,
    }

    @classmethod
    def get_validator(cls, mode: str) -> ModeValidator:
        """
        Get validator for the specified mode.

        Args:
            mode: Timing mode

        Returns:
            Mode-specific validator instance

        Raises:
            ValueError: If mode is not supported
        """
        if mode not in cls._validators:
            raise ValueError(f"Unsupported timing mode: {mode}")

        return cls._validators[mode]()

    @classmethod
    def get_supported_modes(cls) -> List[str]:
        """Get list of supported timing modes."""
        return list(cls._validators.keys())
