"""
Validator interfaces for abstracting validation operations.
Implements Dependency Inversion Principle for ValueError handling and validation logic.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict
from enum import Enum


class ValidationSeverity(Enum):
    """Severity levels for validation results."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationResult:
    """
    Result of a validation operation.

    Provides structured validation results with severity levels,
    messages, and optional error codes for better error handling.
    """

    def __init__(
        self,
        is_valid: bool,
        message: str = "",
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        error_code: Optional[str] = None,
        field_name: Optional[str] = None,
    ):
        """
        Initialize validation result.

        Args:
            is_valid: Whether the validation passed
            message: Human-readable validation message
            severity: Severity level of the validation issue
            error_code: Optional error code for programmatic handling
            field_name: Optional field name that failed validation
        """
        self.is_valid = is_valid
        self.message = message
        self.severity = severity
        self.error_code = error_code
        self.field_name = field_name

    def __bool__(self) -> bool:
        """Allow boolean evaluation of validation results."""
        return self.is_valid

    def __str__(self) -> str:
        """String representation of validation result."""
        prefix = "✅" if self.is_valid else "❌"
        field_info = f" ({self.field_name})" if self.field_name else ""
        return f"{prefix} {self.message}{field_info}"


class IValidator(ABC):
    """
    Interface for abstracting validation operations.

    This interface abstracts validation logic to enable better testability,
    flexibility, and adherence to the Dependency Inversion Principle.
    """

    @abstractmethod
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate a single value.

        Args:
            value: Value to validate
            context: Optional context information for validation

        Returns:
            ValidationResult indicating success/failure with details
        """
        pass


class IMultiValidator(ABC):
    """
    Interface for validating multiple values or complex objects.
    """

    @abstractmethod
    def validate_multiple(
        self, values: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> List[ValidationResult]:
        """
        Validate multiple values.

        Args:
            values: Dictionary of field names to values
            context: Optional context information for validation

        Returns:
            List of ValidationResult objects for each validation
        """
        pass

    def is_valid_multiple(
        self, values: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if all validations pass.

        Args:
            values: Dictionary of field names to values
            context: Optional context information for validation

        Returns:
            True if all validations pass, False otherwise
        """
        results = self.validate_multiple(values, context)
        return all(result.is_valid for result in results)


class IAllocationModeValidator(ABC):
    """
    Specialized interface for validating allocation modes in trading components.
    """

    @abstractmethod
    def validate_allocation_mode(self, mode: str) -> ValidationResult:
        """
        Validate allocation mode value.

        Args:
            mode: Allocation mode string to validate

        Returns:
            ValidationResult indicating if mode is valid
        """
        pass

    @abstractmethod
    def get_valid_allocation_modes(self) -> List[str]:
        """
        Get list of valid allocation modes.

        Returns:
            List of valid allocation mode strings
        """
        pass


class ITradeValidator(ABC):
    """
    Specialized interface for validating trade parameters.
    """

    @abstractmethod
    def validate_trade_value(self, trade_value: float) -> ValidationResult:
        """
        Validate trade value parameter.

        Args:
            trade_value: Trade value to validate

        Returns:
            ValidationResult indicating if trade value is valid
        """
        pass

    @abstractmethod
    def validate_transaction_cost(self, cost: float) -> ValidationResult:
        """
        Validate transaction cost parameter.

        Args:
            cost: Transaction cost to validate

        Returns:
            ValidationResult indicating if cost is valid
        """
        pass

    @abstractmethod
    def validate_trade_quantity(self, quantity: float, trade_side: str) -> ValidationResult:
        """
        Validate trade quantity based on trade side.

        Args:
            quantity: Trade quantity to validate
            trade_side: Trade side ("buy" or "sell")

        Returns:
            ValidationResult indicating if quantity is valid for the trade side
        """
        pass


class IModelValidator(ABC):
    """
    Specialized interface for validating model configurations.
    """

    @abstractmethod
    def validate_model_name(self, model_name: str) -> ValidationResult:
        """
        Validate model name parameter.

        Args:
            model_name: Model name to validate

        Returns:
            ValidationResult indicating if model name is supported
        """
        pass

    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """
        Get list of supported model names.

        Returns:
            List of supported model name strings
        """
        pass


# Factory functions for creating validator instances


def create_validator() -> IValidator:
    """Create default validator implementation."""
    from .validator_implementations import DefaultValidator

    return DefaultValidator()


def create_allocation_mode_validator() -> IAllocationModeValidator:
    """Create allocation mode validator implementation."""
    from .validator_implementations import AllocationModeValidator

    return AllocationModeValidator()


def create_trade_validator() -> ITradeValidator:
    """Create trade validator implementation."""
    from .validator_implementations import TradeValidator

    return TradeValidator()


def create_model_validator() -> IModelValidator:
    """Create model validator implementation."""
    from .validator_implementations import ModelValidator

    return ModelValidator()


def create_composite_validator() -> IMultiValidator:
    """Create composite validator for complex validation scenarios."""
    from .validator_implementations import CompositeValidator

    return CompositeValidator()
