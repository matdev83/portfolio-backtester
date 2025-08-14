"""
Concrete implementations of validator interfaces.
Provides type-safe validation implementations following the Dependency Inversion Principle.
"""

import logging
from typing import Any, Dict, List, Optional, Set

from .validator_interface import (
    IValidator,
    IMultiValidator,
    IAllocationModeValidator,
    ITradeValidator,
    IModelValidator,
    ValidationResult,
    ValidationSeverity,
)

logger = logging.getLogger(__name__)


class DefaultValidator(IValidator):
    """
    Default implementation of validator interface.
    Provides basic validation with customizable rules.
    """

    def __init__(self, custom_rules: Optional[Dict[str, Any]] = None):
        """
        Initialize default validator.

        Args:
            custom_rules: Optional custom validation rules
        """
        self.custom_rules = custom_rules or {}

    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Basic validation - checks for None values."""
        if value is None:
            return ValidationResult(
                is_valid=False,
                message="Value cannot be None",
                severity=ValidationSeverity.ERROR,
                error_code="NULL_VALUE",
            )

        return ValidationResult(
            is_valid=True, message="Value is valid", severity=ValidationSeverity.INFO
        )


class AllocationModeValidator(IAllocationModeValidator):
    """
    Validator for allocation modes in trading components.
    Handles validation of portfolio allocation strategies.
    """

    def __init__(self, additional_modes: Optional[Set[str]] = None):
        """
        Initialize allocation mode validator.

        Args:
            additional_modes: Optional additional valid modes beyond defaults
        """
        self._base_modes = {
            "reinvestment",
            "compound",
            "fixed_fractional",
            "fixed_capital",
        }
        self._additional_modes = additional_modes or set()
        self._valid_modes = self._base_modes | self._additional_modes

        logger.debug(f"AllocationModeValidator initialized with modes: {self._valid_modes}")

    def validate_allocation_mode(self, mode: str) -> ValidationResult:
        """Validate allocation mode value."""
        if not isinstance(mode, str):
            return ValidationResult(
                is_valid=False,
                message=f"Allocation mode must be a string, got {type(mode).__name__}",
                severity=ValidationSeverity.ERROR,
                error_code="INVALID_TYPE",
                field_name="allocation_mode",
            )

        if mode not in self._valid_modes:
            return ValidationResult(
                is_valid=False,
                message=f"Invalid allocation_mode '{mode}'. Must be one of: {sorted(self._valid_modes)}",
                severity=ValidationSeverity.ERROR,
                error_code="INVALID_ALLOCATION_MODE",
                field_name="allocation_mode",
            )

        return ValidationResult(
            is_valid=True,
            message=f"Allocation mode '{mode}' is valid",
            severity=ValidationSeverity.INFO,
        )

    def get_valid_allocation_modes(self) -> List[str]:
        """Get list of valid allocation modes."""
        return sorted(self._valid_modes)


class TradeValidator(ITradeValidator):
    """
    Validator for trade parameters and constraints.
    Handles validation of trading operations and parameters.
    """

    def __init__(self, min_trade_value: float = 0.01, max_cost_ratio: float = 0.1):
        """
        Initialize trade validator.

        Args:
            min_trade_value: Minimum allowed trade value
            max_cost_ratio: Maximum transaction cost as ratio of trade value
        """
        self.min_trade_value = min_trade_value
        self.max_cost_ratio = max_cost_ratio

    def validate_trade_value(self, trade_value: float) -> ValidationResult:
        """Validate trade value parameter."""
        if not isinstance(trade_value, (int, float)):
            return ValidationResult(
                is_valid=False,
                message=f"Trade value must be numeric, got {type(trade_value).__name__}",
                severity=ValidationSeverity.ERROR,
                error_code="INVALID_TYPE",
                field_name="trade_value",
            )

        if trade_value < 0:
            return ValidationResult(
                is_valid=False,
                message="Trade value cannot be negative",
                severity=ValidationSeverity.ERROR,
                error_code="NEGATIVE_TRADE_VALUE",
                field_name="trade_value",
            )

        if trade_value < self.min_trade_value:
            return ValidationResult(
                is_valid=False,
                message=f"Trade value {trade_value} below minimum {self.min_trade_value}",
                severity=ValidationSeverity.WARNING,
                error_code="TRADE_VALUE_TOO_SMALL",
                field_name="trade_value",
            )

        return ValidationResult(
            is_valid=True,
            message=f"Trade value {trade_value} is valid",
            severity=ValidationSeverity.INFO,
        )

    def validate_transaction_cost(self, cost: float) -> ValidationResult:
        """Validate transaction cost parameter."""
        if not isinstance(cost, (int, float)):
            return ValidationResult(
                is_valid=False,
                message=f"Transaction cost must be numeric, got {type(cost).__name__}",
                severity=ValidationSeverity.ERROR,
                error_code="INVALID_TYPE",
                field_name="transaction_cost",
            )

        if cost < 0:
            return ValidationResult(
                is_valid=False,
                message="Transaction cost cannot be negative",
                severity=ValidationSeverity.ERROR,
                error_code="NEGATIVE_TRANSACTION_COST",
                field_name="transaction_cost",
            )

        return ValidationResult(
            is_valid=True,
            message=f"Transaction cost {cost} is valid",
            severity=ValidationSeverity.INFO,
        )

    def validate_trade_quantity(self, quantity: float, trade_side: str) -> ValidationResult:
        """Validate trade quantity based on trade side."""
        if not isinstance(quantity, (int, float)):
            return ValidationResult(
                is_valid=False,
                message=f"Trade quantity must be numeric, got {type(quantity).__name__}",
                severity=ValidationSeverity.ERROR,
                error_code="INVALID_TYPE",
                field_name="quantity",
            )

        if quantity == 0:
            return ValidationResult(
                is_valid=False,
                message="Trade quantity cannot be zero",
                severity=ValidationSeverity.ERROR,
                error_code="ZERO_QUANTITY",
                field_name="quantity",
            )

        if trade_side == "buy" and quantity <= 0:
            return ValidationResult(
                is_valid=False,
                message="Buy trades must have positive quantities",
                severity=ValidationSeverity.ERROR,
                error_code="INVALID_BUY_QUANTITY",
                field_name="quantity",
            )

        if trade_side == "sell" and quantity >= 0:
            return ValidationResult(
                is_valid=False,
                message="Sell trades must have negative quantities",
                severity=ValidationSeverity.ERROR,
                error_code="INVALID_SELL_QUANTITY",
                field_name="quantity",
            )

        return ValidationResult(
            is_valid=True,
            message=f"Trade quantity {quantity} is valid for {trade_side} trade",
            severity=ValidationSeverity.INFO,
        )


class ModelValidator(IModelValidator):
    """
    Validator for model configurations and names.
    Handles validation of supported model types and configurations.
    """

    def __init__(self, supported_models: Optional[Set[str]] = None):
        """
        Initialize model validator.

        Args:
            supported_models: Set of supported model names
        """
        # Default supported transaction cost models
        self._default_models = {
            "realistic",
            "fixed",
            "percentage",
            "tiered",
            "interactive_brokers",
            "flat_fee",
            "combined",
            "custom",
        }
        self._supported_models = supported_models or self._default_models

        logger.debug(f"ModelValidator initialized with models: {self._supported_models}")

    def validate_model_name(self, model_name: str) -> ValidationResult:
        """Validate model name parameter."""
        if not isinstance(model_name, str):
            return ValidationResult(
                is_valid=False,
                message=f"Model name must be a string, got {type(model_name).__name__}",
                severity=ValidationSeverity.ERROR,
                error_code="INVALID_TYPE",
                field_name="model_name",
            )

        if not model_name.strip():
            return ValidationResult(
                is_valid=False,
                message="Model name cannot be empty",
                severity=ValidationSeverity.ERROR,
                error_code="EMPTY_MODEL_NAME",
                field_name="model_name",
            )

        if model_name not in self._supported_models:
            return ValidationResult(
                is_valid=False,
                message=f"Unsupported transaction cost model: {model_name}. Supported models: {sorted(self._supported_models)}",
                severity=ValidationSeverity.ERROR,
                error_code="UNSUPPORTED_MODEL",
                field_name="model_name",
            )

        return ValidationResult(
            is_valid=True,
            message=f"Model '{model_name}' is supported",
            severity=ValidationSeverity.INFO,
        )

    def get_supported_models(self) -> List[str]:
        """Get list of supported model names."""
        return sorted(self._supported_models)


class CompositeValidator(IMultiValidator):
    """
    Composite validator that can validate multiple values using different validators.
    Provides coordinated validation across multiple validation domains.
    """

    def __init__(
        self,
        allocation_validator: Optional[IAllocationModeValidator] = None,
        trade_validator: Optional[ITradeValidator] = None,
        model_validator: Optional[IModelValidator] = None,
    ):
        """
        Initialize composite validator.

        Args:
            allocation_validator: Validator for allocation modes
            trade_validator: Validator for trade parameters
            model_validator: Validator for model configurations
        """
        self.allocation_validator = allocation_validator or AllocationModeValidator()
        self.trade_validator = trade_validator or TradeValidator()
        self.model_validator = model_validator or ModelValidator()

    def validate_multiple(
        self, values: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> List[ValidationResult]:
        """Validate multiple values using appropriate validators."""
        results = []

        # Validate allocation mode if present
        if "allocation_mode" in values:
            result = self.allocation_validator.validate_allocation_mode(values["allocation_mode"])
            results.append(result)

        # Validate trade value if present
        if "trade_value" in values:
            result = self.trade_validator.validate_trade_value(values["trade_value"])
            results.append(result)

        # Validate transaction cost if present
        if "transaction_cost" in values:
            result = self.trade_validator.validate_transaction_cost(values["transaction_cost"])
            results.append(result)

        # Validate trade quantity if present
        if "quantity" in values and "trade_side" in values:
            result = self.trade_validator.validate_trade_quantity(
                values["quantity"], values["trade_side"]
            )
            results.append(result)

        # Validate model name if present
        if "model_name" in values:
            result = self.model_validator.validate_model_name(values["model_name"])
            results.append(result)

        return results


# Specialized validators for specific use cases


class StrictAllocationModeValidator(AllocationModeValidator):
    """
    Strict allocation mode validator that only allows core allocation modes.
    """

    def __init__(self):
        """Initialize with only core allocation modes."""
        super().__init__(additional_modes=set())  # No additional modes allowed


class PermissiveTradeValidator(TradeValidator):
    """
    Permissive trade validator with relaxed constraints.
    """

    def __init__(self):
        """Initialize with relaxed validation constraints."""
        super().__init__(min_trade_value=0.001, max_cost_ratio=0.5)


# Factory functions for specialized validators


def create_strict_allocation_validator() -> IAllocationModeValidator:
    """Create strict allocation mode validator."""
    return StrictAllocationModeValidator()


def create_permissive_trade_validator() -> ITradeValidator:
    """Create permissive trade validator."""
    return PermissiveTradeValidator()
