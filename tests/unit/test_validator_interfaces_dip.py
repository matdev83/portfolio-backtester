"""
Tests for Validator DIP (Dependency Inversion Principle) implementation.

This test suite validates the validator interfaces and their concrete
implementations, ensuring proper dependency injection and SOLID compliance.
"""

import pytest
from unittest.mock import Mock

from portfolio_backtester.interfaces.validator_interface import (
    IValidator,
    IAllocationModeValidator,
    ITradeValidator,
    IModelValidator,
    IMultiValidator,
    ValidationResult,
    ValidationSeverity,
    create_validator,
    create_allocation_mode_validator,
    create_trade_validator,
    create_model_validator,
    create_composite_validator,
)
from portfolio_backtester.interfaces.validator_implementations import (
    DefaultValidator,
    AllocationModeValidator,
    TradeValidator,
    ModelValidator,
    CompositeValidator,
    create_strict_allocation_validator,
    create_permissive_trade_validator,
)


class TestValidationResult:
    """Test ValidationResult data structure."""

    def test_validation_result_creation(self):
        """Test basic validation result creation."""
        result = ValidationResult(True, "Valid", ValidationSeverity.INFO)
        assert result.is_valid
        assert result.message == "Valid"
        assert result.severity == ValidationSeverity.INFO

    def test_validation_result_boolean_evaluation(self):
        """Test boolean evaluation of validation results."""
        valid_result = ValidationResult(True, "Valid")
        invalid_result = ValidationResult(False, "Invalid")

        assert bool(valid_result) is True
        assert bool(invalid_result) is False

    def test_validation_result_string_representation(self):
        """Test string representation of validation results."""
        valid_result = ValidationResult(True, "Test passed")
        invalid_result = ValidationResult(False, "Test failed", field_name="test_field")

        assert "✅" in str(valid_result)
        assert "❌" in str(invalid_result)
        assert "test_field" in str(invalid_result)


class TestValidatorInterfaces:
    """Test validator interface contracts."""

    def test_default_validator_interface_compliance(self):
        """Test that DefaultValidator implements IValidator interface."""
        validator = DefaultValidator()
        assert isinstance(validator, IValidator)

    def test_allocation_mode_validator_interface_compliance(self):
        """Test that AllocationModeValidator implements IAllocationModeValidator interface."""
        validator = AllocationModeValidator()
        assert isinstance(validator, IAllocationModeValidator)

    def test_trade_validator_interface_compliance(self):
        """Test that TradeValidator implements ITradeValidator interface."""
        validator = TradeValidator()
        assert isinstance(validator, ITradeValidator)

    def test_model_validator_interface_compliance(self):
        """Test that ModelValidator implements IModelValidator interface."""
        validator = ModelValidator()
        assert isinstance(validator, IModelValidator)

    def test_composite_validator_interface_compliance(self):
        """Test that CompositeValidator implements IMultiValidator interface."""
        validator = CompositeValidator()
        assert isinstance(validator, IMultiValidator)

    def test_factory_functions_return_correct_types(self):
        """Test that factory functions return correct interface implementations."""
        default_validator = create_validator()
        allocation_validator = create_allocation_mode_validator()
        trade_validator = create_trade_validator()
        model_validator = create_model_validator()
        composite_validator = create_composite_validator()

        assert isinstance(default_validator, IValidator)
        assert isinstance(allocation_validator, IAllocationModeValidator)
        assert isinstance(trade_validator, ITradeValidator)
        assert isinstance(model_validator, IModelValidator)
        assert isinstance(composite_validator, IMultiValidator)


class TestDefaultValidator:
    """Test DefaultValidator implementation."""

    def setup_method(self):
        """Set up test environment."""
        self.validator = DefaultValidator()

    def test_validate_with_valid_value(self):
        """Test validation with valid non-None value."""
        result = self.validator.validate("test_value")
        assert result.is_valid
        assert "valid" in result.message.lower()

    def test_validate_with_none_value(self):
        """Test validation with None value."""
        result = self.validator.validate(None)
        assert not result.is_valid
        assert "none" in result.message.lower()
        assert result.error_code == "NULL_VALUE"


class TestAllocationModeValidator:
    """Test AllocationModeValidator implementation."""

    def setup_method(self):
        """Set up test environment."""
        self.validator = AllocationModeValidator()

    def test_validate_valid_allocation_modes(self):
        """Test validation of valid allocation modes."""
        valid_modes = ["reinvestment", "compound", "fixed_fractional", "fixed_capital"]

        for mode in valid_modes:
            result = self.validator.validate_allocation_mode(mode)
            assert result.is_valid, f"Mode {mode} should be valid"

    def test_validate_invalid_allocation_mode(self):
        """Test validation of invalid allocation mode."""
        result = self.validator.validate_allocation_mode("invalid_mode")
        assert not result.is_valid
        assert "invalid_mode" in result.message.lower()
        assert result.error_code == "INVALID_ALLOCATION_MODE"

    def test_validate_non_string_allocation_mode(self):
        """Test validation of non-string allocation mode."""
        result = self.validator.validate_allocation_mode(123)
        assert not result.is_valid
        assert result.error_code == "INVALID_TYPE"

    def test_get_valid_allocation_modes(self):
        """Test getting list of valid allocation modes."""
        modes = self.validator.get_valid_allocation_modes()
        assert isinstance(modes, list)
        assert "reinvestment" in modes
        assert len(modes) >= 4

    def test_additional_modes(self):
        """Test validator with additional modes."""
        additional_modes = {"custom_mode", "test_mode"}
        validator = AllocationModeValidator(additional_modes=additional_modes)

        result = validator.validate_allocation_mode("custom_mode")
        assert result.is_valid


class TestTradeValidator:
    """Test TradeValidator implementation."""

    def setup_method(self):
        """Set up test environment."""
        self.validator = TradeValidator()

    def test_validate_valid_trade_value(self):
        """Test validation of valid trade value."""
        result = self.validator.validate_trade_value(100.0)
        assert result.is_valid

    def test_validate_negative_trade_value(self):
        """Test validation of negative trade value."""
        result = self.validator.validate_trade_value(-100.0)
        assert not result.is_valid
        assert result.error_code == "NEGATIVE_TRADE_VALUE"

    def test_validate_invalid_trade_value_type(self):
        """Test validation of invalid trade value type."""
        result = self.validator.validate_trade_value("invalid")
        assert not result.is_valid
        assert result.error_code == "INVALID_TYPE"

    def test_validate_valid_transaction_cost(self):
        """Test validation of valid transaction cost."""
        result = self.validator.validate_transaction_cost(10.0)
        assert result.is_valid

    def test_validate_negative_transaction_cost(self):
        """Test validation of negative transaction cost."""
        result = self.validator.validate_transaction_cost(-10.0)
        assert not result.is_valid
        assert result.error_code == "NEGATIVE_TRANSACTION_COST"

    def test_validate_buy_trade_quantity(self):
        """Test validation of buy trade quantity."""
        # Valid buy quantity (positive)
        result = self.validator.validate_trade_quantity(100.0, "buy")
        assert result.is_valid

        # Invalid buy quantity (negative)
        result = self.validator.validate_trade_quantity(-100.0, "buy")
        assert not result.is_valid
        assert result.error_code == "INVALID_BUY_QUANTITY"

    def test_validate_sell_trade_quantity(self):
        """Test validation of sell trade quantity."""
        # Valid sell quantity (negative)
        result = self.validator.validate_trade_quantity(-100.0, "sell")
        assert result.is_valid

        # Invalid sell quantity (positive)
        result = self.validator.validate_trade_quantity(100.0, "sell")
        assert not result.is_valid
        assert result.error_code == "INVALID_SELL_QUANTITY"

    def test_validate_zero_trade_quantity(self):
        """Test validation of zero trade quantity."""
        result = self.validator.validate_trade_quantity(0.0, "buy")
        assert not result.is_valid
        assert result.error_code == "ZERO_QUANTITY"


class TestModelValidator:
    """Test ModelValidator implementation."""

    def setup_method(self):
        """Set up test environment."""
        self.validator = ModelValidator()

    def test_validate_supported_model(self):
        """Test validation of supported model."""
        result = self.validator.validate_model_name("realistic")
        assert result.is_valid

    def test_validate_unsupported_model(self):
        """Test validation of unsupported model."""
        result = self.validator.validate_model_name("unsupported_model")
        assert not result.is_valid
        assert result.error_code == "UNSUPPORTED_MODEL"

    def test_validate_empty_model_name(self):
        """Test validation of empty model name."""
        result = self.validator.validate_model_name("")
        assert not result.is_valid
        assert result.error_code == "EMPTY_MODEL_NAME"

    def test_validate_non_string_model_name(self):
        """Test validation of non-string model name."""
        result = self.validator.validate_model_name(123)
        assert not result.is_valid
        assert result.error_code == "INVALID_TYPE"

    def test_get_supported_models(self):
        """Test getting list of supported models."""
        models = self.validator.get_supported_models()
        assert isinstance(models, list)
        assert "realistic" in models
        assert len(models) > 0


class TestCompositeValidator:
    """Test CompositeValidator implementation."""

    def setup_method(self):
        """Set up test environment."""
        self.validator = CompositeValidator()

    def test_validate_multiple_valid_values(self):
        """Test validation of multiple valid values."""
        values = {
            "allocation_mode": "reinvestment",
            "trade_value": 100.0,
            "transaction_cost": 10.0,
            "model_name": "realistic",
        }

        results = self.validator.validate_multiple(values)
        assert all(result.is_valid for result in results)
        assert len(results) == 4

    def test_validate_multiple_with_invalid_values(self):
        """Test validation of multiple values with some invalid."""
        values = {
            "allocation_mode": "invalid_mode",
            "trade_value": -100.0,  # Invalid negative value
            "transaction_cost": 10.0,  # Valid
            "model_name": "realistic",  # Valid
        }

        results = self.validator.validate_multiple(values)
        valid_results = [r for r in results if r.is_valid]
        invalid_results = [r for r in results if not r.is_valid]

        assert len(valid_results) == 2  # transaction_cost and model_name
        assert len(invalid_results) == 2  # allocation_mode and trade_value

    def test_is_valid_multiple(self):
        """Test is_valid_multiple convenience method."""
        valid_values = {"allocation_mode": "reinvestment", "trade_value": 100.0}

        invalid_values = {"allocation_mode": "invalid_mode", "trade_value": 100.0}

        assert self.validator.is_valid_multiple(valid_values)
        assert not self.validator.is_valid_multiple(invalid_values)


class TestSpecializedValidators:
    """Test specialized validator implementations."""

    def test_strict_allocation_validator(self):
        """Test strict allocation validator."""
        validator = create_strict_allocation_validator()

        # Should only allow core modes
        assert validator.validate_allocation_mode("reinvestment").is_valid

        # Additional modes should not be available
        modes = validator.get_valid_allocation_modes()
        assert len(modes) == 4  # Only the 4 core modes

    def test_permissive_trade_validator(self):
        """Test permissive trade validator."""
        validator = create_permissive_trade_validator()

        # Should accept very small trade values
        result = validator.validate_trade_value(0.001)
        assert result.is_valid


class TestDependencyInjectionIntegration:
    """Test integration of DIP pattern with trading components."""

    def test_portfolio_value_tracker_accepts_validator_injection(self):
        """Test that PortfolioValueTracker accepts dependency injection."""
        from portfolio_backtester.trading.portfolio_value_tracker import PortfolioValueTracker

        mock_validator = Mock(spec=IAllocationModeValidator)
        mock_validator.validate_allocation_mode.return_value = ValidationResult(True, "Valid")

        # Should not raise exception
        tracker = PortfolioValueTracker(
            initial_portfolio_value=100000.0,
            allocation_mode="reinvestment",
            allocation_validator=mock_validator,
        )

        # Verify validator was used
        assert tracker._allocation_validator == mock_validator
        mock_validator.validate_allocation_mode.assert_called_once_with("reinvestment")

    def test_trade_aggregator_accepts_validator_injection(self):
        """Test that TradeAggregator accepts dependency injection."""
        from portfolio_backtester.strategies._core.base.base.trade_aggregator import TradeAggregator

        mock_allocation_validator = Mock(spec=IAllocationModeValidator)
        mock_allocation_validator.validate_allocation_mode.return_value = ValidationResult(
            True, "Valid"
        )

        mock_trade_validator = Mock(spec=ITradeValidator)

        # Should not raise exception
        aggregator = TradeAggregator(
            initial_capital=100000.0,
            allocation_mode="reinvestment",
            allocation_validator=mock_allocation_validator,
            trade_validator=mock_trade_validator,
        )

        # Verify dependencies were injected
        assert aggregator._allocation_validator == mock_allocation_validator
        assert aggregator._trade_validator == mock_trade_validator
        mock_allocation_validator.validate_allocation_mode.assert_called_once_with("reinvestment")

    def test_transaction_cost_model_accepts_validator_injection(self):
        """Test that transaction cost model factory accepts dependency injection."""
        from portfolio_backtester.trading.transaction_costs import get_transaction_cost_model

        mock_validator = Mock(spec=IModelValidator)
        mock_validator.validate_model_name.return_value = ValidationResult(True, "Valid")

        config = {"transaction_cost_model": "realistic"}

        # Should not raise exception
        model = get_transaction_cost_model(config, model_validator=mock_validator)

        # Verify validator was used
        mock_validator.validate_model_name.assert_called_once_with("realistic")
        assert model is not None


class TestBackwardCompatibility:
    """Test that DIP implementation maintains backward compatibility."""

    def test_portfolio_value_tracker_works_without_injection(self):
        """Test that PortfolioValueTracker works without explicit dependency injection."""
        from portfolio_backtester.trading.portfolio_value_tracker import PortfolioValueTracker

        # Should work with default factory
        tracker = PortfolioValueTracker(allocation_mode="reinvestment")
        assert tracker._allocation_validator is not None

    def test_trade_aggregator_works_without_injection(self):
        """Test that TradeAggregator works without explicit dependency injection."""
        from portfolio_backtester.strategies._core.base.base.trade_aggregator import TradeAggregator

        # Should work with default factories
        aggregator = TradeAggregator(initial_capital=100000.0, allocation_mode="reinvestment")
        assert aggregator._allocation_validator is not None
        assert aggregator._trade_validator is not None

    def test_transaction_cost_model_works_without_injection(self):
        """Test that transaction cost model factory works without explicit dependency injection."""
        from portfolio_backtester.trading.transaction_costs import get_transaction_cost_model

        config = {"transaction_cost_model": "realistic"}

        # Should work with default factory
        model = get_transaction_cost_model(config)
        assert model is not None


class TestValidationErrorHandling:
    """Test proper error handling with validation results."""

    def test_invalid_allocation_mode_raises_value_error(self):
        """Test that invalid allocation mode raises ValueError with proper message."""
        from portfolio_backtester.trading.portfolio_value_tracker import PortfolioValueTracker

        with pytest.raises(ValueError) as exc_info:
            PortfolioValueTracker(allocation_mode="invalid_mode")

        error_message = str(exc_info.value)
        assert "invalid_mode" in error_message.lower()

    def test_invalid_model_name_raises_value_error(self):
        """Test that invalid model name raises ValueError with proper message."""
        from portfolio_backtester.trading.transaction_costs import get_transaction_cost_model

        config = {"transaction_cost_model": "unsupported_model"}

        with pytest.raises(ValueError) as exc_info:
            get_transaction_cost_model(config)

        error_message = str(exc_info.value)
        assert "unsupported" in error_message.lower()
        assert "unsupported_model" in error_message
