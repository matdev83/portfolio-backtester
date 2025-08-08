"""
Strategy validation system for ensuring discovered strategies are usable.

Provides comprehensive validation of strategy classes to catch issues early
and provide detailed feedback about what's wrong with broken strategies.
"""

import inspect
from typing import List, Type, Dict, Optional, Tuple
import logging

# Import BaseStrategy using the same package path as strategy modules use
# This ensures we get the same BaseStrategy instance that strategies inherit from
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # For type checking, use the original import
    from ..base.base_strategy import BaseStrategy
else:
    # At runtime, use the package path to get the correct BaseStrategy instance
    _base_strategy_module = importlib.import_module(
        "portfolio_backtester.strategies.base.base_strategy"
    )
    BaseStrategy = _base_strategy_module.BaseStrategy

logger = logging.getLogger(__name__)


class StrategyValidationError(Exception):
    """Raised when a strategy fails validation."""

    def __init__(self, strategy_name: str, errors: List[str]):
        self.strategy_name = strategy_name
        self.errors = errors
        error_msg = f"Strategy '{strategy_name}' failed validation:\n" + "\n".join(
            f"  - {error}" for error in errors
        )
        super().__init__(error_msg)


class StrategyValidator:
    """
    Validates discovered strategy classes to ensure they're properly implemented.

    Performs comprehensive checks including:
    - Abstract method implementation
    - Required method existence
    - Constructor signature validation
    - Method call testing (where safe)
    - Class hierarchy verification
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator.

        Args:
            strict_mode: If True, applies stricter validation rules
        """
        self.strict_mode = strict_mode

        # Required methods that all strategies must have
        self.required_methods = {
            "generate_signals": "Generate trading signals",
            "__init__": "Constructor for initialization",
            "tunable_parameters": "Get configurable parameters",
        }

        # Methods that should be callable without side effects
        self.safe_callable_methods = {
            "tunable_parameters": "Should return parameter definitions",
            "get_minimum_required_periods": "Should return minimum data periods",
        }

    def validate_strategy(
        self, strategy_class: Type[BaseStrategy], strategy_name: str
    ) -> List[str]:
        """
        Validate a single strategy class comprehensively.

        Args:
            strategy_class: The strategy class to validate
            strategy_name: Name of the strategy for error reporting

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        try:
            # 1. Basic type validation
            errors.extend(self._validate_base_class(strategy_class, strategy_name))

            # 2. Abstract method validation
            errors.extend(self._validate_abstract_methods(strategy_class, strategy_name))

            # 3. Required method validation
            errors.extend(self._validate_required_methods(strategy_class, strategy_name))

            # 4. Constructor validation
            errors.extend(self._validate_constructor(strategy_class, strategy_name))

            # 5. Safe method call validation
            errors.extend(self._validate_safe_method_calls(strategy_class, strategy_name))

            # 6. Strict mode additional validations
            if self.strict_mode:
                errors.extend(self._validate_strict_mode(strategy_class, strategy_name))

        except Exception as e:
            errors.append(f"Validation failed with unexpected error: {str(e)}")
            logger.warning(f"Unexpected error validating {strategy_name}: {e}")

        return errors

    def validate_strategy_safe(
        self, strategy_class: Type[BaseStrategy], strategy_name: str
    ) -> Tuple[bool, List[str]]:
        """
        Safely validate a strategy, catching and logging any exceptions.

        Args:
            strategy_class: The strategy class to validate
            strategy_name: Name of the strategy for error reporting

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        try:
            errors = self.validate_strategy(strategy_class, strategy_name)
            return len(errors) == 0, errors
        except Exception as e:
            error_msg = f"Validation crashed with exception: {str(e)}"
            logger.error(f"Strategy validation crashed for {strategy_name}: {e}", exc_info=True)
            return False, [error_msg]

    def _validate_base_class(
        self, strategy_class: Type[BaseStrategy], strategy_name: str
    ) -> List[str]:
        """Validate that the class properly inherits from BaseStrategy."""
        errors = []

        if not issubclass(strategy_class, BaseStrategy):
            errors.append("Must be a subclass of BaseStrategy")

        # Check if it's one of the base classes that shouldn't be instantiated
        base_class_names = {
            "BaseStrategy",
            "SignalStrategy",
            "PortfolioStrategy",
            "BaseMetaStrategy",
        }
        if strategy_class.__name__ in base_class_names:
            errors.append(
                f"Cannot register base class '{strategy_class.__name__}' as a concrete strategy"
            )

        return errors

    def _validate_abstract_methods(
        self, strategy_class: Type[BaseStrategy], strategy_name: str
    ) -> List[str]:
        """Validate that all abstract methods are implemented."""
        errors = []

        if inspect.isabstract(strategy_class):
            abstract_methods: set = getattr(strategy_class, "__abstractmethods__", set())
            if abstract_methods:
                method_list = ", ".join(sorted(abstract_methods))
                errors.append(f"Strategy is abstract - missing implementations: {method_list}")

        return errors

    def _validate_required_methods(
        self, strategy_class: Type[BaseStrategy], strategy_name: str
    ) -> List[str]:
        """Validate that all required methods exist."""
        errors = []

        for method_name, description in self.required_methods.items():
            if not hasattr(strategy_class, method_name):
                errors.append(f"Missing required method '{method_name}' - {description}")
            else:
                # Check if method is callable
                method = getattr(strategy_class, method_name)
                if not callable(method):
                    errors.append(f"'{method_name}' exists but is not callable")

        return errors

    def _validate_constructor(
        self, strategy_class: Type[BaseStrategy], strategy_name: str
    ) -> List[str]:
        """Validate the constructor signature."""
        errors = []

        try:
            init_method = strategy_class.__init__
            signature = inspect.signature(init_method)
            params = list(signature.parameters.keys())

            # Should have at least 'self' and one config parameter
            if len(params) < 2:
                errors.append("Constructor must accept at least one parameter (besides self)")

            # Check for common parameter names that indicate proper strategy pattern
            expected_params = {"strategy_config", "strategy_params", "config", "params"}
            param_names = set(params[1:])  # Exclude 'self'

            if not param_names.intersection(expected_params):
                errors.append(
                    f"Constructor should accept a configuration parameter (expected one of: {expected_params})"
                )

        except Exception as e:
            errors.append(f"Could not inspect constructor signature: {str(e)}")

        return errors

    def _validate_safe_method_calls(
        self, strategy_class: Type[BaseStrategy], strategy_name: str
    ) -> List[str]:
        """Test methods that should be safely callable without side effects."""
        errors = []

        for method_name, description in self.safe_callable_methods.items():
            if hasattr(strategy_class, method_name):
                try:
                    method = getattr(strategy_class, method_name)

                    # Only test class methods (not instance methods)
                    result = None
                    if isinstance(method, classmethod):
                        # For class methods, call the underlying function with the class as first arg
                        result = method.__func__(strategy_class)
                    elif inspect.ismethod(method) and method.__self__ is strategy_class:
                        result = method()

                    if result is not None:
                        # Basic sanity check on result
                        if method_name == "tunable_parameters":
                            if not isinstance(result, dict):
                                errors.append(
                                    f"tunable_parameters() should return a dict, got {type(result)}"
                                )

                        elif method_name == "get_minimum_required_periods":
                            if not isinstance(result, (int, type(None))):
                                errors.append(
                                    f"get_minimum_required_periods() should return int or None, got {type(result)}"
                                )

                except Exception as e:
                    errors.append(f"Method '{method_name}' failed when called: {str(e)}")

        return errors

    def _validate_strict_mode(
        self, strategy_class: Type[BaseStrategy], strategy_name: str
    ) -> List[str]:
        """Additional validations for strict mode."""
        errors = []

        # Check for proper docstring
        if not strategy_class.__doc__ or len(strategy_class.__doc__.strip()) < 10:
            errors.append("Strategy should have a meaningful docstring")

        # Check that generate_signals method has proper signature
        if hasattr(strategy_class, "generate_signals"):
            try:
                sig = inspect.signature(strategy_class.generate_signals)
                params = list(sig.parameters.keys())

                # Should have self, asset_data, current_date, and optionally other params
                if len(params) < 3:  # self + at least 2 required params
                    errors.append(
                        "generate_signals should accept asset_data and current_date parameters"
                    )

            except Exception as e:
                errors.append(f"Could not inspect generate_signals signature: {str(e)}")

        return errors


class BatchStrategyValidator:
    """Validates multiple strategies efficiently with detailed reporting."""

    def __init__(self, validator: Optional[StrategyValidator] = None):
        """
        Initialize batch validator.

        Args:
            validator: Strategy validator to use (creates default if None)
        """
        self.validator = validator or StrategyValidator()
        self.results: Dict[str, List[str]] = {}

    def validate_strategies(
        self, strategies: Dict[str, Type[BaseStrategy]]
    ) -> Dict[str, List[str]]:
        """
        Validate multiple strategies and return comprehensive results.

        Args:
            strategies: Dictionary of strategy_name -> strategy_class

        Returns:
            Dictionary mapping strategy names to lists of validation errors
        """
        results = {}

        for strategy_name, strategy_class in strategies.items():
            is_valid, errors = self.validator.validate_strategy_safe(strategy_class, strategy_name)
            results[strategy_name] = errors

            if not is_valid:
                logger.warning(
                    f"Strategy '{strategy_name}' failed validation: {len(errors)} errors"
                )
                for error in errors[:3]:  # Log first 3 errors
                    logger.warning(f"  - {error}")

        self.results = results
        return results

    def get_valid_strategies(
        self, strategies: Dict[str, Type[BaseStrategy]]
    ) -> Dict[str, Type[BaseStrategy]]:
        """
        Get only the strategies that pass validation.

        Args:
            strategies: Dictionary of strategy_name -> strategy_class

        Returns:
            Dictionary of valid strategies only
        """
        validation_results = self.validate_strategies(strategies)

        valid_strategies = {
            name: strategy_class
            for name, strategy_class in strategies.items()
            if len(validation_results.get(name, [])) == 0
        }

        invalid_count = len(strategies) - len(valid_strategies)
        if invalid_count > 0:
            logger.info(f"Filtered out {invalid_count} invalid strategies during validation")

        return valid_strategies

    def generate_validation_report(self) -> str:
        """Generate a comprehensive validation report."""
        if not self.results:
            return "No validation results available"

        total_strategies = len(self.results)
        valid_strategies = sum(1 for errors in self.results.values() if len(errors) == 0)
        invalid_strategies = total_strategies - valid_strategies

        report_lines = [
            "=== Strategy Validation Report ===",
            f"Total Strategies: {total_strategies}",
            f"Valid: {valid_strategies}",
            f"Invalid: {invalid_strategies}",
            "",
        ]

        if invalid_strategies > 0:
            report_lines.append("❌ Invalid Strategies:")
            for strategy_name, errors in self.results.items():
                if errors:
                    report_lines.append(f"  {strategy_name}:")
                    for error in errors:
                        report_lines.append(f"    - {error}")
                    report_lines.append("")

        return "\n".join(report_lines)
