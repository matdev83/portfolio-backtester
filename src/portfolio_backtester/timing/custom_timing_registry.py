"""
Custom timing controller registry and factory.

This class serves as a façade over the comprehensive registration system components
(RegistrationManager, RegistrationValidator, and RegistryLister). The internal
implementation uses the advanced registration system for better separation of concerns,
while maintaining the original public API for backward compatibility.

Supports dynamic loading and registration of custom timing controllers.
"""

import importlib
import logging
from typing import Dict, Type, Optional, Any, List
from ..api_stability import api_stable

import pandas as pd

from .timing_controller import TimingController
from ..registration import RegistrationManager, RegistrationValidator, RegistryLister
from ..interfaces.attribute_accessor_interface import (
    IClassAttributeAccessor,
    create_class_attribute_accessor,
)
from ..interfaces.time_based_timing_interface import (
    ITimeBasedTiming,
    create_time_based_timing,
)


logger = logging.getLogger(__name__)


def _validate_timing_controller(data: Dict[str, Any]) -> List[str]:
    """Custom validation function for timing controllers."""
    errors = []

    component = data.get("component")
    if component is not None:
        # Check if component inherits from TimingController
        if not isinstance(component, type) or not issubclass(component, TimingController):
            errors.append("Component must be a class that inherits from TimingController")

    return errors


class CustomTimingRegistry:
    """Registry for custom timing controller classes."""

    # Class-level instances for backward compatibility using comprehensive registration system
    _validator = RegistrationValidator(
        custom_validators=[_validate_timing_controller],
        naming_pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$",  # Alphanumeric with underscores
        reserved_names={"timing", "controller", "base", "system"},
        max_alias_count=5,
    )
    _manager = RegistrationManager(validator=_validator)
    _lister = RegistryLister(_manager)

    @classmethod
    def register(
        cls,
        name: str,
        controller_class: Type[TimingController],
        aliases: Optional[list] = None,
    ):
        """
        Register a custom timing controller class.

        Args:
            name: Unique name for the controller
            controller_class: Controller class (must inherit from TimingController)
            aliases: Optional list of alternative names
        """
        # Use the comprehensive registration system
        cls._manager.register(
            name,
            controller_class,
            aliases=aliases,
            metadata={"type": "timing_controller", "builtin": False},
        )

    @classmethod
    @api_stable(version="1.0", strict_params=True, strict_return=False)
    def get(cls, name: str) -> Optional[Type[TimingController]]:
        """
        Get a registered timing controller class by name.

        Args:
            name: Controller name or alias

        Returns:
            Controller class or None if not found
        """
        # Use the manager to get the controller
        return cls._manager.get_component(name)

    @classmethod
    def list_registered(cls) -> Dict[str, str]:
        """
        List all registered timing controllers.

        Returns:
            Dictionary mapping names to class names
        """
        result = {}
        components = cls._lister.list_components()
        for name in components:
            info = cls._lister.get_component_info(name)
            if info and info["component"]:
                component_class = info["component"]
                result[name] = f"{component_class.__module__}.{component_class.__name__}"
        return result

    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        Unregister a timing controller.

        Args:
            name: Controller name to unregister

        Returns:
            True if unregistered, False if not found
        """
        return cls._manager.deregister(name)

    @classmethod
    def clear(cls) -> None:
        """Clear user-registered controllers while keeping built-ins."""
        # Clear all but preserve built-in controllers
        all_components = cls._lister.list_components()
        for name in all_components:
            info = cls._lister.get_component_info(name)
            if info and not info.get("metadata", {}).get("builtin"):
                cls._manager.deregister(name)


class TimingControllerFactory:
    """Factory for creating timing controller instances."""

    @staticmethod
    def create_controller(config: Dict[str, Any]) -> TimingController:
        """
        Create a timing controller instance based on configuration.

        Args:
            config: Timing configuration dictionary

        Returns:
            TimingController instance

        Raises:
            ValueError: If configuration is invalid or controller cannot be created
        """
        mode = config.get("mode", "time_based")

        if mode == "time_based":
            from .time_based_timing import TimeBasedTiming
            from ..interfaces.timing_state_interface import create_timing_state

            # Create timing state using interface factory
            timing_state = create_timing_state()
            return TimeBasedTiming(config, timing_state)

        elif mode == "signal_based":
            from .signal_based_timing import SignalBasedTiming
            from ..interfaces.timing_state_interface import create_timing_state

            # Create timing state using interface factory
            timing_state = create_timing_state()
            return SignalBasedTiming(config, timing_state)

        elif mode == "custom":
            return TimingControllerFactory._create_custom_controller(config)

        else:
            raise ValueError(f"Unknown timing mode: {mode}")

    @staticmethod
    def _create_custom_controller(config: Dict[str, Any]) -> TimingController:
        """Create a custom timing controller instance."""
        controller_class_name = config.get("custom_controller_class")
        if not controller_class_name:
            raise ValueError("custom_controller_class is required for custom timing mode")

        controller_params = config.get("custom_controller_params", {})

        # First, try to get from registry
        controller_class = CustomTimingRegistry.get(controller_class_name)

        if controller_class is None:
            # Try to dynamically import the class
            controller_class = TimingControllerFactory._import_class(controller_class_name)

        if controller_class is None:
            raise ValueError(f"Cannot find timing controller class: {controller_class_name}")

        try:
            # Create instance with custom parameters
            instance = controller_class(config, **controller_params)
            logger.info(f"Created custom timing controller: {controller_class_name}")
            return instance

        except Exception as e:
            raise ValueError(
                f"Failed to create custom timing controller {controller_class_name}: {e}"
            )

    @staticmethod
    def _import_class(
        class_path: str, class_accessor: Optional[IClassAttributeAccessor] = None
    ) -> Optional[Type[TimingController]]:
        """
        Dynamically import a class from a module path.

        Args:
            class_path: Fully qualified class path (e.g., 'mymodule.MyClass')
            class_accessor: Injected accessor for class loading (DIP)

        Returns:
            Class object or None if import fails
        """
        # Use dependency injection for class access
        accessor = class_accessor or create_class_attribute_accessor()

        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            controller_class = accessor.get_class_from_module(module, class_name)

            if not issubclass(controller_class, TimingController):
                logger.error(f"Class {class_path} does not inherit from TimingController")
                return None

            logger.info(f"Successfully imported custom timing controller: {class_path}")
            return controller_class

        except (ImportError, AttributeError, TypeError, ValueError) as e:
            logger.error(f"Failed to import timing controller class {class_path}: {e}")
            return None


# Decorator for easy registration of custom timing controllers
def register_timing_controller(name: str, aliases: Optional[list] = None):
    """
    Decorator to register a custom timing controller.

    Args:
        name: Unique name for the controller
        aliases: Optional list of alternative names

    Example:
        @register_timing_controller('my_custom_timing', aliases=['mct'])
        class MyCustomTimingController(TimingController):
            pass
    """

    def decorator(controller_class: Type[TimingController]):
        # Register as built-in since this is defined at module import time
        CustomTimingRegistry._manager.register(
            name,
            controller_class,
            aliases=aliases,
            metadata={"type": "timing_controller", "builtin": True},
        )
        return controller_class

    return decorator


# Built-in custom timing controllers can be registered here
@register_timing_controller("adaptive_timing", aliases=["adaptive"])
class AdaptiveTimingController(TimingController):
    """
    Example adaptive timing controller that adjusts frequency based on market volatility.
    This is a demonstration of how custom timing controllers can be implemented.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        time_based_timing: Optional[ITimeBasedTiming] = None,
        volatility_threshold: float = 0.02,
        **kwargs,
    ):
        """
        Initialize adaptive timing controller.

        Args:
            config: Timing configuration
            time_based_timing: Time-based timing implementation for dependency injection
            volatility_threshold: Volatility threshold for frequency adjustment
        """
        super().__init__(config)
        self.volatility_threshold = volatility_threshold
        self.base_frequency = config.get("base_frequency", "M")
        self.high_vol_frequency = config.get("high_vol_frequency", "W")
        self.low_vol_frequency = config.get("low_vol_frequency", "Q")

        # Use dependency injection for TimeBasedTiming operations
        if time_based_timing is None:
            time_based_timing = create_time_based_timing(config)
        self._time_based_timing = time_based_timing

        logger.info(
            f"Initialized AdaptiveTimingController with volatility threshold {volatility_threshold}"
        )

    def should_generate_signal(self, current_date, strategy) -> bool:
        """
        Determine if signal should be generated based on market volatility.

        This is a simplified example - real implementation would analyze
        market volatility and adjust timing accordingly.
        """
        # For demonstration, use a simple time-based approach
        # Real implementation would analyze volatility metrics

        # Default to monthly rebalancing
        if hasattr(strategy, "get_market_volatility"):
            try:
                volatility = strategy.get_market_volatility(current_date)
                try:
                    numeric_vol = float(volatility)
                except (TypeError, ValueError):
                    numeric_vol = None

                if numeric_vol is not None and numeric_vol > self.volatility_threshold:
                    # High volatility - rebalance more frequently
                    return self._should_rebalance_frequency(current_date, self.high_vol_frequency)
                else:
                    # Low volatility - rebalance less frequently
                    return self._should_rebalance_frequency(current_date, self.low_vol_frequency)
            except AttributeError:
                pass

        # Fallback to base frequency
        return self._should_rebalance_frequency(current_date, self.base_frequency)

    def _should_rebalance_frequency(self, current_date, frequency: str) -> bool:
        """Helper method to check if rebalancing should occur for given frequency."""
        # Simplified frequency check - real implementation would be more sophisticated
        if frequency == "D":
            return True
        elif frequency == "W":
            return bool(current_date.weekday() == 0)  # Monday
        elif frequency == "M":
            return bool(current_date.day <= 7 and current_date.weekday() < 5)  # First week, weekday
        elif frequency == "Q":
            return bool(
                current_date.month % 3 == 1 and current_date.day <= 7
            )  # First week of quarter
        else:
            return False

    def get_rebalance_dates(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        available_dates: pd.DatetimeIndex,
        strategy: Any,
    ) -> pd.DatetimeIndex:
        """
        Get rebalance dates for adaptive timing.

        This is a simplified implementation for demonstration.
        """
        # Use dependency-injected time-based timing
        # For simplicity, use base frequency for date generation
        # Real implementation would analyze historical volatility
        self._time_based_timing.set_frequency(self.base_frequency)

        result = self._time_based_timing.get_rebalance_dates(
            start_date, end_date, available_dates, strategy
        )
        return result


# Example of a momentum-based timing controller
@register_timing_controller("momentum_timing", aliases=["momentum"])
class MomentumTimingController(TimingController):
    """
    Example momentum-based timing controller that rebalances based on momentum signals.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        time_based_timing: Optional[ITimeBasedTiming] = None,
        momentum_period: int = 20,
        **kwargs,
    ):
        """
        Initialize momentum timing controller.

        Args:
            config: Timing configuration
            time_based_timing: Time-based timing implementation for dependency injection
            momentum_period: Period for momentum calculation
        """
        super().__init__(config)
        self.momentum_period = momentum_period
        self.last_momentum = None

        # Use dependency injection for TimeBasedTiming operations
        if time_based_timing is None:
            time_based_timing = create_time_based_timing(config)
        self._time_based_timing = time_based_timing

        logger.info(
            f"Initialized MomentumTimingController with {momentum_period}-day momentum period"
        )

    def should_generate_signal(self, current_date, strategy) -> bool:
        """
        Generate signal based on momentum changes.

        This is a simplified example for demonstration.
        """
        # This would need access to price data to calculate momentum
        # For demonstration, use a simple time-based approach

        # Rebalance weekly for momentum strategies
        return bool(current_date.weekday() == 0)  # Monday

    def get_rebalance_dates(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        available_dates: pd.DatetimeIndex,
        strategy: Any,
    ) -> pd.DatetimeIndex:
        """Get weekly rebalance dates for momentum timing."""
        # Use dependency-injected time-based timing
        self._time_based_timing.set_frequency("W")

        result = self._time_based_timing.get_rebalance_dates(
            start_date, end_date, available_dates, strategy
        )
        return result
