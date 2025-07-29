"""
Custom timing controller registry and factory.
Supports dynamic loading and registration of custom timing controllers.
"""

import importlib
import logging
from typing import Dict, Type, Optional, Any, Callable, List
from abc import ABC
from ..api_stability import api_stable

import pandas as pd

from .timing_controller import TimingController


logger = logging.getLogger(__name__)


class CustomTimingRegistry:
    """Registry for custom timing controller classes."""
    
    _registry: Dict[str, Type[TimingController]] = {}
    _aliases: Dict[str, str] = {}
    # Unified lookup cache for O(1) access - combines registry and resolved aliases
    _unified_lookup: Dict[str, Type[TimingController]] = {}
    
    @classmethod
    def register(cls, name: str, controller_class: Type[TimingController], aliases: Optional[list] = None):
        """
        Register a custom timing controller class.
        
        Args:
            name: Unique name for the controller
            controller_class: Controller class (must inherit from TimingController)
            aliases: Optional list of alternative names
        """
        if not issubclass(controller_class, TimingController):
            raise ValueError(f"Controller class {controller_class} must inherit from TimingController")
        
        cls._registry[name] = controller_class
        # Update unified lookup cache
        cls._unified_lookup[name] = controller_class
        logger.info(f"Registered custom timing controller: {name} -> {controller_class}")
        
        # Register aliases
        if aliases:
            for alias in aliases:
                cls._aliases[alias] = name
                # Update unified lookup cache for aliases
                cls._unified_lookup[alias] = controller_class
                logger.debug(f"Registered alias: {alias} -> {name}")
    
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
        # Single O(1) lookup from unified cache
        return cls._unified_lookup.get(name)
    
    @classmethod
    def list_registered(cls) -> Dict[str, str]:
        """
        List all registered timing controllers.
        
        Returns:
            Dictionary mapping names to class names
        """
        result = {}
        for name, controller_class in cls._registry.items():
            result[name] = f"{controller_class.__module__}.{controller_class.__name__}"
        
        # Add aliases
        for alias, actual_name in cls._aliases.items():
            if actual_name in cls._registry:
                controller_class = cls._registry[actual_name]
                result[f"{alias} (alias)"] = f"{controller_class.__module__}.{controller_class.__name__}"
        
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
        if name in cls._registry:
            del cls._registry[name]
            # Remove from unified lookup cache
            cls._unified_lookup.pop(name, None)
            
            # Remove any aliases pointing to this name
            aliases_to_remove = [alias for alias, target in cls._aliases.items() if target == name]
            for alias in aliases_to_remove:
                del cls._aliases[alias]
                # Remove alias from unified lookup cache
                cls._unified_lookup.pop(alias, None)
            logger.info(f"Unregistered timing controller: {name}")
            return True
        
        return False
    
    @classmethod
    def clear(cls) -> None:
        """Clear user-registered controllers while keeping built-ins."""
        # Determine built-ins the first time we call clear
        if not hasattr(cls, "_built_in_names"):
            cls._built_in_names = set(cls._registry.keys())

        # Remove everything that is NOT a built-in
        for name in list(cls._registry.keys()):
            if name not in cls._built_in_names:
                del cls._registry[name]

        # Rebuild alias map excluding removed controllers
        cls._aliases = {alias: tgt for alias, tgt in cls._aliases.items() if tgt in cls._registry}

        # Rebuild unified lookup cache with remaining controllers and aliases
        cls._unified_lookup.clear()
        # Add remaining registry entries
        cls._unified_lookup.update(cls._registry)
        # Add remaining aliases
        for alias, target_name in cls._aliases.items():
            if target_name in cls._registry:
                cls._unified_lookup[alias] = cls._registry[target_name]

        logger.info("Cleared custom timing controllers (built-ins preserved)")


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
        mode = config.get('mode', 'time_based')
        
        if mode == 'time_based':
            from .time_based_timing import TimeBasedTiming
            return TimeBasedTiming(config)
        
        elif mode == 'signal_based':
            from .signal_based_timing import SignalBasedTiming
            return SignalBasedTiming(config)
        
        elif mode == 'custom':
            return TimingControllerFactory._create_custom_controller(config)
        
        else:
            raise ValueError(f"Unknown timing mode: {mode}")
    
    @staticmethod
    def _create_custom_controller(config: Dict[str, Any]) -> TimingController:
        """Create a custom timing controller instance."""
        controller_class_name = config.get('custom_controller_class')
        if not controller_class_name:
            raise ValueError("custom_controller_class is required for custom timing mode")
        
        controller_params = config.get('custom_controller_params', {})
        
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
            raise ValueError(f"Failed to create custom timing controller {controller_class_name}: {e}")
    
    @staticmethod
    def _import_class(class_path: str) -> Optional[Type[TimingController]]:
        """
        Dynamically import a class from a module path.
        
        Args:
            class_path: Fully qualified class path (e.g., 'mymodule.MyClass')
            
        Returns:
            Class object or None if import fails
        """
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            controller_class = getattr(module, class_name)
            
            if not issubclass(controller_class, TimingController):
                logger.error(f"Class {class_path} does not inherit from TimingController")
                return None
            
            logger.info(f"Successfully imported custom timing controller: {class_path}")
            return controller_class
        
        except (ImportError, AttributeError, ValueError) as e:
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
        CustomTimingRegistry.register(name, controller_class, aliases)
        return controller_class
    
    return decorator


# Built-in custom timing controllers can be registered here
@register_timing_controller('adaptive_timing', aliases=['adaptive'])
class AdaptiveTimingController(TimingController):
    """
    Example adaptive timing controller that adjusts frequency based on market volatility.
    This is a demonstration of how custom timing controllers can be implemented.
    """
    
    def __init__(self, config: Dict[str, Any], volatility_threshold: float = 0.02, **kwargs):
        """
        Initialize adaptive timing controller.
        
        Args:
            config: Timing configuration
            volatility_threshold: Volatility threshold for frequency adjustment
        """
        super().__init__(config)
        self.volatility_threshold = volatility_threshold
        self.base_frequency = config.get('base_frequency', 'M')
        self.high_vol_frequency = config.get('high_vol_frequency', 'W')
        self.low_vol_frequency = config.get('low_vol_frequency', 'Q')
        
        logger.info(f"Initialized AdaptiveTimingController with volatility threshold {volatility_threshold}")
    
    def should_generate_signal(self, current_date, strategy) -> bool:
        """
        Determine if signal should be generated based on market volatility.
        
        This is a simplified example - real implementation would analyze
        market volatility and adjust timing accordingly.
        """
        # For demonstration, use a simple time-based approach
        # Real implementation would analyze volatility metrics
        
        # Default to monthly rebalancing
        if hasattr(strategy, 'get_market_volatility'):
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
        if frequency == 'D':
            return True
        elif frequency == 'W':
            return current_date.weekday() == 0  # Monday
        elif frequency == 'M':
            return current_date.day <= 7 and current_date.weekday() < 5  # First week, weekday
        elif frequency == 'Q':
            return current_date.month % 3 == 1 and current_date.day <= 7  # First week of quarter
        else:
            return False
    
    def get_rebalance_dates(self, start_date: pd.Timestamp, end_date: pd.Timestamp, available_dates: List[pd.Timestamp], strategy: Any) -> List[pd.Timestamp]:
        """
        Get rebalance dates for adaptive timing.
        
        This is a simplified implementation for demonstration.
        """
        from .time_based_timing import TimeBasedTiming
        
        # For simplicity, use base frequency for date generation
        # Real implementation would analyze historical volatility
        temp_config = self.config.copy()
        temp_config['rebalance_frequency'] = self.base_frequency
        
        time_based = TimeBasedTiming(temp_config)
        return time_based.get_rebalance_dates(start_date, end_date, available_dates, strategy)


# Example of a momentum-based timing controller
@register_timing_controller('momentum_timing', aliases=['momentum'])
class MomentumTimingController(TimingController):
    """
    Example momentum-based timing controller that rebalances based on momentum signals.
    """
    
    def __init__(self, config: Dict[str, Any], momentum_period: int = 20, **kwargs):
        """
        Initialize momentum timing controller.
        
        Args:
            config: Timing configuration
            momentum_period: Period for momentum calculation
        """
        super().__init__(config)
        self.momentum_period = momentum_period
        self.last_momentum = None
        
        logger.info(f"Initialized MomentumTimingController with {momentum_period}-day momentum period")
    
    def should_generate_signal(self, current_date, strategy) -> bool:
        """
        Generate signal based on momentum changes.
        
        This is a simplified example for demonstration.
        """
        # This would need access to price data to calculate momentum
        # For demonstration, use a simple time-based approach
        
        # Rebalance weekly for momentum strategies
        return current_date.weekday() == 0  # Monday
    
    def get_rebalance_dates(self, start_date: pd.Timestamp, end_date: pd.Timestamp, available_dates: List[pd.Timestamp], strategy: Any) -> List[pd.Timestamp]:
        """Get weekly rebalance dates for momentum timing."""
        from .time_based_timing import TimeBasedTiming
        
        temp_config = self.config.copy()
        temp_config['rebalance_frequency'] = 'W'
        
        time_based = TimeBasedTiming(temp_config)
        return time_based.get_rebalance_dates(start_date, end_date, available_dates, strategy)