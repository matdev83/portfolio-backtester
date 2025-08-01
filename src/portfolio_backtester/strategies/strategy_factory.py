"""Strategy factory for dynamic instantiation of strategies."""

from typing import Dict, Any, Type, Set
import logging

from .base.base_strategy import BaseStrategy
from . import enumerate_strategies_with_params

logger = logging.getLogger(__name__)


class StrategyFactory:
    """Factory for creating strategy instances dynamically.
    
    AGENT_NOTE: Do not register strategies manually. This factory uses a dynamic
    discovery mechanism to find and register all available strategies.
    """
    
    _registry: Dict[str, Type[BaseStrategy]] = {}
    _circular_detection: Set[str] = set()
    _registry_populated = False
    
    @classmethod
    def _populate_registry_if_needed(cls) -> None:
        """Populate the registry by discovering strategies if not already done."""
        if not cls._registry_populated:
            # The snake_case keys from enumerate_strategies_with_params are not what's
            # used for strategy_class lookups. The class name itself is used.
            # We need to populate the registry with ClassName -> ClassType.
            discovered_strategies = enumerate_strategies_with_params()
            for strategy_cls in discovered_strategies.values():
                cls.register_strategy(strategy_cls.__name__, strategy_cls)
            cls._registry_populated = True

    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[BaseStrategy]) -> None:
        """Register a strategy class with the factory."""
        cls._registry[name] = strategy_class
        logger.debug(f"Registered strategy: {name}")
    
    @classmethod
    def create_strategy(cls, strategy_class: str, strategy_params: Dict[str, Any], global_config: Dict[str, Any] = None) -> BaseStrategy:
        """
        Create a strategy instance from class name and parameters.
        
        Args:
            strategy_class: Name of the strategy class
            strategy_params: Parameters for strategy initialization
            global_config: Global configuration (for meta strategies)
            
        Returns:
            Instantiated strategy
            
        Raises:
            ValueError: If strategy class is unknown or circular dependency detected
        """
        cls._populate_registry_if_needed()

        # Check for circular dependencies in meta strategies
        if strategy_class in cls._circular_detection:
            raise ValueError(f"Circular dependency detected for strategy: {strategy_class}")
        
        if strategy_class not in cls._registry:
            raise ValueError(f"Unknown strategy class: {strategy_class}. Available: {list(cls._registry.keys())}")
        
        # Add to circular detection set for meta strategies
        if strategy_class.endswith("MetaStrategy"):
            cls._circular_detection.add(strategy_class)
        
        try:
            strategy_cls = cls._registry[strategy_class]
            
            # Check if this is a meta strategy that accepts global_config
            if strategy_class.endswith("MetaStrategy") and global_config is not None:
                # Try to pass global_config to meta strategies
                try:
                    instance = strategy_cls(strategy_params, global_config)
                except TypeError:
                    # Fallback for meta strategies that don't accept global_config yet
                    instance = strategy_cls(strategy_params)
            else:
                instance = strategy_cls(strategy_params)
            
            return instance
        finally:
            # Remove from circular detection set
            cls._circular_detection.discard(strategy_class)
    
    @classmethod
    def get_registered_strategies(cls) -> Dict[str, Type[BaseStrategy]]:
        """Get all registered strategies."""
        cls._populate_registry_if_needed()
        return cls._registry.copy()
    
    @classmethod
    def clear_registry(cls) -> None:
        """Clear the strategy registry (mainly for testing)."""
        cls._registry.clear()
        cls._circular_detection.clear()
        cls._registry_populated = False