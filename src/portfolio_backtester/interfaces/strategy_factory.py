"""
StrategyFactory interface and implementations for polymorphic strategy creation.

Replaces isinstance checks in strategy resolution with proper polymorphic behavior.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


class IStrategyFactory(ABC):
    """Abstract interface for creating strategy instances."""

    @abstractmethod
    def create_strategy(self, strategy_spec: Any, params: Dict[str, Any]) -> Any:
        """
        Create a strategy instance from specification and parameters.

        Args:
            strategy_spec: Strategy specification (string name, class, or dict)
            params: Strategy parameters

        Returns:
            Strategy instance

        Raises:
            ValueError: If strategy specification is invalid
            ImportError: If strategy class cannot be imported
        """
        pass


class DefaultStrategyFactory(IStrategyFactory):
    """Default implementation for strategy creation."""

    def create_strategy(self, strategy_spec: Any, params: Dict[str, Any]) -> Any:
        """
        Create strategy using polymorphic specification handling.

        Args:
            strategy_spec: Strategy specification
            params: Strategy parameters

        Returns:
            Strategy instance

        Raises:
            ValueError: If strategy cannot be created
        """
        # Use polymorphic strategy factory to eliminate isinstance violations
        from .strategy_specification_handler import create_polymorphic_strategy_factory

        factory = create_polymorphic_strategy_factory()
        return factory.create_strategy(strategy_spec, params)


class StrategyFactoryRegistry:
    """Registry for managing different strategy factory implementations."""

    _factories: Dict[str, IStrategyFactory] = {}
    _default_factory: IStrategyFactory = DefaultStrategyFactory()

    @classmethod
    def register_factory(cls, name: str, factory: IStrategyFactory) -> None:
        """
        Register a strategy factory.

        Args:
            name: Factory name
            factory: Factory implementation
        """
        cls._factories[name] = factory

    @classmethod
    def get_factory(cls, name: str = "default") -> IStrategyFactory:
        """
        Get a registered factory or the default factory.

        Args:
            name: Factory name

        Returns:
            Factory implementation
        """
        if name == "default":
            return cls._default_factory
        return cls._factories.get(name, cls._default_factory)
