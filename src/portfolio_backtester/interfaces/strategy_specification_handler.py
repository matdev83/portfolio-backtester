"""
Strategy specification handlers for eliminating isinstance violations in strategy factory.

This module provides polymorphic interfaces for handling different types of
strategy specifications without using isinstance checks.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


class IStrategySpecificationHandler(ABC):
    """Interface for handling different types of strategy specifications."""

    @abstractmethod
    def can_handle(self, specification: Any) -> bool:
        """Check if this handler can process the given specification type."""
        pass

    @abstractmethod
    def create_strategy(self, specification: Any, params: Dict[str, Any]) -> Any:
        """Create a strategy instance from the specification."""
        pass


class StringSpecificationHandler(IStrategySpecificationHandler):
    """Handler for string-based strategy specifications."""

    def can_handle(self, specification: Any) -> bool:
        """Check if specification is a string."""
        return hasattr(specification, "lower") and callable(getattr(specification, "lower"))

    def create_strategy(self, specification: Any, params: Dict[str, Any]) -> Any:
        """Create strategy from string specification."""
        from ..utils import _resolve_strategy

        strategy_class = _resolve_strategy(specification)
        if not strategy_class:
            raise ValueError(f"Could not resolve strategy: {specification}")
        return strategy_class(**params)


class TypeSpecificationHandler(IStrategySpecificationHandler):
    """Handler for type/class-based strategy specifications."""

    def can_handle(self, specification: Any) -> bool:
        """Check if specification is a type/class."""
        return (
            hasattr(specification, "__name__")
            and hasattr(specification, "__module__")
            and callable(specification)
        )

    def create_strategy(self, specification: Any, params: Dict[str, Any]) -> Any:
        """Create strategy from type specification."""
        return specification(**params)


class DictSpecificationHandler(IStrategySpecificationHandler):
    """Handler for dictionary-based strategy specifications."""

    def can_handle(self, specification: Any) -> bool:
        """Check if specification is a dictionary."""
        return hasattr(specification, "get") and callable(getattr(specification, "get"))

    def create_strategy(self, specification: Any, params: Dict[str, Any]) -> Any:
        """Create strategy from dict specification."""
        from ..utils import _resolve_strategy

        if "class" not in specification:
            raise ValueError(f"Dict strategy spec missing 'class' key: {specification}")

        strategy_class = _resolve_strategy(specification["class"])
        if not strategy_class:
            raise ValueError(f"Could not resolve strategy: {specification['class']}")

        # Merge parameters from dict and passed params
        merged_params = {**specification.get("params", {}), **params}
        return strategy_class(**merged_params)


class NullSpecificationHandler(IStrategySpecificationHandler):
    """Fallback handler for unsupported specification types."""

    def can_handle(self, specification: Any) -> bool:
        """Always returns True as fallback."""
        return True

    def create_strategy(self, specification: Any, params: Dict[str, Any]) -> Any:
        """Always raises ValueError for unsupported types."""
        raise ValueError(f"Unsupported strategy specification type: {type(specification)}")


class StrategySpecificationHandlerFactory:
    """Factory for creating appropriate strategy specification handlers."""

    def __init__(self):
        self._handlers = [
            StringSpecificationHandler(),
            TypeSpecificationHandler(),
            DictSpecificationHandler(),
            NullSpecificationHandler(),  # Fallback
        ]

    def get_handler(self, specification: Any) -> IStrategySpecificationHandler:
        """Get the appropriate handler for the given specification."""
        for handler in self._handlers:
            if handler.can_handle(specification):
                return handler
        # Should never reach here due to NullHandler fallback
        return self._handlers[-1]


class PolymorphicStrategyFactory:
    """Polymorphic strategy factory that eliminates isinstance violations."""

    def __init__(self):
        self._handler_factory = StrategySpecificationHandlerFactory()

    def create_strategy(self, strategy_spec: Any, params: Dict[str, Any]) -> Any:
        """
        Create strategy instance using polymorphic specification handling.

        Args:
            strategy_spec: Strategy specification (string, type, dict, etc.)
            params: Strategy parameters

        Returns:
            Strategy instance

        Raises:
            ValueError: If strategy cannot be created
        """
        try:
            handler = self._handler_factory.get_handler(strategy_spec)
            return handler.create_strategy(strategy_spec, params)
        except Exception as e:
            logger.error(f"Failed to create strategy from spec {strategy_spec}: {e}")
            raise ValueError(f"Cannot create strategy: {e}")


# Factory function for easy instantiation
def create_polymorphic_strategy_factory() -> PolymorphicStrategyFactory:
    """Create a new polymorphic strategy factory instance."""
    return PolymorphicStrategyFactory()
