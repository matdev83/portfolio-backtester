"""
Strategy resolver interfaces for eliminating isinstance violations.

This module provides polymorphic interfaces for handling different types of
strategy specifications (dict, string) without using isinstance checks.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Union, Dict, cast


class IStrategySpecificationResolver(ABC):
    """Interface for resolving strategy specifications to strategy names."""

    @abstractmethod
    def can_resolve(self, specification: Any) -> bool:
        """Check if this resolver can handle the given specification type."""
        pass

    @abstractmethod
    def resolve_to_name(self, specification: Any) -> Optional[str]:
        """Resolve the specification to a strategy name string."""
        pass


class DictStrategySpecificationResolver(IStrategySpecificationResolver):
    """Resolver for dictionary-based strategy specifications."""

    def can_resolve(self, specification: Any) -> bool:
        """Check if specification is a dictionary."""
        return hasattr(specification, "get") and callable(getattr(specification, "get"))

    def resolve_to_name(self, specification: Any) -> Optional[str]:
        """Extract strategy name from dictionary specification."""
        name = (
            specification.get("name") or specification.get("strategy") or specification.get("type")
        )
        return str(name) if name is not None else None


class StringStrategySpecificationResolver(IStrategySpecificationResolver):
    """Resolver for string-based strategy specifications."""

    def can_resolve(self, specification: Any) -> bool:
        """Check if specification is a string."""
        return hasattr(specification, "lower") and callable(getattr(specification, "lower"))

    def resolve_to_name(self, specification: Any) -> Optional[str]:
        """Return the string specification as-is."""
        return str(specification) if specification is not None else None


class NullStrategySpecificationResolver(IStrategySpecificationResolver):
    """Resolver for invalid/null strategy specifications."""

    def can_resolve(self, specification: Any) -> bool:
        """Always returns True as fallback resolver."""
        return True

    def resolve_to_name(self, specification: Any) -> Optional[str]:
        """Always returns None for invalid specifications."""
        return None


class StrategySpecificationResolverFactory:
    """Factory for creating appropriate strategy specification resolvers."""

    def __init__(self):
        self._resolvers = [
            DictStrategySpecificationResolver(),
            StringStrategySpecificationResolver(),
            NullStrategySpecificationResolver(),  # Fallback
        ]

    def get_resolver(self, specification: Any) -> IStrategySpecificationResolver:
        """Get the appropriate resolver for the given specification."""
        for resolver in self._resolvers:
            if resolver.can_resolve(specification):
                return resolver
        # Should never reach here due to NullResolver fallback
        return self._resolvers[-1]


class IStrategyLookup(ABC):
    """Interface for looking up strategy classes by name."""

    @abstractmethod
    def lookup_strategy(self, name: str, strategy_params: Optional[Dict[str, Any]] = None) -> Any:
        """Look up a strategy class by name."""
        pass


class DefaultStrategyLookup(IStrategyLookup):
    """Default implementation using the SOLID-compliant strategy registry."""

    def lookup_strategy(self, name: str, strategy_params: Optional[Dict[str, Any]] = None) -> Any:
        """Look up strategy using the new strategy registry."""
        try:
            from ..strategies import (
                strategy_factory as _sf_module,
            )  # attribute exposed in strategies/__init__

            StrategyFactory = getattr(_sf_module, "StrategyFactory")
        except Exception:
            from ..strategies._core.strategy_factory import StrategyFactory

        try:
            return StrategyFactory.create_strategy(name, strategy_params or {})
        except ValueError:
            return None


class PolymorphicStrategyResolver:
    """Polymorphic strategy resolver that eliminates isinstance violations."""

    def __init__(self, strategy_lookup: Optional[IStrategyLookup] = None):
        self._specification_factory = StrategySpecificationResolverFactory()
        self._strategy_lookup = strategy_lookup or DefaultStrategyLookup()

    @staticmethod
    def enumerate_strategies_with_params():
        """Enumerate discovered strategies with their tunable parameters."""
        try:
            from ..strategies._core.registry import get_strategy_registry
        except Exception:
            return {}
        registry = get_strategy_registry()
        strategies = registry.get_all_strategies()
        result: Dict[str, Dict[str, Any]] = {}
        for name, cls in strategies.items():
            params: Dict[str, Any] = {}
            try:
                if hasattr(cls, "tunable_parameters") and callable(
                    getattr(cls, "tunable_parameters")
                ):
                    params = cls.tunable_parameters()
            except Exception:
                params = {}
            result[name] = params
        return result

    def resolve_strategy(
        self, specification: Union[str, Dict, Any], strategy_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Resolve strategy specification to a strategy class.

        Args:
            specification: Strategy specification (string name, dict config, etc.)
            strategy_params: Parameters for strategy initialization

        Returns:
            Strategy class or None if not found
        """
        # Step 1: Resolve specification to strategy name
        resolver = self._specification_factory.get_resolver(specification)
        strategy_name = resolver.resolve_to_name(specification)

        # Step 2: Return None if no valid name found
        if strategy_name is None:
            return None

        # Step 3: Look up strategy class by name
        return self._strategy_lookup.lookup_strategy(strategy_name, strategy_params)

    def tunable_parameters(self, strategy_class: Any) -> Dict[str, Any]:
        """Return strategy's tunable parameters if available."""
        try:
            if strategy_class is None:
                return {}
            fn = getattr(strategy_class, "tunable_parameters", None)
            if callable(fn):
                return cast(Dict[str, Any], fn())
        except Exception:
            pass
        return {}

    def is_meta_strategy(self, strategy_class: Any) -> bool:
        """
        Check if a strategy class is a meta strategy.

        Args:
            strategy_class: Strategy class to check

        Returns:
            True if meta strategy, False otherwise
        """
        # Check based on class name convention and strategy name
        if strategy_class is None:
            return False

        # For string strategy names
        if isinstance(strategy_class, str):
            return "meta" in strategy_class.lower()

        # For actual class objects
        class_name = getattr(strategy_class, "__name__", "")
        return "MetaStrategy" in class_name or "meta" in class_name.lower()

    def detect_strategy_type(self, strategy_class: Any) -> str:
        """
        Detect the type of strategy based on class hierarchy.

        Args:
            strategy_class: Strategy class to detect type for

        Returns:
            Strategy type as string: "meta", "portfolio", "signal" or "unknown"
        """
        if strategy_class is None:
            return "unknown"

        # For string strategy names
        if isinstance(strategy_class, str):
            strategy_name = strategy_class.lower()
            if "meta" in strategy_name:
                return "meta"
            if "portfolio" in strategy_name:
                return "portfolio"
            if "signal" in strategy_name:
                return "signal"
            return "unknown"

        # For actual class objects
        class_name = getattr(strategy_class, "__name__", "")

        if "MetaStrategy" in class_name or "meta" in class_name.lower():
            return "meta"
        elif "PortfolioStrategy" in class_name or "portfolio" in class_name.lower():
            return "portfolio"
        elif "SignalStrategy" in class_name or "signal" in class_name.lower():
            return "signal"
        elif ("Tester" in class_name) or ("Diagnostic" in class_name):
            return "unknown"
        else:
            return "unknown"


# Factory function for easy instantiation


def create_strategy_resolver(
    strategy_lookup: Optional[IStrategyLookup] = None,
) -> PolymorphicStrategyResolver:
    """Create a new polymorphic strategy resolver instance."""
    return PolymorphicStrategyResolver(strategy_lookup)


# Factory for creating strategy resolvers
class StrategyResolverFactory:
    """Factory for creating strategy resolvers."""

    @staticmethod
    def create() -> PolymorphicStrategyResolver:
        """Create a new strategy resolver instance."""
        return create_strategy_resolver()
