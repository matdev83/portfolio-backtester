"""
Strategy resolver interface for resolving strategy specifications.

This module provides interfaces for resolving strategy specifications
without using isinstance checks.
"""

from .strategy_resolver_interface import PolymorphicStrategyResolver, create_strategy_resolver


class StrategyResolverFactory:
    """Factory for creating strategy resolvers."""

    @staticmethod
    def create() -> PolymorphicStrategyResolver:
        """Create a new strategy resolver instance."""
        return create_strategy_resolver()
