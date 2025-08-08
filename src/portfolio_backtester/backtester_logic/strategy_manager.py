"""
Strategy management logic extracted from Backtester class.

This module implements the StrategyManager class that handles all strategy-related operations
including strategy creation, management, and validation.
"""

import logging
from typing import Any, Dict, Type

from ..strategies.base.base_strategy import BaseStrategy
from ..strategies.registry import get_strategy_registry
from ..strategies.strategy_factory import StrategyFactory

logger = logging.getLogger(__name__)


class StrategyManager:
    """
    Handles strategy creation and management for backtesting.

    This class encapsulates all strategy-related operations that were previously
    part of the Backtester class, following the Single Responsibility Principle.
    """

    def __init__(self):
        """Initialize StrategyManager with SOLID-compliant strategy registry."""
        self._registry = get_strategy_registry()
        self.logger = logger

        if logger.isEnabledFor(logging.DEBUG):
            strategy_count = len(self._registry.get_all_strategies())
            logger.debug(f"StrategyManager initialized with {strategy_count} strategies")

    @property
    def strategy_map(self) -> Dict[str, Type[BaseStrategy]]:
        """Get strategy mapping for backward compatibility."""
        return self._registry.get_all_strategies()

    def get_strategy(self, strategy_spec: Any, params: Dict[str, Any]) -> BaseStrategy:
        """
        Create a strategy instance from specification and parameters.

        Args:
            strategy_spec: Strategy specification (string name or dict with strategy info)
            params: Strategy parameters dictionary

        Returns:
            BaseStrategy instance

        Raises:
            ValueError: If strategy specification is invalid or strategy not found
            TypeError: If strategy class doesn't return BaseStrategy instance
        """
        # Support both string and dict specifications
        if isinstance(strategy_spec, dict):
            strategy_name = (
                strategy_spec.get("strategy")
                or strategy_spec.get("type")
                or strategy_spec.get("name")
            )
        else:
            strategy_name = strategy_spec

        # Enforce that strategy_name is a string; raise if not provided correctly
        if not isinstance(strategy_name, str) or not strategy_name:
            raise ValueError(f"Invalid strategy specification: {strategy_spec!r}")

        # Use factory to instantiate strategy (single-path)
        return StrategyFactory.create_strategy(str(strategy_name), params)

    def get_available_strategies(self) -> Dict[str, type]:
        """
        Get dictionary of all available strategies.

        Returns:
            Dictionary mapping strategy names to strategy classes
        """
        return self.strategy_map.copy()

    def is_strategy_available(self, strategy_name: str) -> bool:
        """
        Check if a strategy is available.

        Args:
            strategy_name: Name of the strategy to check

        Returns:
            True if strategy is available, False otherwise
        """
        return self._registry.is_strategy_registered(strategy_name)

    def get_strategy_class(self, strategy_name: str) -> type:
        """
        Get strategy class by name.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Strategy class

        Raises:
            ValueError: If strategy is not found
        """
        strategy_class = self._registry.get_strategy_class(strategy_name)
        if strategy_class is None:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        return strategy_class

    def create_strategy_with_global_config(
        self, strategy_spec: Any, params: Dict[str, Any], global_config: Dict[str, Any]
    ) -> BaseStrategy:
        """
        Create a strategy instance that may require global configuration.

        Some strategies (especially meta strategies) may need access to global configuration.
        This method handles both cases gracefully.

        Args:
            strategy_spec: Strategy specification (string name or dict with strategy info)
            params: Strategy parameters dictionary
            global_config: Global configuration dictionary

        Returns:
            BaseStrategy instance

        Raises:
            ValueError: If strategy specification is invalid or strategy not found
            TypeError: If strategy class doesn't return BaseStrategy instance
        """
        # Get strategy name
        if isinstance(strategy_spec, dict):
            strategy_name = (
                strategy_spec.get("strategy")
                or strategy_spec.get("type")
                or strategy_spec.get("name")
            )
        else:
            strategy_name = strategy_spec

        if not isinstance(strategy_name, str) or not strategy_name:
            raise ValueError(f"Invalid strategy specification: {strategy_spec!r}")

        # Use factory with global_config support
        return StrategyFactory.create_strategy(
            str(strategy_name), params, global_config=global_config
        )
