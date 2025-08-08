"""
Strategy registry interfaces following SOLID principles.

This module provides clean abstractions for strategy registry operations,
following the same architectural patterns as other framework components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type, Any


class IStrategyRegistry(ABC):
    """
    Interface for strategy registry operations.

    Defines the contract for strategy lookup and enumeration operations.
    Follows ISP by focusing only on registry access concerns.
    """

    @abstractmethod
    def get_strategy_class(self, name: str) -> Optional[Type[Any]]:
        """
        Get strategy class by name.

        Args:
            name: Strategy class name

        Returns:
            Strategy class if found, None otherwise
        """
        pass

    @abstractmethod
    def get_all_strategies(self) -> Dict[str, Type[Any]]:
        """
        Get all discovered strategies.

        Returns:
            Dictionary mapping strategy names to classes
        """
        pass

    @abstractmethod
    def is_strategy_registered(self, name: str) -> bool:
        """
        Check if a strategy is registered.

        Args:
            name: Strategy class name

        Returns:
            True if strategy is registered, False otherwise
        """
        pass

    @abstractmethod
    def get_strategy_count(self) -> int:
        """
        Get total number of registered strategies.

        Returns:
            Number of registered strategies
        """
        pass


class IStrategyDiscoveryEngine(ABC):
    """
    Interface for strategy discovery mechanisms.

    Defines the contract for discovering and loading strategies.
    Follows SRP by focusing only on discovery concerns.
    """

    @abstractmethod
    def discover_strategies(self) -> Dict[str, Type[Any]]:
        """
        Discover and load all available strategies.

        Returns:
            Dictionary mapping strategy names to classes

        Raises:
            StrategyDiscoveryError: If discovery fails
        """
        pass

    @abstractmethod
    def get_discovery_paths(self) -> List[str]:
        """
        Get the paths where strategies are discovered.

        Returns:
            List of directory paths used for discovery
        """
        pass


class IStrategyValidator(ABC):
    """
    Interface for strategy validation operations.

    Defines the contract for validating discovered strategies.
    Follows SRP by focusing only on validation concerns.
    """

    @abstractmethod
    def is_valid_strategy(self, strategy_class: Type[Any]) -> bool:
        """
        Check if a class is a valid strategy.

        Args:
            strategy_class: Class to validate

        Returns:
            True if valid strategy, False otherwise
        """
        pass

    @abstractmethod
    def get_validation_errors(self, strategy_class: Type[Any]) -> List[str]:
        """
        Get detailed validation errors for a strategy class.

        Args:
            strategy_class: Class to validate

        Returns:
            List of validation error messages
        """
        pass

    @abstractmethod
    def get_base_strategy_types(self) -> List[Type[Any]]:
        """
        Get the base strategy types that should be excluded.

        Returns:
            List of base strategy classes
        """
        pass


class StrategyDiscoveryError(Exception):
    """Raised when strategy discovery fails."""

    def __init__(self, message: str, discovery_path: Optional[str] = None):
        self.discovery_path = discovery_path
        super().__init__(message)
