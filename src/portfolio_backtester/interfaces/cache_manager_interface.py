"""
Interface for cache management dependencies implementing Dependency Inversion Principle.

This module provides abstractions for cache management functionality,
enabling dependency inversion for backtester components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd


class ICacheManager(ABC):
    """
    Abstract interface for cache management.

    This interface defines the contract that all cache manager implementations
    must follow, enabling dependency inversion for backtester components.
    """

    @abstractmethod
    def get_cached_returns(self, data: pd.DataFrame, identifier: str = "default") -> pd.DataFrame:
        """
        Get cached returns if available.

        Args:
            data: DataFrame to get cached returns for
            identifier: Optional identifier for cache uniqueness

        Returns:
            Cached returns DataFrame if available, computed and cached otherwise
        """
        pass

    @abstractmethod
    def get_window_returns_by_dates(
        self,
        daily_data: pd.DataFrame,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
    ) -> Optional[pd.DataFrame]:
        """
        Get cached window returns by date range.

        Args:
            daily_data: Full daily price data
            window_start: Start of the window
            window_end: End of the window

        Returns:
            DataFrame of returns for the window, or None if not cached
        """
        pass

    @abstractmethod
    def precompute_window_returns(
        self, daily_data: pd.DataFrame, windows: list
    ) -> Dict[str, pd.DataFrame]:
        """
        Precompute returns for multiple windows and cache them.

        Args:
            daily_data: Full daily price data
            windows: List of window configurations

        Returns:
            Dictionary mapping window identifiers to returns DataFrames
        """
        pass

    @abstractmethod
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary containing cache statistics
        """
        pass

    @abstractmethod
    def clear_cache(self) -> None:
        """Clear all cached data."""
        pass


class ICacheManagerFactory(ABC):
    """
    Abstract factory interface for creating cache manager instances.
    """

    @abstractmethod
    def create_cache_manager(self) -> ICacheManager:
        """
        Create a cache manager instance.

        Returns:
            Cache manager instance implementing ICacheManager
        """
        pass


class ConcreteCacheManager(ICacheManager):
    """
    Concrete implementation of cache manager.

    This implementation directly manages a DataPreprocessingCache instance
    following the DIP pattern without depending on legacy global cache functions.
    """

    _global_cache = None

    def __init__(self):
        """Initialize the concrete cache manager."""
        # Import here to avoid circular dependencies
        from ..data_cache import DataPreprocessingCache

        # Manage our own global cache instance following singleton pattern
        if ConcreteCacheManager._global_cache is None:
            ConcreteCacheManager._global_cache = DataPreprocessingCache()
        self._cache = ConcreteCacheManager._global_cache

    def get_cached_returns(self, data: pd.DataFrame, identifier: str = "default") -> pd.DataFrame:
        """
        Get cached returns if available.

        Args:
            data: DataFrame to get cached returns for
            identifier: Optional identifier for cache uniqueness

        Returns:
            Cached returns DataFrame if available, computed and cached otherwise
        """
        return self._cache.get_cached_returns(data, identifier)

    def get_window_returns_by_dates(
        self,
        daily_data: pd.DataFrame,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
    ) -> Optional[pd.DataFrame]:
        """
        Get cached window returns by date range.

        Args:
            daily_data: Full daily price data
            window_start: Start of the window
            window_end: End of the window

        Returns:
            DataFrame of returns for the window, or None if not cached
        """
        return self._cache.get_window_returns_by_dates(daily_data, window_start, window_end)

    def precompute_window_returns(
        self, daily_data: pd.DataFrame, windows: list
    ) -> Dict[str, pd.DataFrame]:
        """
        Precompute returns for multiple windows and cache them.

        Args:
            daily_data: Full daily price data
            windows: List of window configurations

        Returns:
            Dictionary mapping window identifiers to returns DataFrames
        """
        return self._cache.precompute_window_returns(daily_data, windows)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary containing cache statistics
        """
        return self._cache.get_cache_stats()

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear_cache()


class ConcreteCacheManagerFactory(ICacheManagerFactory):
    """
    Concrete implementation of cache manager factory.

    This factory creates cache manager instances without exposing
    concrete implementation details.
    """

    def create_cache_manager(self) -> ICacheManager:
        """
        Create a cache manager instance.

        Returns:
            Cache manager instance implementing ICacheManager
        """
        return ConcreteCacheManager()


# Factory instance for dependency injection
def create_cache_manager_factory() -> ICacheManagerFactory:
    """
    Create a cache manager factory instance.

    Returns:
        Cache manager factory implementing ICacheManagerFactory
    """
    return ConcreteCacheManagerFactory()


def create_cache_manager() -> ICacheManager:
    """
    Create a cache manager instance using the factory.

    Returns:
        Cache manager instance implementing ICacheManager
    """
    factory = create_cache_manager_factory()
    return factory.create_cache_manager()
