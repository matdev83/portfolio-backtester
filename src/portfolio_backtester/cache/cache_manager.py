"""
Cache management utilities for data preprocessing cache.

This module provides cache storage management, including size limits,
cleanup operations, and cache item organization.
"""

import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages cache storage, size limits, and cleanup operations.

    This class handles the storage and retrieval of cached data while
    maintaining configurable size limits and performing cleanup when needed.
    """

    def __init__(self, max_cache_size: int = 100) -> None:
        """
        Initialize the cache manager.

        Args:
            max_cache_size: Maximum number of cached items before cleanup
        """
        self.max_cache_size = max_cache_size
        self.cached_returns: Dict[str, pd.DataFrame] = {}
        self.cached_window_returns: Dict[str, pd.DataFrame] = {}
        self.data_hashes: Dict[str, str] = {}

    def get_cached_item(self, cache_key: str, is_window: bool = False) -> Optional[pd.DataFrame]:
        """
        Get a cached item by key.

        Args:
            cache_key: The cache key to look up
            is_window: Whether this is a window-specific cache lookup

        Returns:
            Cached DataFrame if found, None otherwise
        """
        if is_window:
            return self.cached_window_returns.get(cache_key)
        else:
            return self.cached_returns.get(cache_key)

    def store_cached_item(
        self,
        cache_key: str,
        data: pd.DataFrame,
        is_window: bool = False,
        identifier: Optional[str] = None,
    ):
        """
        Store an item in the cache.

        Args:
            cache_key: The cache key to store under
            data: The DataFrame to cache
            is_window: Whether this is a window-specific cache item
            identifier: Optional identifier for data_hashes tracking
        """
        if is_window:
            self.cached_window_returns[cache_key] = data
        else:
            self.cached_returns[cache_key] = data

        # Store data hash if identifier provided
        if identifier:
            self.data_hashes[identifier] = cache_key

        self.cleanup_cache_if_needed()

    def cleanup_cache_if_needed(self):
        """Remove oldest entries if cache exceeds max size."""
        total_items = len(self.cached_returns) + len(self.cached_window_returns)
        if total_items > self.max_cache_size:
            # Remove oldest entries (simplified - remove half)
            items_to_remove = total_items - self.max_cache_size // 2

            # Remove from regular cache first
            keys_to_remove = list(self.cached_returns.keys())[: items_to_remove // 2]
            for key in keys_to_remove:
                del self.cached_returns[key]

            # Remove from window cache
            remaining_to_remove = items_to_remove - len(keys_to_remove)
            if remaining_to_remove > 0:
                window_keys_to_remove = list(self.cached_window_returns.keys())[
                    :remaining_to_remove
                ]
                for key in window_keys_to_remove:
                    del self.cached_window_returns[key]

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Cache cleanup: removed {len(keys_to_remove)} regular items and {len(window_keys_to_remove) if remaining_to_remove > 0 else 0} window items"
                )

    def clear_cache(self):
        """Clear all cached data."""
        self.cached_returns.clear()
        self.cached_window_returns.clear()
        self.data_hashes.clear()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("All cache data cleared")

    def get_cache_size_info(self) -> Dict[str, int]:
        """
        Get information about cache sizes.

        Returns:
            Dictionary with cache size information
        """
        return {
            "total_cached_items": len(self.cached_returns) + len(self.cached_window_returns),
            "regular_cache_items": len(self.cached_returns),
            "window_cache_items": len(self.cached_window_returns),
            "max_cache_size": self.max_cache_size,
        }
