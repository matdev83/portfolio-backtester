"""
Cache statistics tracking for performance monitoring.

This module provides functionality for tracking and reporting cache
performance metrics including hit rates and request counts.
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class CacheStatistics:
    """
    Tracks and reports cache performance statistics.

    This class maintains counters for cache hits, misses, and provides
    methods for calculating performance metrics like hit rates.
    """

    def __init__(self) -> None:
        """Initialize cache statistics counters."""
        self._stats = {"hits": 0, "misses": 0, "window_hits": 0, "window_misses": 0}

    def record_hit(self, is_window: bool = False) -> None:
        """
        Record a cache hit.

        Args:
            is_window: Whether this was a window-specific cache hit
        """
        if is_window:
            self._stats["window_hits"] += 1
        else:
            self._stats["hits"] += 1

    def record_miss(self, is_window: bool = False) -> None:
        """
        Record a cache miss.

        Args:
            is_window: Whether this was a window-specific cache miss
        """
        if is_window:
            self._stats["window_misses"] += 1
        else:
            self._stats["misses"] += 1

    def get_statistics(self, cache_size_info: Dict[str, int]) -> Dict[str, Any]:
        """
        Get comprehensive cache performance statistics.

        Args:
            cache_size_info: Dictionary with cache size information

        Returns:
            Dictionary containing all cache statistics and metrics
        """
        total_requests = self._stats["hits"] + self._stats["misses"]
        window_requests = self._stats["window_hits"] + self._stats["window_misses"]

        return {
            # Size information
            **cache_size_info,
            # Hit rates
            "regular_hit_rate": self._stats["hits"] / max(1, total_requests),
            "window_hit_rate": self._stats["window_hits"] / max(1, window_requests),
            # Request counts
            "total_requests": total_requests,
            "window_requests": window_requests,
            # Raw statistics
            **self._stats,
        }

    def reset_statistics(self) -> None:
        """Reset all statistics counters to zero."""
        self._stats = {"hits": 0, "misses": 0, "window_hits": 0, "window_misses": 0}

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Cache statistics reset")

    def log_cache_operation(
        self, operation: str, cache_key: str, hit: bool, is_window: bool = False
    ) -> None:
        """
        Log a cache operation and record the statistics.

        Args:
            operation: Description of the operation (e.g., "returns", "window returns")
            cache_key: The cache key involved
            hit: Whether this was a cache hit or miss
            is_window: Whether this was a window-specific operation
        """
        if hit:
            self.record_hit(is_window)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Cache HIT for {operation}: {cache_key}")
        else:
            self.record_miss(is_window)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Cache MISS for {operation}: {cache_key} - computing...")
