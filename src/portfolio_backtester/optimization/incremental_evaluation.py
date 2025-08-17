"""
Incremental window evaluation for genetic algorithm optimization.

This module provides utilities to incrementally evaluate windows when parameter
changes are minimal, avoiding redundant calculations across generations.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from loguru import logger


class IncrementalEvaluationManager:
    """
    Manages incremental window evaluation to reduce redundant calculations.

    This class tracks parameter changes between generations and determines which
    windows need to be re-evaluated based on parameter dependencies.
    """

    def __init__(self, enable_caching: bool = True, cache_ttl: int = 3):
        """
        Initialize the incremental evaluation manager.

        Args:
            enable_caching: Whether to enable caching
            cache_ttl: How many generations to keep cached results (time to live)
        """
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl

        # Parameter dependency tracking
        self._parameter_dependencies: Dict[str, Set[str]] = {}

        # Window evaluation caching
        self._result_cache: Dict[Tuple[int, str], Any] = {}
        self._cache_generation: Dict[Tuple[int, str], int] = {}
        self._current_generation = 0

        # Stats
        self._cache_hits = 0
        self._cache_misses = 0
        self._windows_evaluated = 0
        self._windows_skipped = 0

        logger.debug(f"IncrementalEvaluationManager initialized with TTL={cache_ttl}")

    def register_parameter_dependencies(self, dependencies: Dict[str, List[str]]) -> None:
        """
        Register parameter dependencies on strategy components.

        Args:
            dependencies: Dictionary mapping parameter names to affected component names
        """
        for param, components in dependencies.items():
            self._parameter_dependencies[param] = set(components)

        logger.debug(f"Registered dependencies for {len(dependencies)} parameters")

    def detect_affected_components(
        self, old_params: Dict[str, Any], new_params: Dict[str, Any]
    ) -> Set[str]:
        """
        Detect which components are affected by parameter changes.

        Args:
            old_params: Previous parameter set
            new_params: New parameter set

        Returns:
            Set of component names that need to be re-evaluated
        """
        if not old_params or not new_params:
            # If either parameter set is empty, all components are affected
            return set(["*"])

        affected_components: Set[str] = set()

        # Check each parameter for changes
        all_params = set(old_params.keys()) | set(new_params.keys())
        for param in all_params:
            old_value = old_params.get(param)
            new_value = new_params.get(param)

            # If parameter changed
            if old_value != new_value:
                if param in self._parameter_dependencies:
                    # Add specific components affected by this parameter
                    affected_components.update(self._parameter_dependencies[param])
                else:
                    # If dependency not registered, assume it affects everything
                    return set(["*"])

        if affected_components:
            logger.debug(f"Detected changes affecting {len(affected_components)} components")
        else:
            logger.debug("No components affected by parameter changes")

        return affected_components

    def should_evaluate_window(
        self,
        window_id: int,
        component_id: str,
        old_params: Dict[str, Any],
        new_params: Dict[str, Any],
    ) -> bool:
        """
        Determine if a specific window needs to be evaluated.

        Args:
            window_id: Identifier for the window
            component_id: Identifier for the component
            old_params: Previous parameter set
            new_params: New parameter set

        Returns:
            True if the window should be evaluated, False if cached result can be used
        """
        if not self.enable_caching:
            return True

        # If no previous params, evaluate all windows
        if not old_params:
            self._windows_evaluated += 1
            self._cache_misses += 1
            return True

        # Detect affected components
        affected_components = self.detect_affected_components(old_params, new_params)

        # If all components are affected, evaluate this window
        if "*" in affected_components:
            self._windows_evaluated += 1
            self._cache_misses += 1
            return True

        # Check if this specific component is affected
        if component_id in affected_components:
            self._windows_evaluated += 1
            self._cache_misses += 1
            return True

        # Check if we have a cached result for this window and component
        cache_key = (window_id, component_id)
        if cache_key in self._result_cache:
            # Check if cache is still valid (not expired)
            if self._current_generation - self._cache_generation[cache_key] <= self.cache_ttl:
                self._windows_skipped += 1
                self._cache_hits += 1
                return False

        # No valid cache, need to evaluate
        self._windows_evaluated += 1
        self._cache_misses += 1
        return True

    def cache_window_result(self, window_id: int, component_id: str, result: Any) -> None:
        """
        Cache the result for a window evaluation.

        Args:
            window_id: Identifier for the window
            component_id: Identifier for the component
            result: Result of the window evaluation
        """
        if not self.enable_caching:
            return

        cache_key = (window_id, component_id)
        self._result_cache[cache_key] = result
        self._cache_generation[cache_key] = self._current_generation

    def get_cached_result(self, window_id: int, component_id: str) -> Optional[Any]:
        """
        Get a cached result for a window if available.

        Args:
            window_id: Identifier for the window
            component_id: Identifier for the component

        Returns:
            Cached result or None if not found
        """
        if not self.enable_caching:
            return None

        cache_key = (window_id, component_id)
        if cache_key in self._result_cache:
            # Check if cache is still valid (not expired)
            if self._current_generation - self._cache_generation[cache_key] <= self.cache_ttl:
                return self._result_cache[cache_key]

        return None

    def new_generation(self) -> None:
        """Mark the start of a new generation."""
        self._current_generation += 1

        # Clean up expired cache entries
        if self._current_generation > self.cache_ttl:
            expired_keys = []
            for key, gen in self._cache_generation.items():
                if self._current_generation - gen > self.cache_ttl:
                    expired_keys.append(key)

            for key in expired_keys:
                del self._result_cache[key]
                del self._cache_generation[key]

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    def reset_cache(self) -> None:
        """Reset the cache state."""
        self._result_cache = {}
        self._cache_generation = {}
        self._current_generation = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._windows_evaluated = 0
        self._windows_skipped = 0

        logger.debug("Incremental evaluation cache reset")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about cache usage."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / max(total_requests, 1)

        return {
            "cache_enabled": self.enable_caching,
            "cache_ttl": self.cache_ttl,
            "current_generation": self._current_generation,
            "cache_size": len(self._result_cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "windows_evaluated": self._windows_evaluated,
            "windows_skipped": self._windows_skipped,
            "evaluation_savings": self._cache_hits / max(total_requests, 1),
        }


class WindowEvaluationCache:
    """
    Cache for window evaluation results to avoid redundant calculations.

    This class is used within evaluator implementations to store and retrieve
    intermediate calculation results.
    """

    def __init__(self, capacity: int = 1000):
        """
        Initialize the window evaluation cache.

        Args:
            capacity: Maximum number of entries to keep in cache
        """
        self.capacity = capacity
        self._cache: Dict[str, Any] = {}
        self._access_count: Dict[str, int] = {}
        self._eviction_count = 0

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get an item from the cache.

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        value = self._cache.get(key, default)

        # Update access count for LRU tracking
        if key in self._cache:
            self._access_count[key] = self._access_count.get(key, 0) + 1

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set an item in the cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        # Check if we need to evict entries
        if len(self._cache) >= self.capacity:
            self._evict()

        self._cache[key] = value
        self._access_count[key] = 1

    def _evict(self) -> None:
        """Evict least recently used items from cache."""
        # Find 10% of least used entries to evict
        num_to_evict = max(1, self.capacity // 10)

        # Sort by access count (ascending)
        sorted_keys = sorted(self._access_count.keys(), key=lambda k: self._access_count[k])

        # Evict least used entries
        for key in sorted_keys[:num_to_evict]:
            del self._cache[key]
            del self._access_count[key]

        self._eviction_count += num_to_evict

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_count.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about cache usage."""
        return {
            "size": len(self._cache),
            "capacity": self.capacity,
            "usage_ratio": len(self._cache) / max(self.capacity, 1),
            "eviction_count": self._eviction_count,
        }
