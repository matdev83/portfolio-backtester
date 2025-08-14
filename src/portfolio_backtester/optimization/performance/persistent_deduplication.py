"""
Persistent trial deduplication for optimization across processes and runs.

This module provides a persistent deduplication cache that can be shared
across multiple processes and optimization runs, reducing redundant evaluations.
"""

import os
import pickle
import tempfile
import time
from typing import Dict, Any, Optional
import logging
from filelock import FileLock

from .deduplication import BaseTrialDeduplicator

logger = logging.getLogger(__name__)


class PersistentTrialDeduplicator(BaseTrialDeduplicator):
    """
    Trial deduplicator with persistent storage to share cache across processes and runs.

    This implementation saves and loads deduplication data to/from disk,
    allowing multiple processes to share the same cache and preserving
    results between optimization runs.
    """

    def __init__(
        self,
        enable_deduplication: bool = True,
        cache_dir: Optional[str] = None,
        cache_file: Optional[str] = None,
        optimizer_type: str = "genetic",
    ):
        """Initialize the persistent trial deduplicator.

        Args:
            enable_deduplication: Whether deduplication is enabled
            cache_dir: Directory to store the cache file (default: temp directory)
            cache_file: Name of the cache file (default: based on optimizer_type)
            optimizer_type: Type of optimizer ('optuna', 'genetic', etc.)
        """
        super().__init__(enable_deduplication)

        self.optimizer_type = optimizer_type
        self.duplicate_count = 0
        self.cache_hits = 0

        # Set up cache file path
        if cache_dir is None:
            self.cache_dir = os.path.join(tempfile.gettempdir(), "portfolio_backtester_dedup_cache")
        else:
            self.cache_dir = cache_dir

        if cache_file is None:
            self.cache_file = f"{optimizer_type}_dedup_cache.pkl"
        else:
            self.cache_file = cache_file

        self.cache_path = os.path.join(self.cache_dir, self.cache_file)
        self.lock_path = f"{self.cache_path}.lock"

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load existing cache if available
        self._load_cache()

    def _load_cache(self) -> None:
        """Load the deduplication cache from disk."""
        if not self.enable_deduplication:
            return

        try:
            with FileLock(self.lock_path):
                if os.path.exists(self.cache_path):
                    with open(self.cache_path, "rb") as f:
                        cache_data = pickle.load(f)

                    self.seen_parameter_hashes = cache_data.get("seen_parameter_hashes", set())
                    self.parameter_to_value = cache_data.get("parameter_to_value", {})

                    logger.debug(
                        f"Loaded deduplication cache with {len(self.seen_parameter_hashes)} "
                        f"unique parameter combinations and {len(self.parameter_to_value)} cached values"
                    )
                else:
                    logger.debug("No existing deduplication cache found, starting fresh")
        except Exception as e:
            logger.warning(f"Failed to load deduplication cache: {e}")
            # Initialize empty cache on failure
            self.seen_parameter_hashes = set()
            self.parameter_to_value = {}

    def _save_cache(self) -> None:
        """Save the deduplication cache to disk."""
        if not self.enable_deduplication:
            return

        try:
            with FileLock(self.lock_path):
                # Prepare cache data
                cache_data = {
                    "seen_parameter_hashes": self.seen_parameter_hashes,
                    "parameter_to_value": self.parameter_to_value,
                    "last_updated": time.time(),
                    "optimizer_type": self.optimizer_type,
                }

                # Save to disk
                with open(self.cache_path, "wb") as f:
                    pickle.dump(cache_data, f)

                logger.debug(
                    f"Saved deduplication cache with {len(self.seen_parameter_hashes)} "
                    f"unique parameter combinations and {len(self.parameter_to_value)} cached values"
                )
        except Exception as e:
            logger.warning(f"Failed to save deduplication cache: {e}")

    def is_duplicate(self, params: Dict[str, Any]) -> bool:
        """Check if parameters have been seen before.

        Args:
            params: Dictionary of parameter names to values

        Returns:
            True if this is a duplicate, False otherwise
        """
        result = super().is_duplicate(params)
        if result:
            self.duplicate_count += 1
        return result

    def add_trial(self, params: Dict[str, Any], result: Optional[float] = None) -> None:
        """Add a trial to the deduplication cache and save to disk.

        Args:
            params: Dictionary of parameter names to values
            result: Optional objective value for this parameter combination
        """
        if not self.enable_deduplication:
            return

        # Add to in-memory cache
        super().add_trial(params, result)

        # Save to disk periodically (every 10 additions)
        if len(self.parameter_to_value) % 10 == 0:
            self._save_cache()

    def get_cached_value(self, parameters: Dict[str, Any]) -> Optional[float]:
        """Get cached objective value for parameters if available.

        Args:
            parameters: Dictionary of parameter names to values

        Returns:
            Cached objective value or None if not available
        """
        result = super().get_cached_value(parameters)
        if result is not None:
            self.cache_hits += 1
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics.

        Returns:
            Dictionary with deduplication stats
        """
        base_stats = super().get_stats()
        base_stats.update(
            {
                "duplicate_trials_detected": self.duplicate_count,
                "cache_hits": self.cache_hits,
                "persistent_cache": True,
                "cache_path": self.cache_path,
            }
        )
        return base_stats

    def clear_cache(self) -> None:
        """Clear the deduplication cache from memory and disk."""
        self.seen_parameter_hashes.clear()
        self.parameter_to_value.clear()
        self.duplicate_count = 0
        self.cache_hits = 0

        try:
            with FileLock(self.lock_path):
                if os.path.exists(self.cache_path):
                    os.remove(self.cache_path)
                    logger.debug(f"Cleared deduplication cache file: {self.cache_path}")
        except Exception as e:
            logger.warning(f"Failed to clear deduplication cache file: {e}")

    def __del__(self) -> None:
        """Save cache when the object is destroyed."""
        self._save_cache()
