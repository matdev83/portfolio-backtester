"""
Abstract trial deduplication interfaces and base implementations for optimization
to avoid repeating identical parameter combinations across different optimizers.
"""

import logging
import hashlib
import json
from typing import Dict, Any, Set, Optional
from abc import ABC

from .interfaces import AbstractTrialDeduplicator

logger = logging.getLogger(__name__)


class BaseTrialDeduplicator(AbstractTrialDeduplicator, ABC):
    """
    Abstract base class for trial deduplication implementations.

    This class defines the interface for trial deduplication that can be
    implemented by different optimizers.
    """

    def __init__(self, enable_deduplication: bool = True):
        self.enable_deduplication = enable_deduplication
        self.seen_parameter_hashes: Set[str] = set()
        self.parameter_to_value: Dict[str, float] = {}

    def _hash_parameters(self, parameters: Dict[str, Any]) -> str:
        """
        Create a deterministic hash of parameter values.

        Args:
            parameters: Dictionary of parameter names to values

        Returns:
            String hash of the parameters
        """
        # Sort parameters by key for deterministic hashing
        sorted_params = dict(sorted(parameters.items()))

        # Convert to JSON string and hash
        param_str = json.dumps(sorted_params, sort_keys=True, default=str)
        return hashlib.md5(param_str.encode(), usedforsecurity=False).hexdigest()

    def is_duplicate(self, params: Dict[str, Any]) -> bool:
        """
        Check if these parameters have been seen before.

        Args:
            params: Dictionary of parameter names to values

        Returns:
            True if this is a duplicate, False otherwise
        """
        if not self.enable_deduplication:
            return False

        param_hash = self._hash_parameters(params)
        return param_hash in self.seen_parameter_hashes

    def add_trial(self, params: Dict[str, Any], result: Optional[float] = None) -> None:
        """
        Add a trial to the deduplication cache.

        Args:
            params: Dictionary of parameter names to values
            result: Optional objective value for this parameter combination
        """
        if not self.enable_deduplication:
            return

        param_hash = self._hash_parameters(params)
        self.seen_parameter_hashes.add(param_hash)

        if result is not None:
            self.parameter_to_value[param_hash] = result

    def get_cached_value(self, parameters: Dict[str, Any]) -> Optional[float]:
        """
        Get cached objective value for parameters if available.

        Args:
            parameters: Dictionary of parameter names to values

        Returns:
            Cached objective value or None if not available
        """
        if not self.enable_deduplication:
            return None

        param_hash = self._hash_parameters(parameters)
        return self.parameter_to_value.get(param_hash)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get deduplication statistics.

        Returns:
            Dictionary with deduplication stats
        """
        return {
            "enabled": self.enable_deduplication,
            "unique_parameter_combinations": len(self.seen_parameter_hashes),
            "cached_values": len(self.parameter_to_value),
        }


class GenericTrialDeduplicator(BaseTrialDeduplicator):
    """
    Generic trial deduplicator implementation that can be used with any optimizer.
    """

    def __init__(self, enable_deduplication: bool = True):
        super().__init__(enable_deduplication)
        self.duplicate_count = 0
        self.cache_hits = 0

    def is_duplicate(self, params: Dict[str, Any]) -> bool:
        """
        Check if these parameters have been seen before.

        Args:
            params: Dictionary of parameter names to values

        Returns:
            True if this is a duplicate, False otherwise
        """
        result = super().is_duplicate(params)
        if result:
            self.duplicate_count += 1
        return result

    def get_stats(self) -> Dict[str, Any]:
        """
        Get deduplication statistics.

        Returns:
            Dictionary with deduplication stats
        """
        base_stats = super().get_stats()
        base_stats.update(
            {
                "duplicate_trials_detected": self.duplicate_count,
                "cache_hits": self.cache_hits,
            }
        )
        return base_stats


def create_parameter_hash(parameters: Dict[str, Any]) -> str:
    """
    Create a deterministic hash of parameter values for use across optimizers.

    Args:
        parameters: Dictionary of parameter names to values

    Returns:
        String hash of the parameters
    """
    # Sort parameters by key for deterministic hashing
    sorted_params = dict(sorted(parameters.items()))

    # Convert to JSON string and hash
    param_str = json.dumps(sorted_params, sort_keys=True, default=str)
    return hashlib.md5(param_str.encode(), usedforsecurity=False).hexdigest()
