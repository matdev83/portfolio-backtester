"""
Factory for creating trial deduplicators for different optimizers.
"""

from typing import Dict, Any, Optional
from .deduplication import GenericTrialDeduplicator
from .interfaces import AbstractTrialDeduplicator


class DeduplicationFactory:
    """
    Factory for creating trial deduplicators for different optimizers.
    """

    @staticmethod
    def create_deduplicator(
        optimizer_type: str, config: Optional[Dict[str, Any]] = None
    ) -> AbstractTrialDeduplicator:
        """
        Create a trial deduplicator for a specific optimizer type.

        Args:
            optimizer_type: Type of optimizer ('optuna', 'genetic', etc.)
            config: Configuration dictionary for the deduplicator

        Returns:
            Trial deduplicator instance
        """
        if config is None:
            config = {}

        enable_deduplication = config.get("enable_deduplication", True)
        use_persistent_cache = config.get("use_persistent_cache", False)
        cache_dir = config.get("cache_dir", None)
        cache_file = config.get("cache_file", None)

        if use_persistent_cache:
            try:
                from .persistent_deduplication import PersistentTrialDeduplicator

                return PersistentTrialDeduplicator(
                    enable_deduplication=enable_deduplication,
                    cache_dir=cache_dir,
                    cache_file=cache_file,
                    optimizer_type=optimizer_type,
                )
            except ImportError:
                # Fall back to generic deduplicator if persistent implementation is not available
                return GenericTrialDeduplicator(enable_deduplication)
        else:
            # Use standard in-memory deduplicator
            return GenericTrialDeduplicator(enable_deduplication)
