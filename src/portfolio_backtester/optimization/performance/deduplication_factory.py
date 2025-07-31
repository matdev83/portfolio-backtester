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
    def create_deduplicator(optimizer_type: str, config: Optional[Dict[str, Any]] = None) -> AbstractTrialDeduplicator:
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
            
        enable_deduplication = config.get('enable_deduplication', True)
        
        if optimizer_type == 'optuna':
            return GenericTrialDeduplicator(enable_deduplication)
        elif optimizer_type == 'genetic':
            return GenericTrialDeduplicator(enable_deduplication)
        else:
            # Default to generic deduplicator for unknown optimizer types
            return GenericTrialDeduplicator(enable_deduplication)