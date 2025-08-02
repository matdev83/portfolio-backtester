"""Simple meta strategy with fixed allocation weights."""

from typing import Dict, Any
import logging

from ..base.meta_strategy import BaseMetaStrategy

logger = logging.getLogger(__name__)


class SimpleMetaStrategy(BaseMetaStrategy):
    """
    Simple meta strategy that allocates capital using fixed percentage weights.
    
    This is the most basic meta strategy implementation that maintains constant
    allocation weights across all sub-strategies as specified in the configuration.
    """
    
    def __init__(self, strategy_params: Dict[str, Any], global_config: Dict[str, Any] = None):
        super().__init__(strategy_params, global_config=global_config)
        
        # Set default parameters
        defaults = {
            "min_allocation": 0.05,
            "rebalance_threshold": 0.05
        }
        
        for key, value in defaults.items():
            self.strategy_params.setdefault(key, value)
    
    def allocate_capital(self) -> Dict[str, float]:
        """
        Return fixed allocation weights as specified in configuration.
        
        For SimpleMetaStrategy, this just returns the configured weights
        without any dynamic adjustment.
        
        Returns:
            Dictionary mapping strategy_id to allocation weight
        """
        allocations = {}
        
        for allocation in self.allocations:
            allocations[allocation.strategy_id] = allocation.weight
        
        return allocations
    

    
    @classmethod
    def tunable_parameters(cls) -> set[str]:
        """Names of hyper-parameters this strategy understands."""
        base_params = super().tunable_parameters()
        return base_params | set()  # SimpleMetaStrategy doesn't add additional parameters