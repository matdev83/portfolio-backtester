"""
Factory for creating performance optimizers for different optimizers.
"""

from typing import Dict, Any, Optional
from .deduplication import GenericTrialDeduplicator
from .parallel_execution import GenericParallelRunner
from .vectorized_tracking import VectorizedTradeTracker
from .interfaces import AbstractPerformanceOptimizer


class PerformanceOptimizerFactory:
    """
    Factory for creating performance optimizers for different optimizers.
    """
    
    @staticmethod
    def create_performance_optimizer(optimizer_type: str, config: Optional[Dict[str, Any]] = None) -> Optional[AbstractPerformanceOptimizer]:
        """
        Create a performance optimizer for a specific optimizer type.
        
        Args:
            optimizer_type: Type of optimizer ('optuna', 'genetic', etc.)
            config: Configuration dictionary for the performance optimizer
            
        Returns:
            Performance optimizer instance or None if not supported
        """
        if config is None:
            config = {}
            
        # Check if performance optimizations are enabled
        if not config.get('enable_performance_optimizations', True):
            return None
            
        # Create components based on optimizer type
        if optimizer_type == 'optuna':
            return OptunaPerformanceOptimizer(config)
        elif optimizer_type == 'genetic':
            return GeneticPerformanceOptimizer(config)
        else:
            # Return generic performance optimizer for unknown optimizer types
            return GenericPerformanceOptimizer(config)


class GenericPerformanceOptimizer(AbstractPerformanceOptimizer):
    """
    Generic performance optimizer that can be used with any optimizer.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the generic performance optimizer.
        
        Args:
            config: Configuration dictionary for the performance optimizer
        """
        self.config = config or {}
        self.deduplicator = GenericTrialDeduplicator(
            enable_deduplication=self.config.get('enable_deduplication', True)
        )
        self.parallel_runner = GenericParallelRunner(
            n_jobs=self.config.get('n_jobs', 1)
        )
        self.trade_tracker = VectorizedTradeTracker(
            portfolio_value=self.config.get('portfolio_value', 100000.0)
        )
    
    def optimize_trade_tracking(self, weights: Any, prices: Any, costs: Any) -> Dict[str, Any]:
        """
        Optimize trade tracking performance.
        
        Args:
            weights: Portfolio weights
            prices: Asset prices
            costs: Transaction costs
            
        Returns:
            Dictionary of optimized trade statistics
        """
        return self.trade_tracker.track_trades_optimized(weights, prices, costs)
    
    def deduplicate_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Check if parameters are duplicates.
        
        Args:
            params: Parameter dictionary to check
            
        Returns:
            True if parameters are duplicates, False otherwise
        """
        return self.deduplicator.is_duplicate(params)
    
    def run_parallel_optimization(self, config: Dict[str, Any]) -> Any:
        """
        Run optimization in parallel.
        
        Args:
            config: Optimization configuration
            
        Returns:
            Optimization result
        """
        return self.parallel_runner.run(config)
    
    def add_trial(self, params: Dict[str, Any], result: float) -> None:
        """
        Add a trial to the deduplication cache.
        
        Args:
            params: Parameter dictionary
            result: Trial result
        """
        self.deduplicator.add_trial(params, result)


class OptunaPerformanceOptimizer(GenericPerformanceOptimizer):
    """
    Performance optimizer specifically for Optuna optimizer.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Optuna performance optimizer.
        
        Args:
            config: Configuration dictionary for the performance optimizer
        """
        super().__init__(config)


class GeneticPerformanceOptimizer(GenericPerformanceOptimizer):
    """
    Performance optimizer specifically for Genetic optimizer.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Genetic performance optimizer.
        
        Args:
            config: Configuration dictionary for the performance optimizer
        """
        super().__init__(config)