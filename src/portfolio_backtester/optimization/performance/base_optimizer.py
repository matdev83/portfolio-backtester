"""Base performance optimizer with common functionality."""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from .interfaces import AbstractPerformanceOptimizer

logger = logging.getLogger(__name__)


class BasePerformanceOptimizer(AbstractPerformanceOptimizer, ABC):
    """Base class for performance optimizers with common functionality."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the base performance optimizer.
        
        Args:
            config: Configuration dictionary for the optimizer
        """
        self.config = config or {}
        self.metrics = {}
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration parameters are invalid
        """
        # Validate common configuration parameters
        if 'enable_vectorized_tracking' in self.config:
            if not isinstance(self.config['enable_vectorized_tracking'], bool):
                raise ValueError("enable_vectorized_tracking must be a boolean")
        
        if 'enable_parallel_execution' in self.config:
            if not isinstance(self.config['enable_parallel_execution'], bool):
                raise ValueError("enable_parallel_execution must be a boolean")
        
        if 'n_jobs' in self.config:
            if not isinstance(self.config['n_jobs'], int) or self.config['n_jobs'] < 1:
                raise ValueError("n_jobs must be a positive integer")
    
    def optimize_trade_tracking(self, weights: Any, prices: Any, costs: Any) -> Dict[str, Any]:
        """Optimize trade tracking performance with timing.
        
        Args:
            weights: Portfolio weights
            prices: Asset prices
            costs: Transaction costs
            
        Returns:
            Dictionary of optimized trade statistics
        """
        start_time = time.time()
        result = self._optimize_trade_tracking_impl(weights, prices, costs)
        execution_time = time.time() - start_time
        
        # Record performance metrics
        self.metrics['trade_tracking_time'] = execution_time
        logger.info(f"Trade tracking optimization completed in {execution_time:.4f} seconds")
        
        return result
    
    @abstractmethod
    def _optimize_trade_tracking_impl(self, weights: Any, prices: Any, costs: Any) -> Dict[str, Any]:
        """Implementation of trade tracking optimization.
        
        Args:
            weights: Portfolio weights
            prices: Asset prices
            costs: Transaction costs
            
        Returns:
            Dictionary of optimized trade statistics
        """
        pass
    
    def deduplicate_parameters(self, params: Dict[str, Any]) -> bool:
        """Check if parameters are duplicates with timing.
        
        Args:
            params: Parameter dictionary to check
            
        Returns:
            True if parameters are duplicates, False otherwise
        """
        start_time = time.time()
        result = self._deduplicate_parameters_impl(params)
        execution_time = time.time() - start_time
        
        # Record performance metrics
        self.metrics['deduplication_time'] = execution_time
        logger.info(f"Parameter deduplication completed in {execution_time:.4f} seconds")
        
        return result
    
    @abstractmethod
    def _deduplicate_parameters_impl(self, params: Dict[str, Any]) -> bool:
        """Implementation of parameter deduplication.
        
        Args:
            params: Parameter dictionary to check
            
        Returns:
            True if parameters are duplicates, False otherwise
        """
        pass
    
    def run_parallel_optimization(self, config: Dict[str, Any]) -> Any:
        """Run optimization in parallel with timing.
        
        Args:
            config: Optimization configuration
            
        Returns:
            Optimization result
        """
        start_time = time.time()
        result = self._run_parallel_optimization_impl(config)
        execution_time = time.time() - start_time
        
        # Record performance metrics
        self.metrics['parallel_optimization_time'] = execution_time
        logger.info(f"Parallel optimization completed in {execution_time:.4f} seconds")
        
        return result
    
    @abstractmethod
    def _run_parallel_optimization_impl(self, config: Dict[str, Any]) -> Any:
        """Implementation of parallel optimization.
        
        Args:
            config: Optimization configuration
            
        Returns:
            Optimization result
        """
        pass
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get collected performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        return self.metrics.copy()
    
    def reset_performance_metrics(self) -> None:
        """Reset collected performance metrics."""
        self.metrics = {}