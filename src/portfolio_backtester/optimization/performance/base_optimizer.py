"""Base performance optimizer with common functionality."""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from .interfaces import AbstractPerformanceOptimizer

logger = logging.getLogger(__name__)


class BasePerformanceOptimizer(AbstractPerformanceOptimizer, ABC):
    """Base class for performance optimizers with common functionality."""

    def __init__(
        self,
        *,
        enable_vectorized_tracking: bool = True,
        enable_deduplication: bool = True,
        enable_parallel_execution: bool = False,
        n_jobs: int = 1,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the base performance optimizer.

        Args:
            enable_vectorized_tracking: Enable fast vectorized trade tracking where supported.
            enable_deduplication: Enable parameter deduplication subsystem.
            enable_parallel_execution: Enable parallel execution paths if available.
            n_jobs: Number of worker jobs/processes to use when parallel execution is enabled.
            config: Optional configuration dictionary for additional optimizer settings.
        """
        self.config: Dict[str, Any] = (config or {}).copy()
        # Expose standardized attributes used by concrete optimizers
        self.enable_vectorized_tracking: bool = bool(enable_vectorized_tracking)
        self.enable_deduplication: bool = bool(enable_deduplication)
        self.enable_parallel_execution: bool = bool(enable_parallel_execution)
        self.n_jobs: int = int(n_jobs)
        self.metrics: Dict[str, Any] = {}
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the configuration parameters.

        Raises:
            ValueError: If configuration parameters are invalid
        """
        # Validate standardized attributes
        if not isinstance(self.enable_vectorized_tracking, bool):
            raise ValueError("enable_vectorized_tracking must be a boolean")
        if not isinstance(self.enable_deduplication, bool):
            raise ValueError("enable_deduplication must be a boolean")
        if not isinstance(self.enable_parallel_execution, bool):
            raise ValueError("enable_parallel_execution must be a boolean")
        if not isinstance(self.n_jobs, int) or self.n_jobs < 1:
            raise ValueError("n_jobs must be a positive integer")

        # Validate optional extra config fields if provided
        if "enable_vectorized_tracking" in self.config and not isinstance(
            self.config["enable_vectorized_tracking"], bool
        ):
            raise ValueError("config['enable_vectorized_tracking'] must be a boolean")
        if "enable_parallel_execution" in self.config and not isinstance(
            self.config["enable_parallel_execution"], bool
        ):
            raise ValueError("config['enable_parallel_execution'] must be a boolean")
        if "n_jobs" in self.config:
            n_jobs_cfg = self.config["n_jobs"]
            if not isinstance(n_jobs_cfg, int) or n_jobs_cfg < 1:
                raise ValueError("config['n_jobs'] must be a positive integer")

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
        self.metrics["trade_tracking_time"] = execution_time
        logger.info(f"Trade tracking optimization completed in {execution_time:.4f} seconds")

        return result

    @abstractmethod
    def _optimize_trade_tracking_impl(
        self, weights: Any, prices: Any, costs: Any
    ) -> Dict[str, Any]:
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
        self.metrics["deduplication_time"] = execution_time
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
        self.metrics["parallel_optimization_time"] = execution_time
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
