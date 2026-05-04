"""
Factory for creating performance optimizers for different optimizers.
"""

from typing import Dict, Any, Optional
from .deduplication import GenericTrialDeduplicator
from .parallel_execution import GenericParallelRunner
from .interfaces import AbstractPerformanceOptimizer


class PerformanceOptimizerFactory:
    """
    Factory for creating performance optimizers for different optimizers.
    """

    @staticmethod
    def create_performance_optimizer(
        optimizer_type: str, config: Optional[Dict[str, Any]] = None
    ) -> Optional[AbstractPerformanceOptimizer]:
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

        if not config.get("enable_performance_optimizations", True):
            return None

        if optimizer_type == "optuna":
            return OptunaPerformanceOptimizer(config)
        if optimizer_type == "genetic":
            return GeneticPerformanceOptimizer(config)

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
            enable_deduplication=self.config.get("enable_deduplication", True)
        )
        self.parallel_runner = GenericParallelRunner(n_jobs=self.config.get("n_jobs", 1))

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
