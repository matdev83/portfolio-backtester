"""
Parallel Benefit Estimator Interface

This module provides abstract interfaces for parallel processing benefit estimation,
implementing the Dependency Inversion Principle for benefit calculation dependencies.

Key interfaces:
- IParallelBenefitEstimator: Core interface for benefit estimation operations
- IParallelBenefitEstimatorFactory: Factory interface for creating benefit estimators

This eliminates direct dependencies on concrete benefit estimation classes.
"""

from abc import ABC, abstractmethod
from typing import Dict

from ..optimization.performance.parallel_benefit_estimator import (
    ParallelBenefitEstimator,
)


class IParallelBenefitEstimator(ABC):
    """
    Abstract interface for parallel processing benefit estimation.

    This interface defines the contract for benefit estimation implementations,
    enabling dependency inversion for processing components.
    """

    @abstractmethod
    def estimate_parallel_benefit(
        self, num_windows: int, avg_window_time: float
    ) -> Dict[str, float]:
        """
        Estimate the potential benefit of parallel processing.

        Args:
            num_windows: Number of WFO windows
            avg_window_time: Average time per window in seconds

        Returns:
            Dictionary with comprehensive timing estimates
        """
        pass

    @abstractmethod
    def is_parallel_beneficial(
        self,
        num_windows: int,
        avg_window_time: float,
        min_speedup: float = 1.5,
        min_time_saved: float = 5.0,
    ) -> bool:
        """
        Determine if parallel processing is beneficial based on estimates.

        Args:
            num_windows: Number of windows to process
            avg_window_time: Average time per window in seconds
            min_speedup: Minimum speedup required to be considered beneficial
            min_time_saved: Minimum time savings in seconds to be considered beneficial

        Returns:
            True if parallel processing is estimated to be beneficial
        """
        pass

    @abstractmethod
    def generate_performance_report(self, num_windows: int, avg_window_time: float) -> str:
        """
        Generate a human-readable performance report.

        Args:
            num_windows: Number of windows to process
            avg_window_time: Average time per window in seconds

        Returns:
            Formatted performance report string
        """
        pass

    @property
    @abstractmethod
    def max_workers(self) -> int:
        """
        Get the maximum number of workers configured for this estimator.

        Returns:
            Number of maximum workers
        """
        pass

    @property
    @abstractmethod
    def parallel_overhead(self) -> float:
        """
        Get the parallel overhead factor configured for this estimator.

        Returns:
            Parallel overhead factor
        """
        pass


class IParallelBenefitEstimatorFactory(ABC):
    """
    Abstract factory interface for creating parallel benefit estimator instances.

    This factory enables creation of appropriate benefit estimator implementations
    based on configuration and system capabilities.
    """

    @abstractmethod
    def create_estimator(
        self,
        max_workers: int,
        parallel_overhead: float = 0.1,
        math_operations=None,
    ) -> IParallelBenefitEstimator:
        """
        Create a parallel benefit estimator instance.

        Args:
            max_workers: Maximum number of worker processes available
            parallel_overhead: Estimated overhead for parallel processing
            math_operations: Optional math operations interface

        Returns:
            Configured parallel benefit estimator instance
        """
        pass


# Concrete implementations


class ParallelBenefitEstimatorAdapter(IParallelBenefitEstimator):
    """
    Adapter that wraps the concrete ParallelBenefitEstimator to implement the interface.

    This adapter maintains full backward compatibility while enabling dependency inversion.
    """

    def __init__(self, benefit_estimator: ParallelBenefitEstimator):
        """
        Initialize the adapter with a concrete benefit estimator.

        Args:
            benefit_estimator: Concrete benefit estimator instance
        """
        self._estimator = benefit_estimator

    def estimate_parallel_benefit(
        self, num_windows: int, avg_window_time: float
    ) -> Dict[str, float]:
        """Estimate the potential benefit of parallel processing."""
        return self._estimator.estimate_parallel_benefit(num_windows, avg_window_time)

    def is_parallel_beneficial(
        self,
        num_windows: int,
        avg_window_time: float,
        min_speedup: float = 1.5,
        min_time_saved: float = 5.0,
    ) -> bool:
        """Determine if parallel processing is beneficial."""
        return self._estimator.is_parallel_beneficial(
            num_windows, avg_window_time, min_speedup, min_time_saved
        )

    def generate_performance_report(self, num_windows: int, avg_window_time: float) -> str:
        """Generate a human-readable performance report."""
        return self._estimator.generate_performance_report(num_windows, avg_window_time)

    @property
    def max_workers(self) -> int:
        """Get the maximum number of workers."""
        return self._estimator.max_workers

    @property
    def parallel_overhead(self) -> float:
        """Get the parallel overhead factor."""
        return self._estimator.parallel_overhead


class ParallelBenefitEstimatorFactory(IParallelBenefitEstimatorFactory):
    """
    Concrete factory for creating parallel benefit estimator instances.

    This factory creates ParallelBenefitEstimatorAdapter instances that wrap
    the concrete ParallelBenefitEstimator implementation.
    """

    def create_estimator(
        self,
        max_workers: int,
        parallel_overhead: float = 0.1,
        math_operations=None,
    ) -> IParallelBenefitEstimator:
        """Create a parallel benefit estimator instance."""
        concrete_estimator = ParallelBenefitEstimator(
            max_workers=max_workers,
            parallel_overhead=parallel_overhead,
            math_operations=math_operations,
        )
        return ParallelBenefitEstimatorAdapter(concrete_estimator)


# Factory function for easy creation
def create_parallel_benefit_estimator_factory() -> IParallelBenefitEstimatorFactory:
    """
    Create a parallel benefit estimator factory instance.

    Returns:
        Configured parallel benefit estimator factory
    """
    return ParallelBenefitEstimatorFactory()


def create_parallel_benefit_estimator(
    max_workers: int,
    parallel_overhead: float = 0.1,
    math_operations=None,
) -> IParallelBenefitEstimator:
    """
    Create a parallel benefit estimator instance using the default factory.

    Args:
        max_workers: Maximum number of worker processes available
        parallel_overhead: Estimated overhead for parallel processing
        math_operations: Optional math operations interface

    Returns:
        Configured parallel benefit estimator instance
    """
    factory = create_parallel_benefit_estimator_factory()
    return factory.create_estimator(max_workers, parallel_overhead, math_operations)
