"""
Parallel Executor Interface

This module provides abstract interfaces for parallel execution functionality,
implementing the Dependency Inversion Principle for processing dependencies.

Key interfaces:
- IParallelExecutor: Core interface for parallel processing operations
- IParallelExecutorFactory: Factory interface for creating parallel executors

This eliminates direct dependencies on concrete parallel execution classes.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional, Callable
import pandas as pd

from ..optimization.performance.parallel_executor import ParallelExecutor


class IParallelExecutor(ABC):
    """
    Abstract interface for parallel execution operations.

    This interface defines the contract for all parallel executor implementations,
    enabling dependency inversion for processing components.
    """

    @abstractmethod
    def execute_parallel(
        self,
        windows: List[Dict[str, Any]],
        worker_func: Callable,
        fallback_func: Optional[Callable] = None,
    ) -> List[Tuple[Any, pd.Series]]:
        """
        Execute windows in parallel using the configured executor.

        Args:
            windows: List of window configurations to process
            worker_func: Prepared worker function (partial) to execute
            fallback_func: Optional fallback function if parallel processing fails

        Returns:
            List of (objective_value, returns) tuples for each window
        """
        pass

    @abstractmethod
    def should_use_parallel(self, num_windows: int, min_windows_threshold: int = 2) -> bool:
        """
        Determine if parallel processing should be used based on the number of windows.

        Args:
            num_windows: Number of windows to process
            min_windows_threshold: Minimum windows required for parallel processing

        Returns:
            True if parallel processing should be used
        """
        pass

    @property
    @abstractmethod
    def max_workers(self) -> int:
        """
        Get the maximum number of workers configured for this executor.

        Returns:
            Number of maximum workers
        """
        pass

    @property
    @abstractmethod
    def process_timeout(self) -> int:
        """
        Get the process timeout configured for this executor.

        Returns:
            Process timeout in seconds
        """
        pass


class IParallelExecutorFactory(ABC):
    """
    Abstract factory interface for creating parallel executor instances.

    This factory enables creation of appropriate parallel executor implementations
    based on configuration and system capabilities.
    """

    @abstractmethod
    def create_executor(
        self,
        max_workers: int,
        process_timeout: int = 300,
        use_threads_in_multiprocess: bool = True,
    ) -> IParallelExecutor:
        """
        Create a parallel executor instance.

        Args:
            max_workers: Maximum number of worker processes/threads
            process_timeout: Timeout for a single window evaluation in seconds
            use_threads_in_multiprocess: Use ThreadPoolExecutor when already in multiprocessing context

        Returns:
            Configured parallel executor instance
        """
        pass


# Concrete implementations


class ParallelExecutorAdapter(IParallelExecutor):
    """
    Adapter that wraps the concrete ParallelExecutor to implement the interface.

    This adapter maintains full backward compatibility while enabling dependency inversion.
    """

    def __init__(self, parallel_executor: ParallelExecutor):
        """
        Initialize the adapter with a concrete parallel executor.

        Args:
            parallel_executor: Concrete parallel executor instance
        """
        self._executor = parallel_executor

    def execute_parallel(
        self,
        windows: List[Dict[str, Any]],
        worker_func: Callable,
        fallback_func: Optional[Callable] = None,
    ) -> List[Tuple[Any, pd.Series]]:
        """Execute windows in parallel using the wrapped executor."""
        return self._executor.execute_parallel(windows, worker_func, fallback_func)

    def should_use_parallel(self, num_windows: int, min_windows_threshold: int = 2) -> bool:
        """Determine if parallel processing should be used."""
        return self._executor.should_use_parallel(num_windows, min_windows_threshold)

    @property
    def max_workers(self) -> int:
        """Get the maximum number of workers."""
        return self._executor.max_workers

    @property
    def process_timeout(self) -> int:
        """Get the process timeout."""
        return self._executor.process_timeout


class ParallelExecutorFactory(IParallelExecutorFactory):
    """
    Concrete factory for creating parallel executor instances.

    This factory creates ParallelExecutorAdapter instances that wrap
    the concrete ParallelExecutor implementation.
    """

    def create_executor(
        self,
        max_workers: int,
        process_timeout: int = 300,
        use_threads_in_multiprocess: bool = True,
    ) -> IParallelExecutor:
        """Create a parallel executor instance."""
        concrete_executor = ParallelExecutor(
            max_workers=max_workers,
            process_timeout=process_timeout,
            use_threads_in_multiprocess=use_threads_in_multiprocess,
        )
        return ParallelExecutorAdapter(concrete_executor)


# Factory function for easy creation
def create_parallel_executor_factory() -> IParallelExecutorFactory:
    """
    Create a parallel executor factory instance.

    Returns:
        Configured parallel executor factory
    """
    return ParallelExecutorFactory()


def create_parallel_executor(
    max_workers: int,
    process_timeout: int = 300,
    use_threads_in_multiprocess: bool = True,
) -> IParallelExecutor:
    """
    Create a parallel executor instance using the default factory.

    Args:
        max_workers: Maximum number of worker processes/threads
        process_timeout: Timeout for a single window evaluation in seconds
        use_threads_in_multiprocess: Use ThreadPoolExecutor when already in multiprocessing context

    Returns:
        Configured parallel executor instance
    """
    factory = create_parallel_executor_factory()
    return factory.create_executor(max_workers, process_timeout, use_threads_in_multiprocess)
