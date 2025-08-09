import logging
from typing import Dict, Any, Optional, Callable, List
from abc import ABC, abstractmethod
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
import functools

from .interfaces import (
    AbstractParallelRunner,
)

logger = logging.getLogger(__name__)


class BaseParallelRunner(AbstractParallelRunner, ABC):
    """
    Abstract base class for parallel execution implementations.

    This class defines the interface for parallel execution that can be
    implemented by different optimizers.
    """

    def __init__(self, n_jobs: int = 1, **kwargs: Dict[str, Any]) -> None:
        """
        Initialize the parallel runner.

        Args:
            n_jobs: Number of parallel jobs to run
            **kwargs: Additional configuration parameters
        """
        self.n_jobs = n_jobs if n_jobs > 0 else cpu_count()
        self.config = kwargs
        self.pool: Optional[Pool] = None

    def __enter__(self) -> "BaseParallelRunner":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit with proper exception handling."""
        try:
            self.stop()
        except Exception as cleanup_error:
            logger.error(f"Error during parallel execution cleanup: {cleanup_error}")
            
        # Handle any exceptions that occurred during execution
        if exc_type is not None:
            logger.error(f"Exception in parallel execution context: {exc_type.__name__}: {exc_val}")
            if exc_tb is not None:
                import traceback
                logger.debug(f"Exception traceback: {''.join(traceback.format_tb(exc_tb))}")
            
            # Don't suppress the exception - let it propagate
            return False
        
        return None

    def start(self) -> None:
        """Start the parallel execution pool."""
        if self.pool is None:
            try:
                self.pool = Pool(processes=self.n_jobs)
                logger.info(f"Started parallel execution pool with {self.n_jobs} workers")
            except Exception as e:
                logger.error(f"Failed to start parallel execution pool: {e}")
                raise

    def stop(self) -> None:
        """Stop the parallel execution pool."""
        if self.pool is not None:
            try:
                self.pool.close()
                self.pool.join()
                logger.info("Stopped parallel execution pool")
            except Exception as e:
                logger.error(f"Error stopping parallel execution pool: {e}")
            finally:
                self.pool = None

    @abstractmethod
    def run(self, config: Dict[str, Any]) -> Any:
        """
        Run optimization in parallel.

        Args:
            config: Runner configuration

        Returns:
            Optimization result
        """
        pass


class GenericParallelRunner(BaseParallelRunner):
    """
    Generic parallel runner implementation that can be used with any optimizer.
    """

    def __init__(self, n_jobs: int = 1, **kwargs: Dict[str, Any]) -> None:
        """
        Initialize the generic parallel runner.

        Args:
            n_jobs: Number of parallel jobs to run
            **kwargs: Additional configuration parameters
        """
        super().__init__(n_jobs, **kwargs)
        self.task_function: Optional[Callable[..., Any]] = None

    def set_task_function(self, func: Callable[..., Any]) -> None:
        """
        Set the function to be executed in parallel.

        Args:
            func: Function to execute in parallel
        """
        self.task_function = func

    def run(self, config: Dict[str, Any]) -> Any:
        """
        Run optimization in parallel.

        Args:
            config: Runner configuration containing tasks to execute

        Returns:
            Optimization result
        """
        tasks = config.get("tasks", [])
        if not tasks:
            logger.warning("No tasks provided for parallel execution")
            return []

        if self.task_function is None:
            raise ValueError("Task function not set. Call set_task_function() first.")

        if self.pool is None:
            self.start()

        if self.pool is None:
            raise RuntimeError("Parallel pool is not initialized")

        results = self.pool.map(self.task_function, tasks)
        logger.info(f"Completed {len(results)} parallel tasks")
        return results


def parallelize_function(func: Callable[..., Any], n_jobs: int = 1) -> Callable[..., Any]:
    """
    Decorator to parallelize a function execution.

    Args:
        func: Function to parallelize
        n_jobs: Number of parallel jobs to run

    Returns:
        Parallelized function
    """

    @functools.wraps(func)
    def wrapper(tasks: List[Any], *args: Any, **kwargs: Any) -> List[Any]:
        if n_jobs <= 1:
            return [func(task, *args, **kwargs) for task in tasks]

        with Pool(processes=n_jobs) as pool:
            partial_func = functools.partial(func, *args, **kwargs)
            return pool.map(partial_func, tasks)

    return wrapper
