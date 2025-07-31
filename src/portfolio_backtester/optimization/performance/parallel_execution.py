"""
Abstract parallel execution interfaces and base implementations for optimization.
"""

import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from multiprocessing import Pool, cpu_count
import functools

from .interfaces import AbstractParallelRunner

logger = logging.getLogger(__name__)


class BaseParallelRunner(AbstractParallelRunner, ABC):
    """
    Abstract base class for parallel execution implementations.
    
    This class defines the interface for parallel execution that can be 
    implemented by different optimizers.
    """
    
    def __init__(self, n_jobs: int = 1, **kwargs):
        """
        Initialize the parallel runner.
        
        Args:
            n_jobs: Number of parallel jobs to run
            **kwargs: Additional configuration parameters
        """
        self.n_jobs = n_jobs if n_jobs > 0 else cpu_count()
        self.config = kwargs
        self.pool = None
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    
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
    
    def __init__(self, n_jobs: int = 1, **kwargs):
        """
        Initialize the generic parallel runner.
        
        Args:
            n_jobs: Number of parallel jobs to run
            **kwargs: Additional configuration parameters
        """
        super().__init__(n_jobs, **kwargs)
        self.task_function = None
    
    def set_task_function(self, func) -> None:
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
        if self.pool is None:
            self.start()
        
        if self.task_function is None:
            raise ValueError("Task function not set. Call set_task_function() first.")
        
        # Extract tasks from config
        tasks = config.get('tasks', [])
        if not tasks:
            logger.warning("No tasks provided for parallel execution")
            return []
        
        try:
            # Execute tasks in parallel
            if self.pool is not None:
                results = self.pool.map(self.task_function, tasks)
                logger.info(f"Completed {len(results)} parallel tasks")
                return results
            else:
                raise RuntimeError("Parallel execution pool not initialized")
        except Exception as e:
            logger.error(f"Error during parallel execution: {e}")
            raise


def parallelize_function(func, n_jobs: int = 1):
    """
    Decorator to parallelize a function execution.
    
    Args:
        func: Function to parallelize
        n_jobs: Number of parallel jobs to run
        
    Returns:
        Parallelized function
    """
    @functools.wraps(func)
    def wrapper(tasks, *args, **kwargs):
        if n_jobs <= 1:
            # Sequential execution
            return [func(task, *args, **kwargs) for task in tasks]
        
        # Parallel execution
        with Pool(processes=n_jobs) as pool:
            partial_func = functools.partial(func, *args, **kwargs)
            return pool.map(partial_func, tasks)
    
    return wrapper