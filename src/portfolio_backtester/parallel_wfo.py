"""
Parallel Walk-Forward Optimization implementation.

This module provides parallel processing capabilities for WFO windows to utilize
multi-core systems effectively. Each WFO window can be processed independently,
making this a safe and effective optimization.

This implementation follows SOLID principles by separating concerns into
focused classes while maintaining backward compatibility through composition.

This implementation now supports Dependency Inversion Principle (DIP) by
accepting interfaces for parallel execution and math operations dependencies.
"""

import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional, Callable
import time

from .optimization.performance.window_manager import WindowManager
from .interfaces.parallel_executor_interface import (
    IParallelExecutor,
    create_parallel_executor,
)
from .interfaces.parallel_benefit_estimator_interface import (
    IParallelBenefitEstimator,
    create_parallel_benefit_estimator,
)
from .interfaces.math_operations_interface import (
    IMathOperations,
    create_math_operations,
)

logger = logging.getLogger(__name__)


class ParallelWFOProcessor:
    """
    Parallel processor for Walk-Forward Optimization windows.

    This class handles the parallel execution of WFO windows across multiple
    CPU cores to significantly speed up optimization processes.

    This implementation uses composition to delegate responsibilities to
    focused classes while maintaining full backward compatibility.
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        enable_parallel: bool = True,
        process_timeout: int = 300,
        parallel_executor: Optional[IParallelExecutor] = None,
        benefit_estimator: Optional[IParallelBenefitEstimator] = None,
        math_operations: Optional[IMathOperations] = None,
    ):
        """
        Initialize the parallel WFO processor.

        Args:
            max_workers: Maximum number of worker processes (default: CPU count - 1)
            enable_parallel: Whether to enable parallel processing
            process_timeout: Timeout for a single window evaluation in seconds
            parallel_executor: Optional parallel executor interface (DIP)
            benefit_estimator: Optional benefit estimator interface (DIP)
            math_operations: Optional math operations interface (DIP)
        """
        # Store original parameters for backward compatibility
        self.enable_parallel = enable_parallel
        self.process_timeout = process_timeout

        # Initialize math operations dependency (DIP)
        self._math_operations = math_operations or create_math_operations()

        # Initialize composed components with DIP-aware initialization
        self._window_manager = WindowManager(
            max_workers=max_workers,
            enable_parallel=enable_parallel,
            math_operations=self._math_operations,
        )

        # Use dependency injection for parallel executor (DIP)
        if parallel_executor is not None:
            self._parallel_executor = parallel_executor
        else:
            # Create using interface factory for backward compatibility
            self._parallel_executor = create_parallel_executor(
                max_workers=self._window_manager.max_workers,
                process_timeout=process_timeout,
            )

        # Use dependency injection for benefit estimator (DIP)
        if benefit_estimator is not None:
            self._benefit_estimator = benefit_estimator
        else:
            # Create using interface factory for backward compatibility
            self._benefit_estimator = create_parallel_benefit_estimator(
                max_workers=self._window_manager.max_workers,
                math_operations=self._math_operations,
            )

        # Maintain backward compatibility for max_workers attribute
        self.max_workers = self._window_manager.max_workers

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"ParallelWFOProcessor initialized: "
                f"parallel={enable_parallel}, max_workers={self.max_workers}"
            )

    def process_windows_parallel(
        self,
        windows: List[Dict[str, Any]],
        evaluate_window_func: Callable[
            [Dict[str, Any], Dict[str, Any], Dict[str, Any]], Tuple[Any, pd.Series]
        ],
        scenario_config: Dict[str, Any],
        shared_data: Dict[str, Any],
    ) -> List[Tuple[Any, pd.Series]]:
        """
        Process WFO windows in parallel.

        Args:
            windows: List of WFO window configurations
            evaluate_window_func: Function to evaluate a single window
            scenario_config: Scenario configuration
            shared_data: Shared data needed for evaluation

        Returns:
            List of (objective_value, returns) tuples for each window
        """
        # Use WindowManager to determine processing strategy
        if not self._window_manager.should_use_parallel_processing(windows):
            # Fall back to sequential processing
            return self._process_windows_sequential(
                windows, evaluate_window_func, scenario_config, shared_data
            )

        start_time = time.time()

        # Prepare worker function using WindowManager
        worker_func = self._window_manager.prepare_worker_function(
            evaluate_window_func, scenario_config, shared_data
        )

        # Create fallback function using WindowManager
        fallback_func = self._window_manager.create_sequential_fallback(
            windows, evaluate_window_func, scenario_config, shared_data
        )

        # Execute using ParallelExecutor
        results = self._parallel_executor.execute_parallel(windows, worker_func, fallback_func)

        elapsed_time = time.time() - start_time
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Parallel WFO processing completed in {elapsed_time:.2f} seconds")

        return results

    def _process_windows_sequential(
        self,
        windows: List[Dict[str, Any]],
        evaluate_window_func: Callable[
            [Dict[str, Any], Dict[str, Any], Dict[str, Any]], Tuple[Any, pd.Series]
        ],
        scenario_config: Dict[str, Any],
        shared_data: Dict[str, Any],
    ) -> List[Tuple[Any, pd.Series]]:
        """
        Process WFO windows sequentially (fallback method).

        Args:
            windows: List of WFO window configurations
            evaluate_window_func: Function to evaluate a single window
            scenario_config: Scenario configuration
            shared_data: Shared data needed for evaluation

        Returns:
            List of (objective_value, returns) tuples for each window
        """
        start_time = time.time()

        # Delegate to WindowManager for sequential processing
        results = self._window_manager.process_windows_sequential(
            windows, evaluate_window_func, scenario_config, shared_data
        )

        elapsed_time = time.time() - start_time
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Sequential WFO processing completed in {elapsed_time:.2f} seconds")

        return results

    def estimate_parallel_benefit(
        self, num_windows: int, avg_window_time: float
    ) -> Dict[str, float]:
        """
        Estimate the potential benefit of parallel processing.

        Args:
            num_windows: Number of WFO windows
            avg_window_time: Average time per window in seconds

        Returns:
            Dictionary with timing estimates
        """
        # Delegate to ParallelBenefitEstimator
        return self._benefit_estimator.estimate_parallel_benefit(num_windows, avg_window_time)


def create_parallel_wfo_processor(
    config: Dict[str, Any],
    parallel_executor: Optional[IParallelExecutor] = None,
    benefit_estimator: Optional[IParallelBenefitEstimator] = None,
    math_operations: Optional[IMathOperations] = None,
) -> ParallelWFOProcessor:
    """
    Create a ParallelWFOProcessor from configuration with optional dependency injection.

    Args:
        config: Configuration dictionary with parallel processing settings
        parallel_executor: Optional parallel executor interface (DIP)
        benefit_estimator: Optional benefit estimator interface (DIP)
        math_operations: Optional math operations interface (DIP)

    Returns:
        Configured ParallelWFOProcessor instance
    """
    parallel_config = config.get("parallel_wfo_config", {})

    enable_parallel = parallel_config.get("enable_parallel", True)
    max_workers = parallel_config.get("max_workers", None)
    process_timeout = parallel_config.get("process_timeout", 300)

    # Disable parallel processing for small numbers of windows
    min_windows_for_parallel = parallel_config.get("min_windows_for_parallel", 2)

    processor = ParallelWFOProcessor(
        max_workers=max_workers,
        enable_parallel=enable_parallel,
        process_timeout=process_timeout,
        parallel_executor=parallel_executor,
        benefit_estimator=benefit_estimator,
        math_operations=math_operations,
    )
    # Store config value on instance safely
    setattr(processor, "min_windows_for_parallel", min_windows_for_parallel)

    return processor


# Configuration defaults for parallel WFO processing
DEFAULT_PARALLEL_WFO_CONFIG = {
    "enable_parallel": True,
    "max_workers": None,  # Auto-detect based on CPU count
    "min_windows_for_parallel": 2,  # Minimum windows to enable parallel processing
    "process_timeout": 300,  # Timeout per window in seconds
    "memory_limit_mb": 1000,  # Memory limit per worker process
}
