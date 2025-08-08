"""
Window management for WFO parallel processing.

This module provides window configuration, worker setup, and orchestration
for parallel WFO processing. It handles both parallel and sequential execution
strategies.

This implementation now supports Dependency Inversion Principle (DIP) by
accepting interfaces for mathematical operations dependencies.
"""

import pandas as pd
import logging
import multiprocessing as mp
from typing import Dict, Any, List, Tuple, Optional, Callable
from functools import partial

logger = logging.getLogger(__name__)


class WindowManager:
    """
    Manages WFO window processing configuration and orchestration.

    This class is responsible for:
    - Managing window processing configuration
    - Setting up worker functions and parameters
    - Orchestrating between parallel and sequential processing
    - Handling CPU core detection and worker allocation
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        enable_parallel: bool = True,
        math_operations: Optional[Any] = None,
    ):
        """
        Initialize the window manager.

        Args:
            max_workers: Maximum number of worker processes (default: CPU count - 1)
            enable_parallel: Whether parallel processing is enabled
            math_operations: Optional math operations interface (DIP)
        """
        self.enable_parallel = enable_parallel

        # Initialize math operations dependency (DIP)
        if math_operations is not None:
            self._math_operations = math_operations
        else:
            # Import here to avoid circular imports
            from ...interfaces.math_operations_interface import create_math_operations

            self._math_operations = create_math_operations()

        if max_workers is None:
            available_cores = mp.cpu_count()
            self.max_workers = self._math_operations.max_value(1, available_cores - 1)
        else:
            self.max_workers = self._math_operations.max_value(1, max_workers)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"WindowManager initialized: parallel={enable_parallel}, max_workers={self.max_workers}"
            )

    def should_use_parallel_processing(
        self, windows: List[Dict[str, Any]], min_windows_for_parallel: int = 2
    ) -> bool:
        """
        Determine if parallel processing should be used based on configuration and window count.

        Args:
            windows: List of WFO window configurations
            min_windows_for_parallel: Minimum windows required for parallel processing

        Returns:
            True if parallel processing should be used
        """
        return (
            self.enable_parallel
            and len(windows) >= min_windows_for_parallel
            and self.max_workers > 1
        )

    def prepare_worker_function(
        self,
        evaluate_window_func: Callable[
            [Dict[str, Any], Dict[str, Any], Dict[str, Any]], Tuple[Any, pd.Series]
        ],
        scenario_config: Dict[str, Any],
        shared_data: Dict[str, Any],
    ) -> Callable:
        """
        Prepare the worker function with shared configuration and data.

        Args:
            evaluate_window_func: Function to evaluate a single window
            scenario_config: Scenario configuration
            shared_data: Shared data needed for evaluation

        Returns:
            Prepared worker function (partial) ready for parallel execution
        """
        worker_func = partial(
            _evaluate_window_worker,
            evaluate_window_func=evaluate_window_func,
            scenario_config=scenario_config,
            shared_data=shared_data,
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Worker function prepared with shared configuration and data")

        return worker_func

    def process_windows_sequential(
        self,
        windows: List[Dict[str, Any]],
        evaluate_window_func: Callable[
            [Dict[str, Any], Dict[str, Any], Dict[str, Any]], Tuple[Any, pd.Series]
        ],
        scenario_config: Dict[str, Any],
        shared_data: Dict[str, Any],
    ) -> List[Tuple[Any, pd.Series]]:
        """
        Process WFO windows sequentially.

        This is used as a fallback when parallel processing is disabled,
        fails, or is not beneficial for the given number of windows.

        Args:
            windows: List of WFO window configurations
            evaluate_window_func: Function to evaluate a single window
            scenario_config: Scenario configuration
            shared_data: Shared data needed for evaluation

        Returns:
            List of (objective_value, returns) tuples for each window
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Processing {len(windows)} WFO windows sequentially")

        results = []
        for i, window in enumerate(windows):
            try:
                result = evaluate_window_func(window, scenario_config, shared_data)
                results.append(result)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Sequential window {i} completed successfully")
            except Exception as e:
                logger.error(f"Sequential window {i} failed: {e}")
                results.append((float("nan"), pd.Series(dtype=float)))

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Sequential processing completed {len(results)} windows")

        return results

    def get_worker_allocation_info(self) -> Dict[str, Any]:
        """
        Get information about worker allocation and system resources.

        Returns:
            Dictionary with worker allocation details
        """
        total_cores = mp.cpu_count()

        return {
            "total_cpu_cores": total_cores,
            "max_workers": self.max_workers,
            "enable_parallel": self.enable_parallel,
            "worker_utilization": self.max_workers / total_cores if total_cores > 0 else 0.0,
            "reserved_cores": total_cores - self.max_workers,
        }

    def create_sequential_fallback(
        self,
        windows: List[Dict[str, Any]],
        evaluate_window_func: Callable,
        scenario_config: Dict[str, Any],
        shared_data: Dict[str, Any],
    ) -> Callable[[], List[Tuple[Any, pd.Series]]]:
        """
        Create a fallback function for sequential processing.

        This creates a closure that can be called if parallel processing fails.

        Args:
            windows: List of WFO window configurations
            evaluate_window_func: Function to evaluate a single window
            scenario_config: Scenario configuration
            shared_data: Shared data needed for evaluation

        Returns:
            Fallback function that returns sequential processing results
        """

        def fallback_function():
            return self.process_windows_sequential(
                windows, evaluate_window_func, scenario_config, shared_data
            )

        return fallback_function


def _evaluate_window_worker(
    window_idx: int,
    window: Dict[str, Any],
    evaluate_window_func: Callable[
        [Dict[str, Any], Dict[str, Any], Dict[str, Any]], Tuple[Any, pd.Series]
    ],
    scenario_config: Dict[str, Any],
    shared_data: Dict[str, Any],
) -> Tuple[Any, pd.Series]:
    """
    Worker function for parallel window evaluation.

    This function runs in a separate process and evaluates a single WFO window.
    It's designed to be used with ProcessPoolExecutor for parallel processing.

    Args:
        window_idx: Index of the window being processed
        window: WFO window configuration
        evaluate_window_func: Function to evaluate the window
        scenario_config: Scenario configuration
        shared_data: Shared data needed for evaluation

    Returns:
        Tuple of (objective_value, returns) for the window

    Raises:
        Exception: Re-raises any exception that occurs during window evaluation
    """
    try:
        # Set up logging for the worker process
        worker_logger = logging.getLogger(f"WFOWorker-{window_idx}")
        if worker_logger.isEnabledFor(logging.DEBUG):
            worker_logger.debug(f"Processing window {window_idx}")

        # Evaluate the window using the provided function
        result = evaluate_window_func(window, scenario_config, shared_data)

        if worker_logger.isEnabledFor(logging.DEBUG):
            worker_logger.debug(f"Window {window_idx} evaluation completed successfully")

        return result

    except Exception as e:
        worker_logger = logging.getLogger(f"WFOWorker-{window_idx}")
        worker_logger.error(f"Error in window {window_idx}: {e}")
        raise
