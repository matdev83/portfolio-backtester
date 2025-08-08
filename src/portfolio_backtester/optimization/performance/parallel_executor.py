"""
Parallel execution engine for WFO windows.

This module provides the core parallel processing capabilities using ProcessPoolExecutor
to distribute WFO windows across multiple CPU cores for optimal performance.
"""

import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, TimeoutError
import time
import os

logger = logging.getLogger(__name__)


class ParallelExecutor:
    """
    Handles parallel execution of WFO windows using ProcessPoolExecutor.

    This class is responsible for:
    - Managing ProcessPoolExecutor lifecycle
    - Distributing work across worker processes
    - Handling timeouts and error recovery
    - Collecting results from completed futures
    """

    def __init__(
        self, max_workers: int, process_timeout: int = 300, use_threads_in_multiprocess: bool = True
    ):
        """
        Initialize the parallel executor.

        Args:
            max_workers: Maximum number of worker processes/threads
            process_timeout: Timeout for a single window evaluation in seconds
            use_threads_in_multiprocess: Use ThreadPoolExecutor when already in multiprocessing context
        """
        self.max_workers = max_workers
        self.process_timeout = process_timeout
        self.use_threads_in_multiprocess = use_threads_in_multiprocess

        # Detect if we're in a multiprocessing context
        self.in_multiprocess_context = os.getenv("OPTUNA_WORKER_PROCESS") is not None or hasattr(
            os, "getppid"
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"ParallelExecutor initialized: max_workers={max_workers}, timeout={process_timeout}, "
                f"use_threads_in_multiprocess={use_threads_in_multiprocess}, "
                f"in_multiprocess_context={self.in_multiprocess_context}"
            )

    def execute_parallel(
        self,
        windows: List[Dict[str, Any]],
        worker_func: Callable,
        fallback_func: Optional[Callable] = None,
    ) -> List[Tuple[Any, pd.Series]]:
        """
        Execute WFO windows in parallel using ProcessPoolExecutor.

        Args:
            windows: List of WFO window configurations
            worker_func: Prepared worker function (partial) to execute
            fallback_func: Optional fallback function if parallel processing fails

        Returns:
            List of (objective_value, returns) tuples for each window
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Executing {len(windows)} windows in parallel using {self.max_workers} workers"
            )

        start_time = time.time()

        try:
            results = []
            # Choose executor type based on context to avoid resource conflicts
            use_threads = self.use_threads_in_multiprocess and self.in_multiprocess_context
            executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
            executor_type = "thread" if use_threads else "process"

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Using {executor_type} pool executor for window parallelization")

            with executor_class(max_workers=self.max_workers) as executor:
                # Submit all windows for processing
                future_to_window = {
                    executor.submit(worker_func, i, window): (i, window)
                    for i, window in enumerate(windows)
                }

                # Collect results as they complete
                window_results: List[Optional[Tuple[Any, pd.Series]]] = [None] * len(windows)
                for future in as_completed(future_to_window):
                    window_idx, window = future_to_window[future]
                    try:
                        result = future.result(timeout=self.process_timeout)
                        window_results[window_idx] = result
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Window {window_idx} completed successfully")
                    except TimeoutError:
                        logger.error(
                            f"Window {window_idx} timed out after {self.process_timeout} seconds"
                        )
                        window_results[window_idx] = (float("nan"), pd.Series(dtype=float))
                    except Exception as e:
                        logger.error(f"Window {window_idx} failed: {e}")
                        # Use fallback result for failed windows
                        window_results[window_idx] = (float("nan"), pd.Series(dtype=float))

                results = [r for r in window_results if r is not None]

            elapsed_time = time.time() - start_time
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Parallel execution completed in {elapsed_time:.2f} seconds")

            return results

        except Exception as e:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(f"Parallel execution failed: {e}")

            if fallback_func is not None:
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning("Falling back to alternative execution method")
                fallback_results: List[Tuple[Any, pd.Series]] = fallback_func()
                return fallback_results
            else:
                raise

    def should_use_parallel(self, num_windows: int, min_windows_threshold: int = 2) -> bool:
        """
        Determine if parallel processing should be used based on the number of windows.

        Args:
            num_windows: Number of windows to process
            min_windows_threshold: Minimum windows required for parallel processing

        Returns:
            True if parallel processing should be used
        """
        return num_windows >= min_windows_threshold and self.max_workers > 1
