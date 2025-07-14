"""
Parallel Walk-Forward Optimization implementation.

This module provides parallel processing capabilities for WFO windows to utilize
multi-core systems effectively. Each WFO window can be processed independently,
making this a safe and effective optimization.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import multiprocessing as mp
import pickle
import time
from functools import partial

logger = logging.getLogger(__name__)


class ParallelWFOProcessor:
    """
    Parallel processor for Walk-Forward Optimization windows.
    
    This class handles the parallel execution of WFO windows across multiple
    CPU cores to significantly speed up optimization processes.
    """
    
    def __init__(self, max_workers: Optional[int] = None, enable_parallel: bool = True, process_timeout: int = 300):
        """
        Initialize the parallel WFO processor.
        
        Args:
            max_workers: Maximum number of worker processes (default: CPU count - 1)
            enable_parallel: Whether to enable parallel processing
            process_timeout: Timeout for a single window evaluation in seconds
        """
        self.enable_parallel = enable_parallel
        self.process_timeout = process_timeout
        
        if max_workers is None:
            # Use CPU count - 1 to leave one core for the main process
            available_cores = mp.cpu_count()
            self.max_workers = max(1, available_cores - 1)
        else:
            self.max_workers = max(1, max_workers)
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"ParallelWFOProcessor initialized: "
                        f"parallel={enable_parallel}, max_workers={self.max_workers}")
    
    def process_windows_parallel(self, 
                                windows: List[Dict[str, Any]], 
                                evaluate_window_func: callable,
                                scenario_config: Dict[str, Any],
                                shared_data: Dict[str, Any]) -> List[Tuple[Any, pd.Series]]:
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
        if not self.enable_parallel or len(windows) <= 1:
            # Fall back to sequential processing
            return self._process_windows_sequential(
                windows, evaluate_window_func, scenario_config, shared_data
            )
        
        if logger.isEnabledFor(logging.DEBUG):

        
            if logger.isEnabledFor(logging.DEBUG):


        
                logger.debug(f"Processing {len(windows)} WFO windows in parallel using {self.max_workers} workers")
        start_time = time.time()
        
        try:
            # Prepare worker function with shared data
            worker_func = partial(
                _evaluate_window_worker,
                evaluate_window_func=evaluate_window_func,
                scenario_config=scenario_config,
                shared_data=shared_data
            )
            
            results = []
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all windows for processing
                future_to_window = {
                    executor.submit(worker_func, i, window): (i, window)
                    for i, window in enumerate(windows)
                }
                
                # Collect results as they complete
                window_results = [None] * len(windows)
                for future in as_completed(future_to_window):
                    window_idx, window = future_to_window[future]
                    try:
                        result = future.result(timeout=self.process_timeout)
                        window_results[window_idx] = result
                        if logger.isEnabledFor(logging.DEBUG):

                            logger.debug(f"Window {window_idx} completed successfully")
                    except TimeoutError:
                        logger.error(f"Window {window_idx} timed out after {self.process_timeout} seconds")
                        window_results[window_idx] = (float('nan'), pd.Series(dtype=float))
                    except Exception as e:
                        logger.error(f"Window {window_idx} failed: {e}")
                        # Use fallback result for failed windows
                        window_results[window_idx] = (float('nan'), pd.Series(dtype=float))
                
                results = [r for r in window_results if r is not None]
            
            elapsed_time = time.time() - start_time
            if logger.isEnabledFor(logging.DEBUG):

                logger.debug(f"Parallel WFO processing completed in {elapsed_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(f"Parallel processing failed: {e}. Falling back to sequential processing.")
            return self._process_windows_sequential(
                windows, evaluate_window_func, scenario_config, shared_data
            )
    
    def _process_windows_sequential(self, 
                                   windows: List[Dict[str, Any]], 
                                   evaluate_window_func: callable,
                                   scenario_config: Dict[str, Any],
                                   shared_data: Dict[str, Any]) -> List[Tuple[Any, pd.Series]]:
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
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Processing {len(windows)} WFO windows sequentially")
        start_time = time.time()
        
        results = []
        for i, window in enumerate(windows):
            try:
                result = evaluate_window_func(window, scenario_config, shared_data)
                results.append(result)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Sequential window {i} completed successfully")
            except Exception as e:
                logger.error(f"Sequential window {i} failed: {e}")
                results.append((float('nan'), pd.Series(dtype=float)))
        
        elapsed_time = time.time() - start_time
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Sequential WFO processing completed in {elapsed_time:.2f} seconds")
        
        return results
    
    def estimate_parallel_benefit(self, num_windows: int, avg_window_time: float) -> Dict[str, float]:
        """
        Estimate the potential benefit of parallel processing.
        
        Args:
            num_windows: Number of WFO windows
            avg_window_time: Average time per window in seconds
            
        Returns:
            Dictionary with timing estimates
        """
        sequential_time = num_windows * avg_window_time
        
        # Estimate parallel time (with overhead)
        parallel_overhead = 0.1  # 10% overhead for process management
        ideal_parallel_time = sequential_time / self.max_workers
        estimated_parallel_time = ideal_parallel_time * (1 + parallel_overhead)
        
        speedup = sequential_time / estimated_parallel_time
        time_saved = sequential_time - estimated_parallel_time
        
        return {
            'sequential_time': sequential_time,
            'estimated_parallel_time': estimated_parallel_time,
            'estimated_speedup': speedup,
            'estimated_time_saved': time_saved,
            'max_workers': self.max_workers,
            'parallel_overhead': parallel_overhead
        }


def _evaluate_window_worker(window_idx: int, 
                           window: Dict[str, Any],
                           evaluate_window_func: callable,
                           scenario_config: Dict[str, Any],
                           shared_data: Dict[str, Any]) -> Tuple[Any, pd.Series]:
    """
    Worker function for parallel window evaluation.
    
    This function runs in a separate process and evaluates a single WFO window.
    
    Args:
        window_idx: Index of the window being processed
        window: WFO window configuration
        evaluate_window_func: Function to evaluate the window
        scenario_config: Scenario configuration
        shared_data: Shared data needed for evaluation
        
    Returns:
        Tuple of (objective_value, returns) for the window
    """
    try:
        # Set up logging for the worker process
        worker_logger = logging.getLogger(f"WFOWorker-{window_idx}")
        worker_logger.debug(f"Processing window {window_idx}")
        
        # Evaluate the window
        result = evaluate_window_func(window, scenario_config, shared_data)
        
        worker_logger.debug(f"Window {window_idx} evaluation completed")
        return result
        
    except Exception as e:
        worker_logger = logging.getLogger(f"WFOWorker-{window_idx}")
        worker_logger.error(f"Error in window {window_idx}: {e}")
        raise


def create_parallel_wfo_processor(config: Dict[str, Any]) -> ParallelWFOProcessor:
    """
    Create a ParallelWFOProcessor from configuration.
    
    Args:
        config: Configuration dictionary with parallel processing settings
        
    Returns:
        Configured ParallelWFOProcessor instance
    """
    parallel_config = config.get('parallel_wfo_config', {})
    
    enable_parallel = parallel_config.get('enable_parallel', True)
    max_workers = parallel_config.get('max_workers', None)
    process_timeout = parallel_config.get('process_timeout', 300)
    
    # Disable parallel processing for small numbers of windows
    min_windows_for_parallel = parallel_config.get('min_windows_for_parallel', 2)
    
    processor = ParallelWFOProcessor(
        max_workers=max_workers,
        enable_parallel=enable_parallel,
        process_timeout=process_timeout
    )
    
    processor.min_windows_for_parallel = min_windows_for_parallel
    
    return processor


# Configuration defaults for parallel WFO processing
DEFAULT_PARALLEL_WFO_CONFIG = {
    'enable_parallel': True,
    'max_workers': None,  # Auto-detect based on CPU count
    'min_windows_for_parallel': 2,  # Minimum windows to enable parallel processing
    'process_timeout': 300,  # Timeout per window in seconds
    'memory_limit_mb': 1000,  # Memory limit per worker process
}