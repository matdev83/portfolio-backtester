"""
Hybrid parallelism for genetic algorithm optimization.

This module provides a hybrid approach that uses thread-level parallelism for window
evaluation and process-level parallelism for population evaluation, maximizing
CPU utilization while reducing overhead.
"""

import os
import threading
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union, cast
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

from .evaluator import BacktestEvaluator

if TYPE_CHECKING:
    from .results import EvaluationResult, OptimizationData, OptimizationDataContext
    from portfolio_backtester.backtesting.strategy_backtester import StrategyBacktester


class HybridParallelismManager:
    """
    Manages hybrid parallelism for genetic algorithm optimization.

    This class coordinates thread-level parallelism for window evaluation within
    each process and process-level parallelism for evaluating different individuals
    in the population.
    """

    def __init__(
        self,
        n_processes: int = 1,
        n_threads_per_process: Optional[int] = None,
        thread_batch_size: int = 1,
    ):
        """
        Initialize the hybrid parallelism manager.

        Args:
            n_processes: Number of parallel processes to use
            n_threads_per_process: Threads per process (None = auto based on CPU cores)
            thread_batch_size: Batch size for thread-level parallelism
        """
        # Normalize special values: -1 or 0 => use all CPU cores
        try:
            cpu_count = os.cpu_count() or 1
        except Exception:
            cpu_count = 1

        # Calculate optimal process/thread distribution
        self.n_processes = cpu_count if (not n_processes or n_processes < 0) else n_processes

        # Default thread count is based on available cores and process count
        if n_threads_per_process is None:
            # Use at least 2 threads per process but don't oversubscribe
            self.n_threads_per_process = max(2, cpu_count // self.n_processes)
        else:
            self.n_threads_per_process = n_threads_per_process

        self.thread_batch_size = thread_batch_size

        # State
        self._worker_pools: Dict[int, ThreadPoolExecutor] = {}
        self._lock = threading.RLock()

        logger.debug(
            f"HybridParallelismManager initialized with {self.n_processes} processes, "
            f"{self.n_threads_per_process} threads per process"
        )

    def _get_thread_pool(self) -> ThreadPoolExecutor:
        """
        Get or create a thread pool for the current process.

        Thread pools are stored per-process to avoid issues with multiprocessing.

        Returns:
            ThreadPoolExecutor instance for the current process
        """
        process_id = os.getpid()

        with self._lock:
            if process_id not in self._worker_pools:
                self._worker_pools[process_id] = ThreadPoolExecutor(
                    max_workers=self.n_threads_per_process
                )

            return self._worker_pools[process_id]

    def _evaluate_windows_threaded(
        self,
        evaluator: "BacktestEvaluator",
        params: Dict[str, Any],
        scenario_config: Dict[str, Any],
        data: "OptimizationData",
        backtester: "StrategyBacktester",
    ) -> "EvaluationResult":
        """
        Evaluate a parameter set across multiple windows using thread-level parallelism.

        This method is designed to be called inside each worker process and uses
        threads to parallelize window evaluation within the process.

        Args:
            evaluator: The evaluator instance
            params: Parameter set to evaluate
            scenario_config: Scenario configuration
            data: Optimization data
            backtester: Backtester instance

        Returns:
            Evaluation result
        """
        # Ensure thread pool is initialized for this process
        self._get_thread_pool()

        # Set thread-level parallelism in the evaluator
        # This temporarily overrides n_jobs to use our thread count
        original_n_jobs = evaluator.n_jobs
        evaluator.n_jobs = self.n_threads_per_process

        try:
            # Delegate to standard evaluation with thread-level parallelism
            result = evaluator.evaluate_parameters(params, scenario_config, data, backtester)
            return result
        finally:
            # Restore original n_jobs
            evaluator.n_jobs = original_n_jobs

    def evaluate_population(
        self,
        population: List[Dict[str, Any]],
        scenario_config: Dict[str, Any],
        data: Union["OptimizationData", "OptimizationDataContext"],
        backtester: "StrategyBacktester",
        evaluator: "BacktestEvaluator",
    ) -> List["EvaluationResult"]:
        """
        Evaluate a population using hybrid parallelism.

        This method distributes population evaluation across processes and
        window evaluation across threads within each process.

        Args:
            population: The population to evaluate
            scenario_config: Scenario configuration
            data: Optimization data or data context
            backtester: Backtester instance
            evaluator: Evaluator instance

        Returns:
            List of evaluation results
        """
        from joblib import Parallel, delayed  # type: ignore

        if not population:
            return []

        # Temporarily disable thread-level parallelism in the evaluator
        # This will be re-enabled inside each worker process
        original_n_jobs = evaluator.n_jobs
        evaluator.n_jobs = 1

        try:
            # Use process-level parallelism for population evaluation
            # Each process will use thread-level parallelism internally
            is_memmap = hasattr(data, "daily_data_path")

            if is_memmap:
                # Memory-mapped data path
                data_context = cast("OptimizationDataContext", data)
                results = Parallel(
                    n_jobs=self.n_processes,
                    prefer="processes",
                )(
                    delayed(self._hybrid_evaluate_with_memmap)(
                        params,
                        scenario_config,
                        data_context,
                        backtester,
                        evaluator,
                        self.n_threads_per_process,
                    )
                    for params in population
                )
            else:
                # Standard data path
                results = Parallel(
                    n_jobs=self.n_processes,
                    prefer="processes",
                )(
                    delayed(self._hybrid_evaluate_with_context)(
                        params,
                        scenario_config,
                        data,
                        backtester,
                        evaluator,
                        self.n_threads_per_process,
                    )
                    for params in population
                )

            return [r for r in results if r is not None]

        finally:
            # Restore original n_jobs
            evaluator.n_jobs = original_n_jobs

    @staticmethod
    def _hybrid_evaluate_with_context(
        params: Dict[str, Any],
        scenario_config: Dict[str, Any],
        data: "OptimizationData",
        backtester: "StrategyBacktester",
        evaluator: "BacktestEvaluator",
        n_threads: int,
    ) -> "EvaluationResult":
        """
        Static method for joblib to evaluate with context and thread-level parallelism.

        Args:
            params: Parameter set to evaluate
            scenario_config: Scenario configuration
            data: Optimization data
            backtester: Backtester instance
            evaluator: Evaluator instance
            n_threads: Number of threads to use

        Returns:
            Evaluation result
        """
        # Initialize worker context
        from .ga_worker_context import ensure_initialized

        ensure_initialized(scenario_config, data, backtester, evaluator)

        # Set thread-level parallelism
        original_n_jobs = evaluator.n_jobs
        evaluator.n_jobs = n_threads

        try:
            # Execute with thread-level parallelism
            return evaluator.evaluate_parameters(params, scenario_config, data, backtester)
        finally:
            evaluator.n_jobs = original_n_jobs

    @staticmethod
    def _hybrid_evaluate_with_memmap(
        params: Dict[str, Any],
        scenario_config: Dict[str, Any],
        data_context: "OptimizationDataContext",
        backtester: "StrategyBacktester",
        evaluator: "BacktestEvaluator",
        n_threads: int,
    ) -> "EvaluationResult":
        """
        Static method for joblib to evaluate with memory-mapped context and thread-level parallelism.

        Args:
            params: Parameter set to evaluate
            scenario_config: Scenario configuration
            data_context: Optimization data context with memory-mapped data
            backtester: Backtester instance
            evaluator: Evaluator instance
            n_threads: Number of threads to use

        Returns:
            Evaluation result
        """
        # Reconstruct data from memory-mapped files
        from .data_context import reconstruct_optimization_data

        try:
            reconstructed_data = reconstruct_optimization_data(data_context)

            # Initialize worker context
            from .ga_worker_context import ensure_initialized

            ensure_initialized(scenario_config, reconstructed_data, backtester, evaluator)

            # Set thread-level parallelism
            original_n_jobs = evaluator.n_jobs
            evaluator.n_jobs = n_threads

            try:
                # Execute with thread-level parallelism
                return evaluator.evaluate_parameters(
                    params, scenario_config, reconstructed_data, backtester
                )
            finally:
                evaluator.n_jobs = original_n_jobs

        except Exception as e:
            logger.error(f"Failed to evaluate with memory-mapped data: {e}")
            raise

    def cleanup(self) -> None:
        """Clean up resources."""
        with self._lock:
            for pool in self._worker_pools.values():
                try:
                    pool.shutdown(wait=False)
                except Exception as e:
                    logger.warning(f"Error shutting down thread pool: {e}")

            self._worker_pools.clear()

    def __del__(self) -> None:
        """Clean up resources when the object is destroyed."""
        self.cleanup()
