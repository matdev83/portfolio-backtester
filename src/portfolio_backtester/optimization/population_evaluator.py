import os
import sys
import uuid
from typing import Any, Dict, List, TYPE_CHECKING, Tuple, Optional
from joblib import Parallel, delayed  # type: ignore
from loguru import logger

from .results import EvaluationResult, OptimizationData
from .performance.deduplication_factory import DeduplicationFactory
from .data_context import DataContextManager
from .performance.factory import PerformanceOptimizerFactory
from .adaptive_batch_sizing import AdaptiveBatchSizer
from .hybrid_parallelism import HybridParallelismManager
from .gpu_acceleration import GPUAccelerationManager

if TYPE_CHECKING:
    from .evaluator import BacktestEvaluator
    from portfolio_backtester.backtesting.strategy_backtester import (
        StrategyBacktester,
    )

# logger = logging.getLogger(__name__)


def _init_worker_logging():
    """Silence the logger in worker processes and redirect stdout/stderr."""
    logger.remove()
    logger.add(lambda _: None)
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")


class PopulationEvaluator:
    """Evaluates a population of parameter sets."""

    def __init__(
        self,
        evaluator: "BacktestEvaluator",
        n_jobs: int = 1,
        joblib_batch_size: Any = None,
        joblib_pre_dispatch: Any = None,
        enable_adaptive_batch_sizing: bool = True,
        enable_hybrid_parallelism: bool = True,
        enable_incremental_evaluation: bool = True,
        enable_gpu_acceleration: bool = True,
    ):
        self.evaluator = evaluator
        # Normalize special values: -1 or 0 => use all CPU cores
        try:
            cpu_count = os.cpu_count() or 1
        except Exception:
            cpu_count = 1
        self.n_jobs = cpu_count if (not n_jobs or n_jobs < 0) else n_jobs

        # Lightweight in-memory cache to avoid re-evaluating identical parameter sets
        # Keyed by a deterministic tuple derived from the params dict
        self._result_cache: Dict[Tuple[str, ...], EvaluationResult] = {}
        # Shared dedup abstraction (stats + cross-component consistency)
        self._deduplicator = DeduplicationFactory.create_deduplicator(
            optimizer_type="genetic", config={"enable_deduplication": True}
        )

        # Optional joblib tuning knobs
        self._joblib_batch_size = joblib_batch_size
        self._joblib_pre_dispatch = joblib_pre_dispatch

        # Adaptive batch sizing
        self._enable_adaptive_batch_sizing = enable_adaptive_batch_sizing
        self._batch_sizer = AdaptiveBatchSizer(
            min_batch_size=1, max_batch_size=50, n_jobs=self.n_jobs
        )
        self._last_evaluation_time = 0.0

        # Hybrid parallelism
        self._enable_hybrid_parallelism = enable_hybrid_parallelism
        self._hybrid_manager = None
        if self._enable_hybrid_parallelism:
            # Calculate optimal distribution of processes and threads
            n_processes = max(1, self.n_jobs // 2)
            threads_per_process = max(2, self.n_jobs // n_processes)

            self._hybrid_manager = HybridParallelismManager(
                n_processes=n_processes, n_threads_per_process=threads_per_process
            )

            logger.debug(
                f"Using hybrid parallelism with {n_processes} processes, "
                f"{threads_per_process} threads per process"
            )

        # Incremental evaluation
        self._enable_incremental_evaluation = enable_incremental_evaluation
        self._previous_params: Optional[Dict[str, Any]] = None

        # GPU acceleration
        self._enable_gpu_acceleration = enable_gpu_acceleration
        self._gpu_manager = None
        if self._enable_gpu_acceleration:
            self._gpu_manager = GPUAccelerationManager()
            gpu_available = self._gpu_manager.initialize()
            if gpu_available:
                logger.debug("GPU acceleration enabled and initialized successfully")
            else:
                logger.debug("GPU acceleration requested but not available, falling back to CPU")

        # Track whether workers are initialized for params-only dispatch
        self._workers_initialized = False
        self._run_id = str(uuid.uuid4())[:8]  # Unique ID for this evaluator instance

        # Data context manager for memory-mapped arrays
        self._data_context_manager: Optional[DataContextManager] = None
        self._using_memmap = False

        # Performance optimizer for vectorized trade tracking
        self._performance_optimizer = PerformanceOptimizerFactory.create_performance_optimizer(
            optimizer_type="genetic",
            config={
                "enable_performance_optimizations": True,
                "enable_vectorized_tracking": True,
                "enable_deduplication": True,
                "n_jobs": self.n_jobs,
            },
        )

    @staticmethod
    def _params_key(params: Dict[str, Any]) -> Tuple[str, ...]:
        """Create a deterministic, hashable key for a parameter dict."""
        # Use repr for stable representation across ints/floats/strings and sort by key
        return tuple(f"{k}={repr(params[k])}" for k in sorted(params.keys()))

    def evaluate_population(
        self,
        population: List[Dict[str, Any]],
        scenario_config: Dict[str, Any],
        data: "OptimizationData",
        backtester: "StrategyBacktester",
    ) -> List["EvaluationResult"]:
        """
        Evaluate a population of parameter sets.

        Args:
            population: A list of parameter dictionaries to evaluate.
            scenario_config: The configuration for the backtesting scenario.
            data: The market data for the backtest.
            backtester: The backtester instance.

        Returns:
            A list of EvaluationResult objects.
        """
        # Ensure vectorized trade tracking is enabled for the backtester
        if hasattr(backtester, "use_vectorized_tracking") and self._performance_optimizer:
            backtester.use_vectorized_tracking = True
            logger.debug("Enabled vectorized trade tracking for backtester")

        # Batch de-duplication: identify unique parameter sets before dispatch
        unique_params: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        param_to_unique_key: Dict[int, Tuple[str, ...]] = {}

        # Build mapping of unique parameter sets
        for i, params in enumerate(population):
            key = self._params_key(params)
            unique_params[key] = params
            param_to_unique_key[i] = key

        logger.debug(
            f"Population size: {len(population)}, unique parameter sets: {len(unique_params)}"
        )

        # Resolve cached results for unique parameter sets
        unique_results: Dict[Tuple[str, ...], EvaluationResult] = {}
        to_eval_params: List[Dict[str, Any]] = []
        to_eval_keys: List[Tuple[str, ...]] = []

        # Check cache for each unique parameter set
        for key, params in unique_params.items():
            cached = self._result_cache.get(key)
            if cached is not None:
                unique_results[key] = cached
            else:
                # Record duplicate stats via shared interface (does not prevent eval here)
                try:
                    if self._deduplicator.is_duplicate(params):
                        pass
                except Exception:
                    pass
                to_eval_params.append(params)
                to_eval_keys.append(key)

        # Evaluate only non-cached unique parameter sets
        if to_eval_params:
            if self.n_jobs > 1:
                # Create memory-mapped data context if needed and not already created
                if not self._using_memmap and self.n_jobs > 1:
                    try:
                        self._data_context_manager = DataContextManager()
                        self._data_context = self._data_context_manager.create_context(data)
                        self._using_memmap = True
                        logger.debug("Created memory-mapped data context for parallel evaluation")
                    except Exception as e:
                        logger.error(f"Failed to create memory-mapped data context: {e}")
                        self._using_memmap = False

                evaluated = self._evaluate_parallel(
                    to_eval_params, scenario_config, data, backtester
                )
            else:
                evaluated = self._evaluate_sequential(
                    to_eval_params, scenario_config, data, backtester
                )

            # Update unique results and cache
            for params, key, result in zip(to_eval_params, to_eval_keys, evaluated):
                unique_results[key] = result
                self._result_cache[key] = result

                # Update dedup cache with objective value for stats and potential reuse
                try:
                    objective_value = result.objective_value
                    if isinstance(objective_value, (int, float)):
                        self._deduplicator.add_trial(params, float(objective_value))
                    # For multi-objective, store first element as a proxy
                    elif isinstance(objective_value, list) and objective_value:
                        self._deduplicator.add_trial(params, float(objective_value[0]))
                except Exception:
                    pass

        # Map results back to original population order
        results: List[EvaluationResult] = []
        for i in range(len(population)):
            key = param_to_unique_key[i]
            results.append(unique_results[key])

        return results

    def get_dedup_stats(self) -> Dict[str, Any]:
        """Return deduplication statistics if available."""
        try:
            stats = self._deduplicator.get_stats()
            # Add local cache stats
            stats["local_cache_size"] = len(self._result_cache)
            stats["batch_dedup_enabled"] = True
            stats["params_only_dispatch"] = self._workers_initialized
            stats["using_memmap"] = self._using_memmap
            stats["vectorized_tracking_enabled"] = self._performance_optimizer is not None

            # Add adaptive batch sizing stats
            if self._enable_adaptive_batch_sizing:
                stats["adaptive_batch_sizing_enabled"] = True

            # Add GPU acceleration stats
            stats["gpu_acceleration_enabled"] = self._enable_gpu_acceleration
            stats["gpu_available"] = (
                self._gpu_manager.is_available() if self._gpu_manager else False
            )

            # Continue with batch sizing stats
            if self._enable_adaptive_batch_sizing:
                stats["current_batch_size"] = self._batch_sizer.get_current_batch_size()
                stats["last_evaluation_time_ms"] = self._last_evaluation_time
            else:
                stats["adaptive_batch_sizing_enabled"] = False

            # Add hybrid parallelism stats
            if self._enable_hybrid_parallelism and self._hybrid_manager:
                stats["hybrid_parallelism_enabled"] = True
                stats["n_processes"] = self._hybrid_manager.n_processes
                stats["threads_per_process"] = self._hybrid_manager.n_threads_per_process
            else:
                stats["hybrid_parallelism_enabled"] = False

            # Add incremental evaluation stats
            stats["incremental_evaluation_enabled"] = self._enable_incremental_evaluation
            if (
                self._enable_incremental_evaluation
                and hasattr(self.evaluator, "incremental_manager")
                and self.evaluator.incremental_manager is not None
            ):
                inc_stats = self.evaluator.incremental_manager.get_stats()
                for key, value in inc_stats.items():
                    stats[f"incremental_{key}"] = value

            return stats
        except Exception:
            return {}

    def _evaluate_sequential(
        self,
        population: List[Dict[str, Any]],
        scenario_config: Dict[str, Any],
        data: "OptimizationData",
        backtester: "StrategyBacktester",
    ) -> List["EvaluationResult"]:
        """Evaluate the population sequentially."""
        results = []
        for i, params in enumerate(population):
            logger.debug(f"Evaluating individual {i + 1}/{len(population)}")
            # Use incremental evaluation if enabled
            if self._enable_incremental_evaluation:
                result = self.evaluator.evaluate_parameters(
                    params,
                    scenario_config,
                    data,
                    backtester,
                    previous_parameters=self._previous_params,
                )
                # Store parameters for next evaluation
                self._previous_params = params
            else:
                result = self.evaluator.evaluate_parameters(
                    params, scenario_config, data, backtester
                )
            results.append(result)
        return results

    def _evaluate_parallel(
        self,
        population: List[Dict[str, Any]],
        scenario_config: Dict[str, Any],
        data: "OptimizationData",
        backtester: "StrategyBacktester",
    ) -> List["EvaluationResult"]:
        """Evaluate the population in parallel using joblib.

        Uses a worker-local context so heavy objects are constructed once per worker
        and reused across evaluations.
        """
        import time

        start_time = time.time()

        try:
            # Get parameter space from scenario config if available
            parameter_space = scenario_config.get(
                "parameter_space", scenario_config.get("optimize", {})
            )

            # Adapt batch size if enabled
            if self._enable_adaptive_batch_sizing and parameter_space:

                # Update batch size based on population and parameter space
                batch_config = self._batch_sizer.update_batch_size(
                    parameter_space=parameter_space,
                    population=population,
                    execution_time_ms=self._last_evaluation_time,
                )

                # Use adaptive batch size or fall back to provided/default
                batch_size = self._joblib_batch_size or batch_config["batch_size"]

                logger.debug(
                    f"Using adaptive batch size: {batch_size} "
                    f"(population: {len(population)}, batches: ~{batch_config['batch_count']})"
                )
            else:
                # Heuristic defaults
                batch_size = self._joblib_batch_size or "auto"

            # Pre-dispatch setting
            pre_dispatch = self._joblib_pre_dispatch or "2*n_jobs"

            # Use hybrid parallelism if enabled
            if self._enable_hybrid_parallelism and self._hybrid_manager:
                logger.debug(
                    f"Using hybrid parallelism for population of {len(population)} individuals"
                )
                return self._hybrid_manager.evaluate_population(
                    population=population,
                    scenario_config=scenario_config,
                    data=data if not self._using_memmap else self._data_context,
                    backtester=backtester,
                    evaluator=self.evaluator,
                )

            # Otherwise, use standard joblib parallelism based on worker initialization state
            if not self._workers_initialized:
                # First run: initialize workers with full context
                if self._using_memmap:
                    # Use memory-mapped data context
                    from .ga_worker_context import evaluate_with_context_memmap

                    logger.debug(
                        "First parallel run: initializing worker contexts with memory-mapped data"
                    )
                    # Add incremental evaluation support
                    if self._enable_incremental_evaluation:
                        # We need to pass previous parameters in the first generation too
                        results = Parallel(
                            n_jobs=self.n_jobs,
                            pre_dispatch=pre_dispatch,
                            batch_size=batch_size,
                            prefer="processes",
                        )(
                            delayed(evaluate_with_context_memmap)(
                                params,
                                scenario_config,
                                self._data_context,
                                backtester,
                                self.evaluator,
                                previous_parameters=self._previous_params,
                            )
                            for params in population
                        )
                        # Store last parameters for next generation
                        if population:
                            self._previous_params = population[-1]
                    else:
                        # Standard evaluation without incremental support
                        results = Parallel(
                            n_jobs=self.n_jobs,
                            pre_dispatch=pre_dispatch,
                            batch_size=batch_size,
                            prefer="processes",
                        )(
                            delayed(evaluate_with_context_memmap)(
                                params,
                                scenario_config,
                                self._data_context,
                                backtester,
                                self.evaluator,
                            )
                            for params in population
                        )
                else:
                    # Use standard data passing
                    from .ga_worker_context import evaluate_with_context

                    logger.debug("First parallel run: initializing worker contexts")
                    # Add incremental evaluation support
                    if self._enable_incremental_evaluation:
                        # We need to pass previous parameters in the first generation too
                        results = Parallel(
                            n_jobs=self.n_jobs,
                            pre_dispatch=pre_dispatch,
                            batch_size=batch_size,
                            prefer="processes",
                        )(
                            delayed(evaluate_with_context)(
                                params,
                                scenario_config,
                                data,
                                backtester,
                                self.evaluator,
                                previous_parameters=self._previous_params,
                            )
                            for params in population
                        )
                        # Store last parameters for next generation
                        if population:
                            self._previous_params = population[-1]
                    else:
                        # Standard evaluation without incremental support
                        results = Parallel(
                            n_jobs=self.n_jobs,
                            pre_dispatch=pre_dispatch,
                            batch_size=batch_size,
                            prefer="processes",
                        )(
                            delayed(evaluate_with_context)(
                                params, scenario_config, data, backtester, self.evaluator
                            )
                            for params in population
                        )

                # Mark workers as initialized for future runs
                self._workers_initialized = True
            else:
                # Subsequent runs: use params-only dispatch to initialized workers
                from .ga_worker_context import evaluate_params_only

                logger.debug("Using params-only dispatch to initialized workers")

                # Handle incremental evaluation
                if self._enable_incremental_evaluation:
                    # For incremental evaluation, we need to track the previous parameters
                    results = []
                    for params in population:
                        result = evaluate_params_only(
                            params, previous_parameters=self._previous_params
                        )
                        results.append(result)
                        # Update previous parameters for next evaluation
                        self._previous_params = params
                else:
                    # Standard evaluation without incremental support
                    results = Parallel(
                        n_jobs=self.n_jobs,
                        pre_dispatch=pre_dispatch,
                        batch_size=batch_size,
                        prefer="processes",
                    )(delayed(evaluate_params_only)(params) for params in population)

            results_list = [r for r in results if r is not None]

            # Record evaluation time for adaptive batch sizing if enabled
            if self._enable_adaptive_batch_sizing:
                import time

                self._last_evaluation_time = (time.time() - start_time) * 1000  # ms
                logger.debug(
                    f"Population evaluation completed in {self._last_evaluation_time:.2f}ms"
                )

            return results_list
        except Exception as e:
            logger.error(f"Parallel evaluation failed: {e}", exc_info=True)
            logger.warning("Falling back to sequential evaluation.")
            self._workers_initialized = False  # Reset initialization state on error
            return self._evaluate_sequential(population, scenario_config, data, backtester)

    def __del__(self):
        """Clean up resources when the evaluator is destroyed."""
        if self._data_context_manager:
            self._data_context_manager.cleanup()

        # Clean up hybrid parallelism manager
        if self._hybrid_manager:
            self._hybrid_manager.cleanup()

        # Clean up GPU acceleration manager
        if self._gpu_manager:
            self._gpu_manager.cleanup()
