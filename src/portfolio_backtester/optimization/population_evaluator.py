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
            result = self.evaluator.evaluate_parameters(params, scenario_config, data, backtester)
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
        try:
            # Heuristic defaults
            pre_dispatch = self._joblib_pre_dispatch or "2*n_jobs"
            batch_size = self._joblib_batch_size or "auto"

            # Use different evaluation strategies based on worker initialization state
            if not self._workers_initialized:
                # First run: initialize workers with full context
                if self._using_memmap:
                    # Use memory-mapped data context
                    from .ga_worker_context import evaluate_with_context_memmap

                    logger.debug(
                        "First parallel run: initializing worker contexts with memory-mapped data"
                    )
                    results = Parallel(
                        n_jobs=self.n_jobs,
                        pre_dispatch=pre_dispatch,
                        batch_size=batch_size,
                        prefer="processes",
                    )(
                        delayed(evaluate_with_context_memmap)(
                            params, scenario_config, self._data_context, backtester, self.evaluator
                        )
                        for params in population
                    )
                else:
                    # Use standard data passing
                    from .ga_worker_context import evaluate_with_context

                    logger.debug("First parallel run: initializing worker contexts")
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
                results = Parallel(
                    n_jobs=self.n_jobs,
                    pre_dispatch=pre_dispatch,
                    batch_size=batch_size,
                    prefer="processes",
                )(delayed(evaluate_params_only)(params) for params in population)

            return [r for r in results if r is not None]
        except Exception as e:
            logger.error(f"Parallel evaluation failed: {e}", exc_info=True)
            logger.warning("Falling back to sequential evaluation.")
            self._workers_initialized = False  # Reset initialization state on error
            return self._evaluate_sequential(population, scenario_config, data, backtester)

    def __del__(self):
        """Clean up resources when the evaluator is destroyed."""
        if self._data_context_manager:
            self._data_context_manager.cleanup()
