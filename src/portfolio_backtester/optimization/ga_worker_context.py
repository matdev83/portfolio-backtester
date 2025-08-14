"""
Worker-local context for GA population evaluation.

This module keeps process-local singletons for heavy objects so that repeated
task executions in the same joblib worker process can reuse them.
"""

from typing import Any, Optional, TYPE_CHECKING, Dict, List
from loguru import logger

if TYPE_CHECKING:
    from .evaluator import BacktestEvaluator
    from .results import OptimizationData, EvaluationResult, OptimizationDataContext
    from portfolio_backtester.backtesting.strategy_backtester import StrategyBacktester


# Process-local singletons (per worker process)
_initialized: bool = False
_scenario_config: Optional[Dict[str, Any]] = None
_data: Optional["OptimizationData"] = None
_backtester: Optional["StrategyBacktester"] = None
_evaluator: Optional["BacktestEvaluator"] = None
_worker_id: Optional[int] = None


def _init_worker_logging() -> None:
    """Silence the logger in worker processes to reduce overhead."""
    try:
        # We avoid altering global logging configuration too aggressively; loguru used at call site
        logger.remove()
        logger.add(lambda _: None)
    except Exception:
        pass


def ensure_initialized(
    scenario_config: Dict[str, Any],
    data: "OptimizationData",
    backtester: "StrategyBacktester",
    evaluator: "BacktestEvaluator",
    worker_id: Optional[int] = None,
) -> None:
    """Initialize worker-local singletons if not set."""
    global _initialized, _scenario_config, _data, _backtester, _evaluator, _worker_id

    if _initialized:
        return

    _init_worker_logging()
    _scenario_config = scenario_config
    _data = data
    _backtester = backtester
    _evaluator = evaluator
    _worker_id = worker_id
    _initialized = True


def evaluate_with_context(
    params: Dict[str, Any],
    scenario_config: Dict[str, Any],
    data: "OptimizationData",
    backtester: "StrategyBacktester",
    evaluator: "BacktestEvaluator",
) -> "EvaluationResult":
    """Top-level function for joblib: lazily init context and evaluate params."""
    ensure_initialized(scenario_config, data, backtester, evaluator)

    assert (
        _evaluator is not None
        and _backtester is not None
        and _data is not None
        and _scenario_config is not None
    )

    # Delegate to shared evaluator with worker-local heavy objects
    return _evaluator.evaluate_parameters(params, _scenario_config, _data, _backtester)


def evaluate_with_context_memmap(
    params: Dict[str, Any],
    scenario_config: Dict[str, Any],
    data_context: "OptimizationDataContext",
    backtester: "StrategyBacktester",
    evaluator: "BacktestEvaluator",
) -> "EvaluationResult":
    """Initialize context from memory-mapped data and evaluate params."""
    global _initialized, _scenario_config, _data, _backtester, _evaluator

    if not _initialized:
        _init_worker_logging()

        # Reconstruct data from memory-mapped files
        try:
            from .data_context import reconstruct_optimization_data

            reconstructed_data = reconstruct_optimization_data(data_context)

            _scenario_config = scenario_config
            _data = reconstructed_data
            _backtester = backtester
            _evaluator = evaluator
            _initialized = True

            logger.debug("Worker initialized with memory-mapped data")
        except Exception as e:
            logger.error(f"Failed to initialize worker with memory-mapped data: {e}")
            raise

    assert (
        _evaluator is not None
        and _backtester is not None
        and _data is not None
        and _scenario_config is not None
    )

    # Delegate to shared evaluator with worker-local heavy objects
    return _evaluator.evaluate_parameters(params, _scenario_config, _data, _backtester)


def evaluate_params_only(params: Dict[str, Any]) -> "EvaluationResult":
    """Evaluate parameters using the already initialized worker context.

    This function should only be called after the worker has been initialized
    with a prior call to evaluate_with_context or evaluate_with_context_memmap.
    """
    assert _initialized, "Worker context not initialized. Call evaluate_with_context first."
    assert (
        _evaluator is not None
        and _backtester is not None
        and _data is not None
        and _scenario_config is not None
    )

    # Use worker-local objects that were already initialized
    return _evaluator.evaluate_parameters(params, _scenario_config, _data, _backtester)


def batch_evaluate_params_only(params_list: List[Dict[str, Any]]) -> List["EvaluationResult"]:
    """Evaluate multiple parameter sets using the already initialized worker context.

    This function should only be called after the worker has been initialized
    with a prior call to evaluate_with_context or evaluate_with_context_memmap.
    """
    assert _initialized, "Worker context not initialized. Call evaluate_with_context first."
    assert (
        _evaluator is not None
        and _backtester is not None
        and _data is not None
        and _scenario_config is not None
    )

    # Process multiple parameter sets in a batch within the same worker
    results = []
    for params in params_list:
        result = _evaluator.evaluate_parameters(params, _scenario_config, _data, _backtester)
        results.append(result)

    return results


def is_initialized() -> bool:
    """Return whether the worker context has been initialized."""
    return _initialized


def get_worker_id() -> Optional[int]:
    """Return the worker ID if set."""
    return _worker_id


def reset_context() -> None:
    """Reset worker-local context (useful for tests)."""
    global _initialized, _scenario_config, _data, _backtester, _evaluator, _worker_id
    _initialized = False
    _scenario_config = None
    _data = None
    _backtester = None
    _evaluator = None
    _worker_id = None
