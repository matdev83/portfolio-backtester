import os
import sys
from typing import Any, Dict, List, TYPE_CHECKING
from joblib import Parallel, delayed  # type: ignore
from loguru import logger

if TYPE_CHECKING:
    from .evaluator import BacktestEvaluator
    from .results import EvaluationResult, OptimizationData
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

    def __init__(self, evaluator: "BacktestEvaluator", n_jobs: int = 1):
        self.evaluator = evaluator
        self.n_jobs = n_jobs

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
        if self.n_jobs > 1:
            return self._evaluate_parallel(population, scenario_config, data, backtester)
        else:
            return self._evaluate_sequential(population, scenario_config, data, backtester)

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
            logger.debug(f"Evaluating individual {i+1}/{len(population)}")
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
        """Evaluate the population in parallel using joblib."""
        try:

            def wrapped_evaluate(params):
                _init_worker_logging()
                return self.evaluator.evaluate_parameters(params, scenario_config, data, backtester)

            results = Parallel(n_jobs=self.n_jobs, pre_dispatch="2*n_jobs")(
                delayed(wrapped_evaluate)(params) for params in population
            )

            return [r for r in results if r is not None]
        except Exception as e:
            logger.error(f"Parallel evaluation failed: {e}", exc_info=True)
            logger.warning("Falling back to sequential evaluation.")
            return self._evaluate_sequential(population, scenario_config, data, backtester)
