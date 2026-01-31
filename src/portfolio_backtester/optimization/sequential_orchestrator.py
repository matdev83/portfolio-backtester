import logging
from typing import Any, Dict, Optional, TYPE_CHECKING, Union

from .orchestrator_interfaces import OptimizationOrchestrator
from .progress_tracker import ProgressTracker
from .results import OptimizationResult

if TYPE_CHECKING:
    from .evaluator import BacktestEvaluator
    from .parameter_generator import ParameterGenerator
    from .results import OptimizationData
    from portfolio_backtester.backtester_logic.backtester_facade import (
        Backtester as BacktesterFacade,
    )
    from portfolio_backtester.canonical_config import CanonicalScenarioConfig

logger = logging.getLogger(__name__)


class SequentialOrchestrator(OptimizationOrchestrator):
    """Coordinates sequential optimization process."""

    def __init__(
        self,
        parameter_generator: "ParameterGenerator",
        evaluator: "BacktestEvaluator",
        timeout_seconds: Optional[int] = None,
        early_stop_patience: Optional[int] = None,
    ):
        self.parameter_generator = parameter_generator
        self.evaluator = evaluator

        self.progress_tracker = ProgressTracker(
            timeout_seconds=timeout_seconds,
            early_stop_patience=early_stop_patience,
            is_multi_objective=evaluator.is_multi_objective,
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("SequentialOrchestrator initialized")

    def optimize(
        self,
        scenario_config: Union[Dict[str, Any], "CanonicalScenarioConfig"],
        optimization_config: Dict[str, Any],
        data: "OptimizationData",
        backtester: "BacktesterFacade",  # Changed to match the interface
    ) -> "OptimizationResult":
        from portfolio_backtester.canonical_config import CanonicalScenarioConfig
        from portfolio_backtester.scenario_normalizer import ScenarioNormalizer

        # Ensure we are working with a canonical config internally
        if not isinstance(scenario_config, CanonicalScenarioConfig):
            logger.warning(
                "ACCIDENTAL BYPASS: Raw scenario dictionary passed to SequentialOrchestrator.optimize. "
                "All scenarios should be canonicalized at the boundary. "
                "Scenario: %s", scenario_config.get('name', 'unnamed')
            )
            normalizer = ScenarioNormalizer()
            # Assuming global_config is available via backtester or something
            global_config = getattr(backtester, "global_config", {})
            canonical_config = normalizer.normalize(scenario=scenario_config, global_config=global_config)
        else:
            canonical_config = scenario_config

        logger.info("Starting sequential optimization process")

        self.parameter_generator.initialize(canonical_config, optimization_config)

        try:
            while (
                not self.parameter_generator.is_finished()
                and not self.progress_tracker.should_stop()
            ):
                parameters = self.parameter_generator.suggest_parameters()

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Evaluating parameters: {parameters}")

                # Type cast to StrategyBacktester since the evaluator expects it
                # This is safe because BacktesterFacade is a supertype of StrategyBacktester
                # and we're only using methods that are available in StrategyBacktester
                from portfolio_backtester.backtesting.strategy_backtester import (
                    StrategyBacktester,
                )

                strategy_backtester = (
                    backtester
                    if isinstance(backtester, StrategyBacktester)
                    else getattr(backtester, "strategy_backtester", backtester)
                )

                evaluation_result = self.evaluator.evaluate_parameters(
                    parameters,
                    scenario_config,
                    data,
                    strategy_backtester,  # type: ignore[arg-type]
                )

                self.progress_tracker.update_progress(evaluation_result.objective_value)
                self.parameter_generator.report_result(parameters, evaluation_result)

            final_result = self.parameter_generator.get_best_result()

            status = self.progress_tracker.get_status()
            logger.info(
                f"Optimization completed: {status['total_evaluations']} evaluations "
                f"in {status['elapsed_seconds']:.1f}s, best value: {status['best_value']}"
            )

            return final_result

        except Exception as e:
            logger.error(f"Optimization failed: {e}", exc_info=True)
            try:
                return self.parameter_generator.get_best_result()
            except Exception:
                return OptimizationResult(
                    best_parameters={},
                    best_value=(
                        -1e9
                        if not self.evaluator.is_multi_objective
                        else [-1e9] * len(self.evaluator.metrics_to_optimize)
                    ),
                    n_evaluations=self.progress_tracker.total_evaluations,
                    optimization_history=[],
                )

    def get_progress_status(self) -> Dict[str, Any]:
        """Get current optimization progress status.

        Returns:
            Dictionary containing current optimization status
        """
        return self.progress_tracker.get_status()
