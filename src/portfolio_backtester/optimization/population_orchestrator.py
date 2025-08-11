from typing import TYPE_CHECKING, Any, Dict, Optional, cast

from loguru import logger
from tqdm import tqdm

from ..backtester_logic.backtester_facade import Backtester
from ..backtesting.strategy_backtester import StrategyBacktester
from .orchestrator_interfaces import OptimizationOrchestrator
from .progress_tracker import ProgressTracker
from .results import OptimizationResult

if TYPE_CHECKING:
    from .population_evaluator import PopulationEvaluator
    from .parameter_generator import PopulationBasedParameterGenerator


class PopulationOrchestrator(OptimizationOrchestrator):
    """Coordinates population-based optimization process."""

    def __init__(
        self,
        parameter_generator: "PopulationBasedParameterGenerator",
        population_evaluator: "PopulationEvaluator",
        timeout_seconds: Optional[int] = None,
        early_stop_patience: Optional[int] = None,
    ):
        self.parameter_generator = parameter_generator
        self.population_evaluator = population_evaluator

        self.progress_tracker = ProgressTracker(
            timeout_seconds=timeout_seconds,
            early_stop_patience=early_stop_patience,
            is_multi_objective=population_evaluator.evaluator.is_multi_objective,
        )

        logger.debug("PopulationOrchestrator initialized")

    def optimize(
        self,
        scenario_config: Dict[str, Any],
        optimization_config: Dict[str, Any],
        data: Any,
        backtester: "Backtester",
    ) -> "OptimizationResult":
        """Run the population-based optimization process with a progress bar."""
        logger.info("Starting population-based optimization...")
        self.parameter_generator.initialize(scenario_config, optimization_config)
        strategy_backtester = cast(StrategyBacktester, backtester)

        # Assuming max_generations is available for GA, otherwise needs a more general approach
        ga_settings = scenario_config.get("ga_settings", {})
        max_generations = ga_settings.get(
            "max_generations",
            optimization_config.get("ga_settings", {}).get("max_generations", 100),
        )

        with tqdm(total=max_generations, desc="Genetic Optimization", unit="gen") as pbar:
            while (
                not self.parameter_generator.is_finished()
                and not self.progress_tracker.should_stop()
            ):
                population = self.parameter_generator.suggest_population()

                results = self.population_evaluator.evaluate_population(
                    population, scenario_config, data, strategy_backtester
                )

                self.parameter_generator.report_population_results(population, results)

                # Update progress
                self.progress_tracker.total_evaluations += len(population)
                best_in_gen = None
                if results:
                    objective_values = [
                        res.objective_value
                        for res in results
                        if isinstance(res.objective_value, (int, float))
                    ]
                    if objective_values:
                        best_in_gen = max(objective_values)
                        self.progress_tracker.update_progress(best_in_gen)
                        pbar.set_postfix({"best_fitness": f"{best_in_gen:.4f}"}, refresh=True)

                pbar.update(1)

        logger.info("Population-based optimization finished.")
        return self.parameter_generator.get_best_result()
