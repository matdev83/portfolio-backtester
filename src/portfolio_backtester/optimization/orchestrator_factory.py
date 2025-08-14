from typing import Any

from .evaluator import BacktestEvaluator
from .orchestrator_interfaces import OptimizationOrchestrator
from .parameter_generator import ParameterGenerator
from .population_evaluator import PopulationEvaluator
from .population_orchestrator import PopulationOrchestrator
from .sequential_orchestrator import SequentialOrchestrator


def create_orchestrator(
    optimizer_type: str,
    parameter_generator: "ParameterGenerator",
    evaluator: "BacktestEvaluator",
    **kwargs: Any,
) -> "OptimizationOrchestrator":
    """
    Create the appropriate optimization orchestrator.

    Args:
        optimizer_type: The type of optimizer to use.
        parameter_generator: The parameter generator instance.
        evaluator: The backtest evaluator instance.
        **kwargs: Additional keyword arguments for the orchestrator.

    Returns:
        An instance of an OptimizationOrchestrator.
    """
    if optimizer_type in ["genetic", "particle_swarm", "differential_evolution"]:
        population_evaluator = PopulationEvaluator(
            evaluator,
            n_jobs=kwargs.get("n_jobs", 1),
            joblib_batch_size=kwargs.get("joblib_batch_size"),
            joblib_pre_dispatch=kwargs.get("joblib_pre_dispatch"),
        )
        return PopulationOrchestrator(
            parameter_generator=parameter_generator,  # type: ignore
            population_evaluator=population_evaluator,
            timeout_seconds=kwargs.get("timeout"),
            early_stop_patience=kwargs.get("early_stop_patience"),
        )
    else:
        return SequentialOrchestrator(
            parameter_generator=parameter_generator,
            evaluator=evaluator,
            timeout_seconds=kwargs.get("timeout"),
            early_stop_patience=kwargs.get("early_stop_patience"),
        )
