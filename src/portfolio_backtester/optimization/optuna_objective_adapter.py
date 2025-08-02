"""
Thin adapter that lets Optuna call our separated stack
without breaking separation of concerns.
"""

import logging
import optuna
import threading
from typing import Any, Dict

from .evaluator import BacktestEvaluator
from ..config_loader import GLOBAL_CONFIG
from ..backtesting.strategy_backtester import StrategyBacktester
from .results import OptimizationData

logger = logging.getLogger(__name__)


class OptunaObjectiveAdapter:
    """
    Callable that Optuna will invoke for every trial.
    It instantiates its own orchestrator/evaluator/backtester
    so that separation of concerns is preserved.
    """

    def __init__(
        self,
        scenario_config: Dict[str, Any],
        data: OptimizationData,
        n_jobs: int = 1,
        parameter_space: Dict[str, Any] = None,
    ):
        """Construct the objective adapter.

        A thread-local backtester is created on first use inside each worker
        thread so that heavy caches are built only once and reused across
        trials.  This keeps the code thread-safe while eliminating the long
        "0 % stall" seen before the first progress-bar update.
        """
        self.scenario_config = scenario_config
        self.data = data
        self.n_jobs = n_jobs
        self.parameter_space = parameter_space or {}

    def __call__(self, trial: optuna.Trial) -> float:
        logger.info("Trial %d started", trial.number)
        
        # 1. Build parameters from trial
        params = self._trial_to_params(trial)

        # 2. Get or create thread-local backtester + evaluator
        local = getattr(self, "_local", None)
        if local is None:
            self._local = threading.local()
            local = self._local
        if not hasattr(local, "backtester"):
            local.backtester = StrategyBacktester(global_config=GLOBAL_CONFIG, data_source=None)
        if not hasattr(local, "evaluator"):
            # Extract metrics to optimize from either optimization_targets or optimization_metric
            optimization_targets = self.scenario_config.get("optimization_targets", [])
            if optimization_targets:
                metrics_to_optimize = [target["name"] for target in optimization_targets]
                is_multi_objective = len(metrics_to_optimize) > 1
            else:
                metrics_to_optimize = [self.scenario_config.get("optimization_metric", "Calmar")]
                is_multi_objective = False
            
            local.evaluator = BacktestEvaluator(
                metrics_to_optimize=metrics_to_optimize,
                is_multi_objective=is_multi_objective,
                n_jobs=self.n_jobs,
                enable_parallel_optimization=True,
            )
        backtester = local.backtester
        evaluator = local.evaluator

        # 3. Evaluate
        result = evaluator.evaluate_parameters(
            parameters=params,
            scenario_config=self.scenario_config,
            data=self.data,
            backtester=backtester,
        )

        return result.objective_value

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _trial_to_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Convert trial suggestions into a parameter dictionary understood by our stack.
        
        Uses the modern parameter_space format with type, low/high bounds, and choices.
        """
        params: Dict[str, Any] = {}
        for param_name, param_config in self.parameter_space.items():
            ptype = param_config.get("type", "float")
            
            if ptype == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config.get("low", 0),
                    param_config.get("high", 100),
                    step=param_config.get("step", 1),
                )
            elif ptype == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config.get("low", 0.0),
                    param_config.get("high", 1.0),
                    step=param_config.get("step"),
                )
            elif ptype == "categorical":
                choices = param_config.get("choices", [])
                if not choices:
                    raise ValueError(f"Categorical parameter '{param_name}' must have 'choices' defined.")
                params[param_name] = trial.suggest_categorical(param_name, choices)
            else:
                raise ValueError(f"Unsupported parameter type: {ptype}")
        
        return params
