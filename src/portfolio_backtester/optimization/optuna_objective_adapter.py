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
            local.evaluator = BacktestEvaluator(
                metrics_to_optimize=[self.scenario_config["optimization_metric"]],
                is_multi_objective=False,
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
        """Convert trial suggestions into a parameter dictionary understood by our stack."""
        params: Dict[str, Any] = {}
        for param_def in self.scenario_config.get("optimize", []):
            name = param_def["parameter"]
            ptype = param_def.get("type")
            # Infer type if not explicitly provided
            if ptype is None:
                if "min_value" in param_def and "max_value" in param_def:
                    if isinstance(param_def["min_value"], int) and isinstance(param_def["max_value"], int):
                        ptype = "int"
                    else:
                        ptype = "float"
                else:
                    ptype = "float"

            if ptype == "int":
                params[name] = trial.suggest_int(
                    name,
                    param_def["min_value"],
                    param_def["max_value"],
                    step=param_def.get("step", 1),
                )
            elif ptype == "float":
                params[name] = trial.suggest_float(
                    name,
                    param_def["min_value"],
                    param_def["max_value"],
                    step=param_def.get("step"),
                )
            elif ptype == "categorical":
                choices = param_def.get("choices") or param_def.get("values")
                if not choices:
                    raise ValueError(f"Categorical parameter '{name}' must have 'choices' or 'values' defined.")
                params[name] = trial.suggest_categorical(name, choices)
            else:
                raise ValueError(f"Unsupported parameter type: {ptype}")
        return params