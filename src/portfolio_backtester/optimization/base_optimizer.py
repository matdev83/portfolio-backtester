"""Abstract base class for optimizers."""

import abc
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..core import Backtester


class BaseOptimizer(abc.ABC):
    """Abstract base class for all optimizers."""

    def __init__(
        self,
        scenario_config: Dict[str, Any],
        backtester_instance: "Backtester",
        global_config: Dict[str, Any],
        monthly_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        rets_full: pd.DataFrame,
        random_state: Optional[int] = None,
    ):
        """Initialize the optimizer.

        Args:
            scenario_config: Configuration for the scenario.
            backtester_instance: Instance of the backtester.
            global_config: Global configuration.
            monthly_data: Monthly data for the optimization.
            daily_data: Daily data for the optimization.
            rets_full: Full returns data.
            random_state: Random state for reproducibility.
        """
        self.scenario_config = scenario_config
        self.backtester = backtester_instance
        self.global_config = global_config
        self.monthly_data = monthly_data
        self.daily_data = daily_data
        self.rets_full = rets_full
        self.random_state = random_state

        # Extract optimization parameters specification
        self.optimization_params_spec = scenario_config.get("optimize", [])
        if not self.optimization_params_spec:
            raise ValueError("Optimizer requires 'optimize' specifications in the scenario config.")

        # Extract metrics to optimize
        self.metrics_to_optimize = [
            t["name"] for t in scenario_config.get("optimization_targets", [])
        ] or [scenario_config.get("optimization_metric", "Calmar")]
        self.is_multi_objective = len(self.metrics_to_optimize) > 1

    @abc.abstractmethod
    def optimize(self) -> Tuple[Dict[str, Any], int, Any]:
        """Run the optimization process.

        Returns:
            A tuple containing:
                - The optimal parameters found.
                - The number of evaluations/trials performed.
                - An optimizer-specific best trial/solution object (e.g., Optuna Trial).
        """
        pass

    def _prepare_search_space(self) -> Tuple[Any, Any]:
        """Prepare the search space for the optimizer.

        Returns:
            A tuple containing:
                - The search space definition (format depends on optimizer).
                - The sampler/gene type definition (format depends on optimizer).
        """
        raise NotImplementedError("_prepare_search_space must be implemented by subclasses")
