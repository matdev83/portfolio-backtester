import logging
import os
from functools import reduce
from operator import mul

import numpy as np
import optuna
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# Optimizer imports

from ..optimization.optuna_optimizer import OptunaOptimizer
from ..optimization.genetic_optimizer import GeneticOptimizer
from ..utils import INTERRUPTED as CENTRAL_INTERRUPTED_FLAG, generate_randomized_wfo_windows

# TESTING NOTE: When testing optimization functions, be aware that Mock objects
# may be passed as timeout values or other numeric parameters. The TimeoutManager
# class in core.py handles this with defensive programming using try-catch blocks
# to prevent TypeError exceptions when Mock objects are used in numeric operations.

# Global progress tracker for optimization
_global_progress_tracker = None

def get_optimizer(optimizer_type, scenario_config, backtester_instance, global_config, monthly_data, daily_data, rets_full, random_state):
    """Factory function to create optimizer instances."""
    if optimizer_type == "genetic":
        return GeneticOptimizer(
            scenario_config=scenario_config,
            backtester_instance=backtester_instance,
            global_config=global_config,
            monthly_data=monthly_data,
            daily_data=daily_data,
            rets_full=rets_full,
            random_state=random_state
        )
    elif optimizer_type == "optuna":
        return OptunaOptimizer(
            scenario_config=scenario_config,
            backtester_instance=backtester_instance,
            global_config=global_config,
            monthly_data=monthly_data,
            daily_data=daily_data,
            rets_full=rets_full,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

def run_optimization(self, scenario_config, monthly_data, daily_data, rets_full):
    global _global_progress_tracker
    
    optimizer_type = self.global_config.get("optimizer_config", {}).get("optimizer_type", "optuna")
    if self.logger.isEnabledFor(logging.DEBUG):
        self.logger.debug(
            f"Running {optimizer_type} optimization for scenario: {scenario_config['name']} with walk-forward splits."
        )

    optimizer = get_optimizer(
        optimizer_type,
        scenario_config,
        self,
        self.global_config,
        monthly_data,
        daily_data,
        rets_full,
        self.random_state
    )
    
    # Call the optimize method and handle the return values
    result = optimizer.optimize()
    
    # The optimize method returns a tuple with 2 or 3 elements
    # For compatibility, we'll return only the first two elements
    if len(result) >= 2:
        return result[0], result[1]
    else:
        raise ValueError("Optimizer.optimize() returned unexpected number of elements")


