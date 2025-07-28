import logging
import os
from functools import reduce
from operator import mul

import optuna

logger = logging.getLogger(__name__)

def setup_optuna_study(backtester, scenario_config, storage, study_name: str):
    optimization_specs = scenario_config.get("optimize", [])
    param_types = [
        backtester.global_config.get("optimizer_parameter_defaults", {}).get(spec["parameter"], {}).get("type")
        for spec in optimization_specs
    ]
    is_grid_search = all(pt == "int" for pt in param_types)

    n_trials_actual = backtester.args.optuna_trials
    if is_grid_search and backtester.n_jobs == 1:
        search_space = {
            spec["parameter"]: list(range(spec["min_value"], spec["max_value"] + 1, spec.get("step", 1)))
            for spec in optimization_specs
        }
        sampler = optuna.samplers.GridSampler(search_space)
        n_trials_actual = reduce(mul, [len(v) for v in search_space.values()], 1)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Using GridSampler. Total trials: {n_trials_actual}")
    else:
        if is_grid_search and backtester.n_jobs > 1:
            logger.warning("Grid search is not supported with n_jobs > 1. Using TPESampler instead.")
        sampler = optuna.samplers.TPESampler(seed=backtester.random_state)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Using TPESampler with {n_trials_actual} trials.")

    if backtester.args.pruning_enabled:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=backtester.args.pruning_n_startup_trials,
            n_warmup_steps=backtester.args.pruning_n_warmup_steps,
            interval_steps=backtester.args.pruning_interval_steps
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("MedianPruner enabled.")
    else:
        pruner = optuna.pruners.NopPruner()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Pruning disabled (NopPruner used).")

    if backtester.args.random_seed is not None:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Deleted existing Optuna study '{study_name}' for fresh start with random seed.")
        except KeyError:
            pass
        except Exception as e:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(f"Could not delete existing Optuna study '{study_name}': {e}")

    optimization_targets_config = scenario_config.get("optimization_targets", [])
    study_directions = [t.get("direction", "maximize").lower() for t in optimization_targets_config] or ["maximize"]
    for i, d in enumerate(study_directions):
        if d not in ["maximize", "minimize"]:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(f"Invalid direction '{d}' for target. Defaulting to 'maximize'.")
            study_directions[i] = "maximize"

    study = optuna.create_study(
        study_name=study_name, storage=storage, sampler=sampler, pruner=pruner,
        directions=study_directions if len(study_directions) > 1 else None,
        direction=study_directions[0] if len(study_directions) == 1 else None,
        load_if_exists=(backtester.args.random_seed is None)
    )
    return study, n_trials_actual
