"""Optuna optimizer implementation."""

import logging
import os
from functools import reduce
from operator import mul
from typing import Any, Dict, Tuple

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

from .base_optimizer import BaseOptimizer
from .trial_evaluator import TrialEvaluator
from ..utils import INTERRUPTED as CENTRAL_INTERRUPTED_FLAG, generate_randomized_wfo_windows

logger = logging.getLogger(__name__)


class OptunaOptimizer(BaseOptimizer):
    """Optuna optimizer implementation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.study = None
        self.n_trials = 0

    def _prepare_search_space(self) -> Tuple[Dict[str, Any], Any]:
        """Prepare the search space for Optuna.

        Returns:
            A tuple containing:
                - The search space definition for GridSampler.
                - The sampler instance.
        """
        optimization_specs = self.scenario_config.get("optimize", [])
        use_grid_sampler = self.scenario_config.get("use_grid_sampler", False)
        self.n_trials = self.backtester.args.optuna_trials

        if use_grid_sampler:
            search_space = {}
            for spec in optimization_specs:
                param_name = spec["parameter"]
                param_type = self.global_config.get("optimizer_parameter_defaults", {}).get(param_name, {}).get("type")
                if not param_type:
                    param_type = spec.get("type")

                if param_type == "int":
                    search_space[param_name] = list(range(spec["min_value"], spec["max_value"] + 1, spec.get("step", 1)))
                elif param_type == "float":
                    search_space[param_name] = list(
                        np.arange(spec["min_value"], spec["max_value"] + spec.get("step", 0.1), spec.get("step", 0.1))
                    )
                elif param_type == "categorical":
                    search_space[param_name] = spec["values"]

            sampler = optuna.samplers.GridSampler(search_space)
            self.n_trials = reduce(mul, [len(v) for v in search_space.values()], 1)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Using GridSampler. Total trials: {self.n_trials}")
            return search_space, sampler
        else:
            sampler = optuna.samplers.TPESampler(seed=self.random_state)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Using TPESampler with {self.n_trials} trials.")
            return {}, sampler

    def optimize(self) -> Tuple[Dict[str, Any], int, Any]:
        """Run the Optuna optimization process.

        Returns:
            A tuple containing:
                - The optimal parameters found.
                - The number of evaluations/trials performed.
                - The best trial object.
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Using Optuna Optimizer.")

        # Log Numba status and relevant configuration
        wfo_robustness_config = self.global_config.get("wfo_robustness_config", {})
        monte_carlo_config = self.global_config.get("monte_carlo_config", {})

        enable_window_randomization = wfo_robustness_config.get("enable_window_randomization", False)
        enable_start_date_randomization = wfo_robustness_config.get("enable_start_date_randomization", False)

        if "enable_monte_carlo_during_optimization" in self.scenario_config:
            enable_monte_carlo = bool(self.scenario_config["enable_monte_carlo_during_optimization"])
        else:
            enable_monte_carlo = monte_carlo_config.get("enable_during_optimization", True)

        numba_enabled = not (enable_window_randomization or enable_start_date_randomization or enable_monte_carlo)

        logger.info(f"Optimization starting for scenario: {self.scenario_config['name']}")
        logger.info(f"  - Numba acceleration: {'ENABLED' if numba_enabled else 'DISABLED'}")
        logger.info(f"  - Window randomization: {enable_window_randomization}")
        logger.info(f"  - Start date randomization: {enable_start_date_randomization}")
        logger.info(f"  - Monte Carlo during optimization: {enable_monte_carlo}")

        windows = generate_randomized_wfo_windows(
            self.monthly_data.index,
            self.scenario_config,
            self.global_config,
            self.random_state
        )

        if not windows:
            raise ValueError("Not enough data for the requested walk-forward windows.")

        robustness_config = self.global_config.get("wfo_robustness_config", {})
        if robustness_config.get("enable_window_randomization", False) or robustness_config.get("enable_start_date_randomization", False):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Generated {len(windows)} randomized walk-forward windows for robustness testing.")
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Generated {len(windows)} standard walk-forward windows.")

        study_name_base = f"{self.scenario_config['name']}_walk_forward"
        if self.backtester.args.study_name:
            study_name_base = f"{self.backtester.args.study_name}_{study_name_base}"

        study_name = f"{study_name_base}_seed_{self.random_state}" if self.backtester.args.random_seed is not None else study_name_base

        if self.backtester.args.storage_url:
            storage = self.backtester.args.storage_url
        else:
            storage = f"sqlite:///{study_name}.db"

        # Prepare search space and get sampler
        _, sampler = self._prepare_search_space()

        if self.backtester.args.pruning_enabled:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=self.backtester.args.pruning_n_startup_trials,
                n_warmup_steps=self.backtester.args.pruning_n_warmup_steps,
                interval_steps=self.backtester.args.pruning_interval_steps
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("MedianPruner enabled.")
        else:
            pruner = optuna.pruners.NopPruner()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Pruning disabled (NopPruner used).")

        if self.backtester.args.random_seed is not None:
            try:
                optuna.delete_study(study_name=study_name, storage=storage)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Deleted existing Optuna study '{study_name}' for fresh start with random seed.")
            except KeyError:
                pass
            except Exception as e:
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(f"Could not delete existing Optuna study '{study_name}': {e}")

        optimization_targets_config = self.scenario_config.get("optimization_targets", [])
        study_directions = [t.get("direction", "maximize").lower() for t in optimization_targets_config] or ["maximize"]
        for i, d in enumerate(study_directions):
            if d not in ["maximize", "minimize"]:
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(f"Invalid direction '{d}' for target. Defaulting to 'maximize'.")
                study_directions[i] = "maximize"

        self.study = optuna.create_study(
            study_name=study_name, storage=storage, sampler=sampler, pruner=pruner,
            directions=study_directions if len(study_directions) > 1 else None,
            direction=study_directions[0] if len(study_directions) == 1 else None,
            load_if_exists=(self.backtester.args.random_seed is None)
        )

        metrics_to_optimize = [t["name"] for t in self.scenario_config.get("optimization_targets", [])] or \
                              [self.scenario_config.get("optimization_metric", "Calmar")]
        is_multi_objective = len(metrics_to_optimize) > 1

        evaluator = TrialEvaluator(self.backtester, self.scenario_config, self.monthly_data, self.daily_data, self.rets_full, metrics_to_optimize, is_multi_objective, windows)

        def objective(trial: optuna.trial.Trial):
            # Pass trial parameters as a dictionary to the evaluator
            params = {p["parameter"]: trial.suggest_float(p["parameter"], p["min_value"], p["max_value"]) if p["type"] == "float" else \
                      trial.suggest_int(p["parameter"], p["min_value"], p["max_value"]) if p["type"] == "int" else \
                      trial.suggest_categorical(p["parameter"], p["values"])
                      for p in self.optimization_params_spec}
            
            values = evaluator.evaluate(params)

            if self.study and self.study.directions and len(self.study.directions) > 1:
                processed_values = []
                # Ensure values is iterable
                if not isinstance(values, (list, tuple)):
                    values = [values]
                for i, v in enumerate(values):
                    if v is None or (isinstance(v, float) and not np.isfinite(v)):
                        if self.study.directions[i] == optuna.study.StudyDirection.MAXIMIZE:
                            processed_values.append(-1e9)
                        else:
                            processed_values.append(1e9)
                    else:
                        processed_values.append(v)
                return tuple(processed_values)

            value = values
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                if self.study and self.study.direction == optuna.study.StudyDirection.MAXIMIZE:
                    return -1e9
                else:
                    return 1e9

            return value

        total_work_units = self.n_trials * len(windows)

        _global_progress_tracker = None # Define locally for this method

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=Console()
        ) as progress:
            task = progress.add_task(f"[cyan]Optimizing ({len(windows)} windows/trial)...", total=total_work_units)

            _global_progress_tracker = {
                'progress': progress,
                'task': task,
                'windows_per_trial': len(windows),
                'current_trial': 0,
                'total_trials': self.n_trials
            }

            zero_streak = 0
            trial_count = 0

            def callback(study, trial):
                nonlocal zero_streak, trial_count
                trial_count += 1

                if _global_progress_tracker:
                    _global_progress_tracker['current_trial'] = trial_count
                    progress.update(
                        task,
                        description=f"[cyan]Trial {trial_count}/{self.n_trials} complete ({len(windows)} windows/trial)..."
                    )

                if trial.user_attrs.get("zero_returns"):
                    zero_streak += 1
                else:
                    zero_streak = 0

                if CENTRAL_INTERRUPTED_FLAG:
                    logger.warning("Optuna optimization interrupted by user via central flag.")
                    study.stop()
                    return

                if zero_streak > self.backtester.early_stop_patience:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Early stopping Optuna study due to {self.backtester.early_stop_patience} consecutive zero-return trials.")
                    study.stop()

            self.study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.backtester.args.optuna_timeout_sec,
                callbacks=[callback]
            )

            _global_progress_tracker = None

        optimal_params = self.scenario_config["strategy_params"].copy()
        best_trial_obj = None
        actual_trial_number_for_dsr = 0

        study_directions = getattr(self.study, 'directions', [])
        study_trials = getattr(self.study, 'trials', [])

        try:
            directions_length = len(study_directions) if hasattr(study_directions, '__len__') else 0
        except (TypeError, AttributeError):
            directions_length = 0

        if directions_length > 1:
            best_trials = getattr(self.study, 'best_trials', [])
            if not best_trials:
                logger.error("Multi-objective optimization finished without finding any best trials.")
                try:
                    trials_count = len(study_trials) if hasattr(study_trials, '__len__') else 0
                except (TypeError, AttributeError):
                    trials_count = 0
                return optimal_params, trials_count, None

            best_trial_obj = best_trials[0]
            optimal_params.update(best_trial_obj.params)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Best parameters (from Pareto front, first trial) found on training set: {best_trial_obj.params}")
                logger.debug(f"Optuna Optimizer - Best parameters found: {best_trial_obj.params}")
            actual_trial_number_for_dsr = best_trial_obj.number
        else:
            completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if not completed_trials:
                logger.error("Optimization finished, but no trials were completed successfully.")
                return optimal_params, 0, None

            best_trial = self.study.best_trial
            if not best_trial:
                logger.error("Single-objective optimization finished without finding a best trial.")
                return optimal_params, len(completed_trials), None

            best_trial_obj = best_trial
            optimal_params.update(best_trial_obj.params)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Best parameters found on training set: {best_trial_obj.params}")
                logger.debug(f"Optuna Optimizer - Best parameters found: {best_trial_obj.params}")
            actual_trial_number_for_dsr = best_trial.number

        return optimal_params, actual_trial_number_for_dsr, best_trial_obj