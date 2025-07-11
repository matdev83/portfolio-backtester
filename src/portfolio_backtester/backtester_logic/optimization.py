import optuna
import os
import random
import numpy as np
from functools import reduce
from operator import mul
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console
import pandas as pd
import logging

from ..utils import INTERRUPTED as CENTRAL_INTERRUPTED_FLAG, generate_randomized_wfo_windows
from ..optimization.genetic_optimizer import GeneticOptimizer
from ..parallel_wfo import create_parallel_wfo_processor
# _evaluate_params_walk_forward is now a method of the Backtester class

# Global progress tracker for window-level updates
_global_progress_tracker = None



def run_optimization(self, scenario_config, monthly_data, daily_data, rets_full):
    global _global_progress_tracker
    
    optimizer_type = getattr(self.args, "optimizer", "optuna")
    if self.logger.isEnabledFor(logging.DEBUG):
        self.logger.debug(
            f"Running {optimizer_type} optimization for scenario: {scenario_config['name']} with walk-forward splits."
        )

    if optimizer_type == "genetic":
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("Using Genetic Algorithm Optimizer.")
        ga_optimizer = GeneticOptimizer(
            scenario_config=scenario_config,
            backtester_instance=self,
            global_config=self.global_config,
            monthly_data=monthly_data,
            daily_data=daily_data,
            rets_full=rets_full,
            random_state=self.random_state
        )
        optimal_params, num_evaluations = ga_optimizer.run()
        return optimal_params, num_evaluations

    if self.logger.isEnabledFor(logging.DEBUG):
        self.logger.debug("Using Optuna Optimizer.")
    
    # Generate randomized WFO windows
    windows = generate_randomized_wfo_windows(
        monthly_data.index, 
        scenario_config, 
        self.global_config, 
        self.random_state
    )

    if not windows:
        raise ValueError("Not enough data for the requested walk-forward windows.")

    # Log randomization details
    robustness_config = self.global_config.get("wfo_robustness_config", {})
    if robustness_config.get("enable_window_randomization", False) or robustness_config.get("enable_start_date_randomization", False):
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Generated {len(windows)} randomized walk-forward windows for robustness testing.")
    else:
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Generated {len(windows)} standard walk-forward windows.")

    if self.args.storage_url:
        storage = self.args.storage_url
    else:
        journal_dir = "optuna_journal"
        os.makedirs(journal_dir, exist_ok=True)
        db_path = os.path.join(journal_dir, f"{self.args.study_name or scenario_config['name']}.log")
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
        storage = JournalStorage(JournalFileBackend(file_path=db_path, lock_obj=JournalFileOpenLock(db_path)))

    study_name_base = f"{scenario_config['name']}_walk_forward"
    if self.args.study_name:
        study_name_base = f"{self.args.study_name}_{study_name_base}"
    
    study, n_trials = _setup_optuna_study(self, scenario_config, storage, study_name_base)

    metrics_to_optimize = [t["name"] for t in scenario_config.get("optimization_targets", [])] or \
                          [scenario_config.get("optimization_metric", "Calmar")]
    is_multi_objective = len(metrics_to_optimize) > 1

    def objective(trial: optuna.trial.Trial):
        current_params = _suggest_optuna_params(self, trial, scenario_config["strategy_params"], scenario_config.get("optimize", []))
        trial_scenario_config = scenario_config.copy()
        trial_scenario_config["strategy_params"] = current_params
        objective_value, full_pnl_returns = self._evaluate_params_walk_forward(
            trial, trial_scenario_config, windows, monthly_data, daily_data, rets_full,
            metrics_to_optimize, is_multi_objective
        )
        # Store full_pnl_returns in user_attrs for later retrieval
        # Convert Timestamp index to string for JSON serialization compatibility
        trial.set_user_attr("full_pnl_returns", full_pnl_returns.rename(index=str).to_dict())
        return objective_value

    # Calculate total work units: trials × windows per trial
    total_work_units = n_trials * len(windows)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=Console()
    ) as progress:
        # Track progress at the window level, not trial level
        task = progress.add_task(f"[cyan]Optimizing ({len(windows)} windows/trial)...", total=total_work_units)
        
        # Set global progress tracker for use in evaluation_logic
        _global_progress_tracker = {
            'progress': progress,
            'task': task,
            'windows_per_trial': len(windows),
            'current_trial': 0,
            'total_trials': n_trials
        }
        
        zero_streak = 0
        completed_trials = 0

        def callback(study, trial):
            nonlocal zero_streak, completed_trials
            
            completed_trials += 1
            
            # Update progress description to show current trial
            if _global_progress_tracker:
                _global_progress_tracker['current_trial'] = completed_trials
                progress.update(
                    task, 
                    description=f"[cyan]Trial {completed_trials}/{n_trials} complete ({len(windows)} windows/trial)..."
                )
            
            if trial.user_attrs.get("zero_returns"):
                zero_streak += 1
            else:
                zero_streak = 0

            if CENTRAL_INTERRUPTED_FLAG:
                self.logger.warning("Optuna optimization interrupted by user via central flag.")
                study.stop()
                return

            if zero_streak > self.early_stop_patience:
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"Early stopping Optuna study due to {self.early_stop_patience} consecutive zero-return trials.")
                study.stop()

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=self.args.optuna_timeout_sec,
            n_jobs=self.n_jobs,
            callbacks=[callback]
        )
        
        # Clear global progress tracker
        _global_progress_tracker = None
    
    optimal_params = scenario_config["strategy_params"].copy()
    best_trial_obj = None # Initialize best_trial_obj
    actual_trial_number_for_dsr = 0 # Initialize to a default value
    if len(study.directions) > 1:
        best_trials = study.best_trials
        if not best_trials:
            self.logger.error("Multi-objective optimization finished without finding any best trials.")
            return optimal_params, len(study.trials), None

        best_trial_obj = best_trials[0]
        optimal_params.update(best_trial_obj.params)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Best parameters (from Pareto front, first trial) found on training set: {best_trial_obj.params}")
            self.logger.debug(f"Optuna Optimizer - Best parameters found: {best_trial_obj.params}")
        actual_trial_number_for_dsr = best_trial_obj.number
    else:
        if not study.best_trial:
             self.logger.error("Single-objective optimization finished without finding a best trial.")
             return optimal_params, len(study.trials)

        best_trial_obj = study.best_trial
        optimal_params.update(best_trial_obj.params)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Best parameters found on training set: {best_trial_obj.params}")
            self.logger.debug(f"Optuna Optimizer - Best parameters found: {best_trial_obj.params}")
        actual_trial_number_for_dsr = study.best_trial.number
    
    # Store the study object in the best_trial_obj for later use in plotting
    if best_trial_obj is not None:
        best_trial_obj.study = study
    
    return optimal_params, actual_trial_number_for_dsr, best_trial_obj

def _setup_optuna_study(self, scenario_config, storage, study_name_base: str):
    study_name = f"{study_name_base}_seed_{self.random_state}" if self.args.random_seed is not None else study_name_base

    optimization_specs = scenario_config.get("optimize", [])
    param_types = [
        self.global_config.get("optimizer_parameter_defaults", {}).get(spec["parameter"], {}).get("type")
        for spec in optimization_specs
    ]
    is_grid_search = all(pt == "int" for pt in param_types)

    n_trials_actual = self.args.optuna_trials
    if is_grid_search and self.n_jobs == 1:
        search_space = {
            spec["parameter"]: list(range(spec["min_value"], spec["max_value"] + 1, spec.get("step", 1)))
            for spec in optimization_specs
        }
        sampler = optuna.samplers.GridSampler(search_space)
        n_trials_actual = reduce(mul, [len(v) for v in search_space.values()], 1)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Using GridSampler. Total trials: {n_trials_actual}")
    else:
        if is_grid_search and self.n_jobs > 1:
            self.logger.warning("Grid search is not supported with n_jobs > 1. Using TPESampler instead.")
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Using TPESampler with {n_trials_actual} trials.")

    if self.args.pruning_enabled:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=self.args.pruning_n_startup_trials,
            n_warmup_steps=self.args.pruning_n_warmup_steps,
            interval_steps=self.args.pruning_interval_steps
        )
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("MedianPruner enabled.")
    else:
        pruner = optuna.pruners.NopPruner()
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("Pruning disabled (NopPruner used).")

    if self.args.random_seed is not None:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Deleted existing Optuna study '{study_name}' for fresh start with random seed.")
        except KeyError:
            pass
        except Exception as e:
            if self.logger.isEnabledFor(logging.WARNING):
                self.logger.warning(f"Could not delete existing Optuna study '{study_name}': {e}")

    optimization_targets_config = scenario_config.get("optimization_targets", [])
    study_directions = [t.get("direction", "maximize").lower() for t in optimization_targets_config] or ["maximize"]
    for i, d in enumerate(study_directions):
        if d not in ["maximize", "minimize"]:
            if self.logger.isEnabledFor(logging.WARNING):
                self.logger.warning(f"Invalid direction '{d}' for target. Defaulting to 'maximize'.")
            study_directions[i] = "maximize"

    study = optuna.create_study(
        study_name=study_name, storage=storage, sampler=sampler, pruner=pruner,
        directions=study_directions if len(study_directions) > 1 else None,
        direction=study_directions[0] if len(study_directions) == 1 else None,
        load_if_exists=(self.args.random_seed is None)
    )
    return study, n_trials_actual

def _suggest_optuna_params(self, trial: optuna.trial.Trial, base_params: dict, opt_specs: list):
    params = base_params.copy()
    for spec in opt_specs:
        pname = spec["parameter"]
        opt_def = self.global_config.get("optimizer_parameter_defaults", {}).get(pname, {})
        ptype = opt_def.get("type", spec.get("type"))

        low = spec.get("min_value", opt_def.get("low"))
        high = spec.get("max_value", opt_def.get("high"))
        step = spec.get("step", opt_def.get("step", 1 if ptype == "int" else None))
        log = spec.get("log", opt_def.get("log", False))

        if ptype == "int":
            params[pname] = trial.suggest_int(pname, int(low), int(high), step=int(step) if step else 1)
        elif ptype == "float":
            params[pname] = trial.suggest_float(pname, float(low), float(high), step=float(step) if step else None, log=log)
        elif ptype == "categorical":
            choices = spec.get("values", opt_def.get("values"))
            if not choices or not isinstance(choices, list) or len(choices) == 0:
                if self.logger.isEnabledFor(logging.WARNING):
                    self.logger.warning(f"Categorical parameter '{pname}' has no choices defined or choices are invalid. Skipping suggestion.")
                continue
            params[pname] = trial.suggest_categorical(pname, choices)
        else:
            if self.logger.isEnabledFor(logging.WARNING):
                self.logger.warning(f"Unsupported parameter type '{ptype}' for {pname}. Skipping suggestion.")
    return params
