import optuna
import os
from functools import reduce
from operator import mul
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console
import numpy as np

from ..utils import INTERRUPTED as CENTRAL_INTERRUPTED_FLAG
from ..optimization.genetic_optimizer import GeneticOptimizer
from ..evaluation_logic import _evaluate_params_walk_forward

def run_optimization(self, scenario_config, monthly_data, daily_data, rets_full):
    optimizer_type = getattr(self.args, "optimizer", "optuna")
    self.logger.info(
        f"Running {optimizer_type} optimization for scenario: {scenario_config['name']} with walk-forward splits."
    )

    if optimizer_type == "genetic":
        self.logger.info("Using Genetic Algorithm Optimizer.")
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

    self.logger.info("Using Optuna Optimizer.")
    train_window_m = scenario_config.get("train_window_months", 24)
    test_window_m = scenario_config.get("test_window_months", 12)
    wf_type = scenario_config.get("walk_forward_type", "expanding").lower()

    idx = monthly_data.index
    windows = []
    start_idx = train_window_m
    while start_idx + test_window_m <= len(idx):
        train_end_idx = start_idx - 1
        test_start_idx = train_end_idx + 1
        test_end_idx = test_start_idx + test_window_m - 1
        if test_end_idx >= len(idx):
            break
        if wf_type == "rolling":
            train_start_idx = train_end_idx - train_window_m + 1
        else:
            train_start_idx = 0
        windows.append(
            (
                idx[train_start_idx],
                idx[train_end_idx],
                idx[test_start_idx],
                idx[test_end_idx],
            )
        )
        start_idx += test_window_m

    if not windows:
        raise ValueError("Not enough data for the requested walk-forward windows.")

    self.logger.info(f"Generated {len(windows)} walk-forward windows using '{wf_type}' splits.")

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
        return _evaluate_params_walk_forward(
            self, trial, trial_scenario_config, windows, monthly_data, daily_data, rets_full,
            metrics_to_optimize, is_multi_objective
        )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=Console()
    ) as progress:
        task = progress.add_task("[cyan]Optimizing...", total=n_trials)
        
        zero_streak = 0

        def callback(study, trial):
            nonlocal zero_streak
            progress.update(task, advance=1)
            if trial.user_attrs.get("zero_returns"):
                zero_streak += 1
            else:
                zero_streak = 0

            if CENTRAL_INTERRUPTED_FLAG:
                self.logger.warning("Optuna optimization interrupted by user via central flag.")
                study.stop()
                return

            if zero_streak > self.early_stop_patience:
                self.logger.info(f"Early stopping Optuna study due to {self.early_stop_patience} consecutive zero-return trials.")
                study.stop()

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=self.args.optuna_timeout_sec,
            n_jobs=self.n_jobs,
            callbacks=[callback]
        )
    
    optimal_params = scenario_config["strategy_params"].copy()
    if len(study.directions) > 1:
        best_trials = study.best_trials
        if not best_trials:
            self.logger.error("Multi-objective optimization finished without finding any best trials.")
            return optimal_params, len(study.trials)

        chosen_best_trial = best_trials[0]
        optimal_params.update(chosen_best_trial.params)
        self.logger.info(f"Best parameters (from Pareto front, first trial) found on training set: {chosen_best_trial.params}")
        print(f"Optuna Optimizer - Best parameters found: {chosen_best_trial.params}")
        actual_trial_number_for_dsr = chosen_best_trial.number
    else:
        if not study.best_trial:
             self.logger.error("Single-objective optimization finished without finding a best trial.")
             return optimal_params, len(study.trials)

        optimal_params.update(study.best_trial.params)
        self.logger.info(f"Best parameters found on training set: {study.best_trial.params}")
        print(f"Optuna Optimizer - Best parameters found: {study.best_trial.params}")
        actual_trial_number_for_dsr = study.best_trial.number
    
    return optimal_params, actual_trial_number_for_dsr

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
        self.logger.info(f"Using GridSampler with search space: {search_space}. Total trials: {n_trials_actual}")
    else:
        if is_grid_search and self.n_jobs > 1:
            self.logger.warning("Grid search is not supported with n_jobs > 1. Using TPESampler instead.")
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        self.logger.info(f"Using TPESampler with {n_trials_actual} trials.")

    if self.args.pruning_enabled:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=self.args.pruning_n_startup_trials,
            n_warmup_steps=self.args.pruning_n_warmup_steps,
            interval_steps=self.args.pruning_interval_steps
        )
        self.logger.info(f"MedianPruner enabled with n_startup_trials={self.args.pruning_n_startup_trials}, n_warmup_steps={self.args.pruning_n_warmup_steps}, interval_steps={self.args.pruning_interval_steps}.")
    else:
        pruner = optuna.pruners.NopPruner()
        self.logger.info("Pruning disabled (NopPruner used).")

    if self.args.random_seed is not None:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
            self.logger.info(f"Deleted existing Optuna study '{study_name}' for fresh start with random seed.")
        except KeyError:
            pass
        except Exception as e:
            self.logger.warning(f"Could not delete existing Optuna study '{study_name}': {e}")

    optimization_targets_config = scenario_config.get("optimization_targets", [])
    study_directions = [t.get("direction", "maximize").lower() for t in optimization_targets_config] or ["maximize"]
    for i, d in enumerate(study_directions):
        if d not in ["maximize", "minimize"]:
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
                self.logger.warning(f"Categorical parameter '{pname}' has no choices defined or choices are invalid. Skipping suggestion.")
                continue
            params[pname] = trial.suggest_categorical(pname, choices)
        else:
            self.logger.warning(f"Unsupported parameter type '{ptype}' for {pname}. Skipping suggestion.")
    return params
