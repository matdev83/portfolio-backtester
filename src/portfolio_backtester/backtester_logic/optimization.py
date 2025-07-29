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

from ..optimization.optuna_setup import setup_optuna_study
from ..optimization.trial_evaluator import TrialEvaluator
from ..optimization.genetic_optimizer import GeneticOptimizer # Added import
from ..utils import INTERRUPTED as CENTRAL_INTERRUPTED_FLAG, generate_randomized_wfo_windows

# TESTING NOTE: When testing optimization functions, be aware that Mock objects
# may be passed as timeout values or other numeric parameters. The TimeoutManager
# class in core.py handles this with defensive programming using try-catch blocks
# to prevent TypeError exceptions when Mock objects are used in numeric operations.

# Global progress tracker for optimization
_global_progress_tracker = None

def run_optimization(self, scenario_config, monthly_data, daily_data, rets_full):
    global _global_progress_tracker
    
    optimizer_type = self.global_config.get("optimizer_config", {}).get("optimizer_type", "optuna")
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
        ga_result = ga_optimizer.run()
        if isinstance(ga_result, tuple):
            if len(ga_result) == 3:
                optimal_params, num_evaluations, best_trial_obj = ga_result
            elif len(ga_result) == 2:
                optimal_params, num_evaluations = ga_result
                best_trial_obj = None
            else:
                raise ValueError("GeneticOptimizer.run() returned unexpected number of elements")
        else:
            raise ValueError("GeneticOptimizer.run() did not return a tuple")
        return optimal_params, num_evaluations

    if self.logger.isEnabledFor(logging.DEBUG):
        self.logger.debug("Using Optuna Optimizer.")
    
    # Log Numba status and relevant configuration
    wfo_robustness_config = self.global_config.get("wfo_robustness_config", {})
    monte_carlo_config = self.global_config.get("monte_carlo_config", {})

    enable_window_randomization = wfo_robustness_config.get("enable_window_randomization", False)
    enable_start_date_randomization = wfo_robustness_config.get("enable_start_date_randomization", False)

    # Scenario-level override: enable_monte_carlo_during_optimization can be true/false.
    if "enable_monte_carlo_during_optimization" in scenario_config:
        enable_monte_carlo = bool(scenario_config["enable_monte_carlo_during_optimization"])
    else:
        # Fallback to global config, default True
        enable_monte_carlo = monte_carlo_config.get("enable_during_optimization", True)
    
    numba_enabled = not (enable_window_randomization or enable_start_date_randomization or enable_monte_carlo)
    
    self.logger.info(f"Optimization starting for scenario: {scenario_config['name']}")
    self.logger.info(f"  - Numba acceleration: {'ENABLED' if numba_enabled else 'DISABLED'}")
    self.logger.info(f"  - Window randomization: {enable_window_randomization}")
    self.logger.info(f"  - Start date randomization: {enable_start_date_randomization}")
    self.logger.info(f"  - Monte Carlo during optimization: {enable_monte_carlo}")
    
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

    study_name_base = f"{scenario_config['name']}_walk_forward"
    if self.args.study_name:
        study_name_base = f"{self.args.study_name}_{study_name_base}"
    
    study_name = f"{study_name_base}_seed_{self.random_state}" if self.args.random_seed is not None else study_name_base

    if self.args.storage_url:
        storage = self.args.storage_url
    else:
         storage = f"sqlite:///{study_name}.db"    
    study, n_trials = setup_optuna_study(self, scenario_config, storage, study_name)

    metrics_to_optimize = [t["name"] for t in scenario_config.get("optimization_targets", [])] or \
                          [scenario_config.get("optimization_metric", "Calmar")]
    is_multi_objective = len(metrics_to_optimize) > 1

    evaluator = TrialEvaluator(self, scenario_config, monthly_data, daily_data, rets_full, metrics_to_optimize, is_multi_objective, windows)

    def objective(trial: optuna.trial.Trial):
        values = evaluator.evaluate(trial)

        # Check for multi-objective case first
        if study.directions and len(study.directions) > 1:
            processed_values = []
            for i, v in enumerate(values):
                if v is None or (isinstance(v, float) and not np.isfinite(v)):
                    if study.directions[i] == optuna.study.StudyDirection.MAXIMIZE:
                        processed_values.append(-1e9)
                    else:
                        processed_values.append(1e9)
                else:
                    processed_values.append(v)
            return tuple(processed_values)

        # Single-objective case
        value = values
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            if study.direction == optuna.study.StudyDirection.MAXIMIZE:
                return -1e9
            else:
                return 1e9
                
        return value

    # Calculate total work units: trials × windows per trial
    total_work_units = n_trials * len(windows)
    
    with Progress(
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
        trial_count = 0

        def callback(study, trial):
            nonlocal zero_streak, trial_count
            
            trial_count += 1
            
            # Update progress description to show current trial
            if _global_progress_tracker:
                _global_progress_tracker['current_trial'] = trial_count
                progress.update(
                    task,
                    description=f"[cyan]Trial {trial_count}/{n_trials} complete ({len(windows)} windows/trial)..."
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
            callbacks=[callback]
        )
        
        # Clear global progress tracker
        _global_progress_tracker = None
    
    optimal_params = scenario_config["strategy_params"].copy()
    best_trial_obj = None # Initialize best_trial_obj
    actual_trial_number_for_dsr = 0 # Initialize to a default value
    
    # CRITICAL: Handle Mock objects and real study objects safely
    # Problem: In tests, Mock objects are used to simulate Optuna study objects, but Mock objects
    # don't automatically support all operations that real objects do (like len(), attribute access)
    # Solution: Use defensive programming with getattr() and proper error handling
    study_directions = getattr(study, 'directions', [])
    study_trials = getattr(study, 'trials', [])
    
    # TRICKY: Check if study.directions supports len() operation
    # Mock objects might not support len() even if they have a __len__ attribute
    # We need to handle both TypeError (Mock doesn't support len) and AttributeError (no __len__)
    try:
        directions_length = len(study_directions) if hasattr(study_directions, '__len__') else 0
    except (TypeError, AttributeError):
        # If len() fails (common with Mock objects), assume single objective (directions_length = 0)
        directions_length = 0
    
    if directions_length > 1:
        best_trials = getattr(study, 'best_trials', [])
        if not best_trials:
            self.logger.error("Multi-objective optimization finished without finding any best trials.")
            # Handle Mock objects that might not support len()
            try:
                trials_count = len(study_trials) if hasattr(study_trials, '__len__') else 0
            except (TypeError, AttributeError):
                trials_count = 0
            return optimal_params, trials_count, None

        best_trial_obj = best_trials[0]
        optimal_params.update(best_trial_obj.params)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Best parameters (from Pareto front, first trial) found on training set: {best_trial_obj.params}")
            self.logger.debug(f"Optuna Optimizer - Best parameters found: {best_trial_obj.params}")
        actual_trial_number_for_dsr = best_trial_obj.number
    else:
        # Check if there are any completed trials
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            self.logger.error("Optimization finished, but no trials were completed successfully.")
            return optimal_params, 0, None

        best_trial = study.best_trial
        if not best_trial:
            self.logger.error("Single-objective optimization finished without finding a best trial.")
            return optimal_params, len(completed_trials), None

        best_trial_obj = best_trial
        optimal_params.update(best_trial_obj.params)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Best parameters found on training set: {best_trial_obj.params}")
            self.logger.debug(f"Optuna Optimizer - Best parameters found: {best_trial_obj.params}")
        actual_trial_number_for_dsr = best_trial.number
    
    return optimal_params, actual_trial_number_for_dsr, best_trial_obj


