import pandas as pd
from ..utils import INTERRUPTED as CENTRAL_INTERRUPTED_FLAG
from ..reporting.optimizer_report_generator import create_optimization_report
from ..reporting.performance_metrics import calculate_metrics
from .constraint_logic import handle_constraints
import logging

logger = logging.getLogger(__name__)

def run_backtest_mode(self, scenario_config, monthly_data, daily_data, rets_full):
    if self.logger.isEnabledFor(logging.DEBUG):
        self.logger.debug(f"Running backtest for scenario: {scenario_config['name']}")
    
    if self.args.study_name:
        try:
            import optuna
            study = optuna.load_study(study_name=self.args.study_name, storage="sqlite:///optuna_studies.db")
            optimal_params = scenario_config["strategy_params"].copy()
            optimal_params.update(study.best_params)
            scenario_config["strategy_params"] = optimal_params
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Loaded best parameters from study '{self.args.study_name}': {optimal_params}")
        except KeyError:
            self.logger.warning(f'Study \'{self.args.study_name}\' not found. Using default parameters for scenario \'{scenario_config["name"]}\'.')
        except Exception as e:
            self.logger.error(f"Error loading Optuna study: {e}. Using default parameters.")

    rets = self.run_scenario(scenario_config, monthly_data, daily_data, rets_full)
    train_end_date = pd.to_datetime(scenario_config.get("train_end_date", "2018-12-31"))
    self.results[scenario_config["name"]] = {"returns": rets, "display_name": scenario_config["name"], "train_end_date": train_end_date}

def run_optimize_mode(self, scenario_config, monthly_data, daily_data, rets_full):
    if self.logger.isEnabledFor(logging.DEBUG):
        self.logger.debug(f"Running optimization for scenario: {scenario_config['name']}")
    
    optimal_params, actual_num_trials, best_trial_obj = self.run_optimization(scenario_config, monthly_data, daily_data, rets_full)
    
    if CENTRAL_INTERRUPTED_FLAG:
        self.logger.warning(f"Optimization for {scenario_config['name']} was interrupted. Skipping full backtest with potentially incomplete optimal parameters.")
        interrupted_name = f'{scenario_config["name"]} (Optimization Interrupted)'
        self.results[interrupted_name] = {
            "returns": pd.Series(dtype=float),
            "display_name": interrupted_name,
            "num_trials_for_dsr": actual_num_trials if actual_num_trials is not None else 0,
            "train_end_date": pd.to_datetime(scenario_config.get("train_end_date", "2018-12-31")),
            "notes": "Optimization process was interrupted by the user."
        }
        return
    
    if optimal_params is None:
        self.logger.error(f"Optimization for {scenario_config['name']} did not yield optimal parameters. Skipping full backtest.")
        return

    optimized_scenario = scenario_config.copy()
    optimized_scenario["strategy_params"] = optimal_params
    
    if self.logger.isEnabledFor(logging.DEBUG):
        self.logger.debug(f"Running full backtest with optimal parameters: {optimal_params}")
    full_rets = self.run_scenario(optimized_scenario, monthly_data, daily_data, rets_full)
    
    optimized_name, full_rets, optimal_params, constraint_status, constraint_message, constraint_violations, constraints_config = handle_constraints(self, scenario_config, optimal_params, full_rets, monthly_data, daily_data, rets_full)
    
    self.results[optimized_name] = {
        "returns": full_rets if full_rets is not None else pd.Series(dtype=float),
        "display_name": optimized_name,
        "optimal_params": optimal_params,
        "num_trials_for_dsr": actual_num_trials if actual_num_trials is not None else 0,
        "train_end_date": pd.to_datetime(scenario_config.get("train_end_date", "2018-12-31")),
        "best_trial_obj": best_trial_obj,  # Pass the best trial object
        "constraint_status": constraint_status,
        "constraint_message": constraint_message,
        "constraint_violations": constraint_violations if constraint_violations else [],
        "constraints_config": constraints_config
    }
    if self.logger.isEnabledFor(logging.DEBUG):
        self.logger.debug(f"Full backtest with optimized parameters completed for {scenario_config['name']}.")
    
    # Generate optimization report (with performance optimizations)
    advanced_reporting_config = self.global_config.get('advanced_reporting_config', {})
    enable_during_optimization = advanced_reporting_config.get('enable_during_optimization', False)
    enable_optimization_reports = advanced_reporting_config.get('enable_optimization_reports', True)
    
    if enable_optimization_reports:
        if enable_during_optimization:
            # Full reporting during optimization (slower but immediate)
            try:
                from ..backtester_logic.reporting_logic import generate_optimization_report
                generate_optimization_report(self, scenario_config, optimal_params, full_rets, best_trial_obj, actual_num_trials)
            except Exception as e:
                self.logger.error(f"Failed to generate optimization report: {e}")
                self.logger.debug("Report generation error details:", exc_info=True)
        else:
            # Deferred reporting for performance (faster optimization)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("⚡ Optimization report generation deferred for performance - will generate after optimization completes")
            # Store data for later report generation
            self._deferred_report_data = {
                'scenario_config': scenario_config,
                'optimal_params': optimal_params,
                'full_rets': full_rets,
                'best_trial_obj': best_trial_obj,
                'actual_num_trials': actual_num_trials
            }
    else:
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("📊 Optimization reports disabled in configuration")



def generate_deferred_report(self):
    """Generate the deferred optimization report after optimization completes."""
    if hasattr(self, '_deferred_report_data') and self._deferred_report_data:
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("📊 Generating deferred optimization report...")
        try:
            from ..backtester_logic.reporting_logic import generate_optimization_report
            generate_optimization_report(self, **self._deferred_report_data)
            # Clear the deferred data
            delattr(self, '_deferred_report_data')
        except Exception as e:
            self.logger.error(f"Failed to generate deferred optimization report: {e}")
    else:
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("No deferred report data available")