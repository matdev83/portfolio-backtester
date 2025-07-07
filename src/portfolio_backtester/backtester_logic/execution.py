import pandas as pd
from ..utils import INTERRUPTED as CENTRAL_INTERRUPTED_FLAG

def run_backtest_mode(self, scenario_config, monthly_data, daily_data, rets_full):
    self.logger.info(f"Running backtest for scenario: {scenario_config['name']}")
    
    if self.args.study_name:
        try:
            import optuna
            study = optuna.load_study(study_name=self.args.study_name, storage="sqlite:///optuna_studies.db")
            optimal_params = scenario_config["strategy_params"].copy()
            optimal_params.update(study.best_params)
            scenario_config["strategy_params"] = optimal_params
            self.logger.info(f"Loaded best parameters from study '{self.args.study_name}': {optimal_params}")
        except KeyError:
            self.logger.warning(f'Study \'{self.args.study_name}\' not found. Using default parameters for scenario \'{scenario_config["name"]}\'.')
        except Exception as e:
            self.logger.error(f"Error loading Optuna study: {e}. Using default parameters.")

    rets = self.run_scenario(scenario_config, monthly_data, daily_data, rets_full, self.features)
    train_end_date = pd.to_datetime(scenario_config.get("train_end_date", "2018-12-31"))
    self.results[scenario_config["name"]] = {"returns": rets, "display_name": scenario_config["name"], "train_end_date": train_end_date}

def run_optimize_mode(self, scenario_config, monthly_data, daily_data, rets_full):
    self.logger.info(f"Running optimization for scenario: {scenario_config['name']}")
    
    optimal_params, actual_num_trials = self.run_optimization(scenario_config, monthly_data, daily_data, rets_full)
    
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
    
    self.logger.info(f"Running full backtest with optimal parameters: {optimal_params}")
    full_rets = self.run_scenario(optimized_scenario, monthly_data, daily_data, rets_full, self.features)
    
    optimized_name = f'{scenario_config["name"]} (Optimized)'
    self.results[optimized_name] = {
        "returns": full_rets if full_rets is not None else pd.Series(dtype=float),
        "display_name": optimized_name,
        "num_trials_for_dsr": actual_num_trials if actual_num_trials is not None else 0,
        "train_end_date": pd.to_datetime(scenario_config.get("train_end_date", "2018-12-31"))
    }
    self.logger.info(f"Full backtest with optimized parameters completed for {scenario_config['name']}.")