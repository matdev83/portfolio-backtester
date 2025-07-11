from ..monte_carlo.monte_carlo import run_monte_carlo_simulation, plot_monte_carlo_results
import pandas as pd

def run_monte_carlo_mode(self, scenario_config, monthly_data, daily_data, rets_full):
    logger = self.logger
    if logger.isEnabledFor(logging.INFO):
        logger.info(f"Running Monte Carlo simulation for scenario: {scenario_config['name']}")
    
    if logger.isEnabledFor(logging.INFO):
        logger.info("Step 1: Finding optimal parameters...")
    optimal_params, actual_num_trials = self.run_optimization(scenario_config, monthly_data, daily_data, rets_full)
    
    optimized_scenario = scenario_config.copy()
    optimized_scenario["strategy_params"] = optimal_params

    if logger.isEnabledFor(logging.INFO):
        logger.info("Step 2: Generating test returns for Monte Carlo simulation...")
    train_end_date = pd.to_datetime(scenario_config.get("train_end_date", "2018-12-31"))
    test_data = monthly_data[monthly_data.index > train_end_date]
    test_rets_sliced = rets_full[rets_full.index > train_end_date]
    test_features_sliced = {name: f[f.index > train_end_date] for name, f in self.features.items()} if self.features else {}
    
    test_returns = self.run_scenario(optimized_scenario, test_data, daily_data[daily_data.index > train_end_date], test_rets_sliced, test_features_sliced, verbose=False)

    if test_returns is None or test_returns.empty:
        logger.error("Cannot run Monte Carlo simulation because test returns could not be generated.")
        return

    if logger.isEnabledFor(logging.INFO):
        logger.info("Step 3: Running Monte Carlo simulation...")
    n_simulations = scenario_config.get("mc_simulations", self.args.mc_simulations)
    n_years = scenario_config.get("mc_years", self.args.mc_years)
    mc_sims = run_monte_carlo_simulation(
        test_returns,
        n_simulations=n_simulations,
        n_years=n_years
    )
    
    if logger.isEnabledFor(logging.INFO):
        logger.info("Step 4: Running full backtest with optimal parameters for final report...")
    full_rets = self.run_scenario(optimized_scenario, monthly_data, daily_data, rets_full, self.features)
    
    optimized_name = f'{scenario_config["name"]} (Optimized)'
    self.results[optimized_name] = {
        "returns": full_rets,
        "display_name": optimized_name,
        "num_trials_for_dsr": actual_num_trials
    }
    
    plot_monte_carlo_results(
        mc_sims,
        title=f"Monte Carlo Simulation: {scenario_config['name']} (Optimized)",
        scenario_name=scenario_config.get('name'),
        params=scenario_config.get('strategy_params'),
        interactive=getattr(self.args, 'interactive', False)
    )
    if logger.isEnabledFor(logging.INFO):
        logger.info("Monte Carlo simulation finished.")