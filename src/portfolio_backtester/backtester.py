import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
import logging
import argparse
import os
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from typing import Dict, Any, Tuple
from functools import reduce
from operator import mul

from .config import GLOBAL_CONFIG, BACKTEST_SCENARIOS, OPTIMIZER_PARAMETER_DEFAULTS
from .data_sources.yfinance_data_source import YFinanceDataSource
from .data_sources.stooq_data_source import StooqDataSource
from . import strategies
from .portfolio.position_sizer import equal_weight_sizer
from .portfolio.rebalancing import rebalance
from .reporting.performance_metrics import calculate_metrics
from .feature import get_required_features_from_scenarios
from .feature_engineering import precompute_features
from .spy_holdings import (
    reset_history_cache,
    get_top_weight_sp500_components,
)
from .utils import _resolve_strategy
from .optimization.optuna_objective import build_objective
from .monte_carlo import run_monte_carlo_simulation, plot_monte_carlo_results

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ZERO_RET_EPS = 1e-8   # |returns| below this is considered "zero"

class Backtester:
    def __init__(self, global_config, scenarios, args, random_state=None):
        self.global_config = global_config
        # Add optimizer defaults to global_config for easy access
        self.global_config["optimizer_parameter_defaults"] = OPTIMIZER_PARAMETER_DEFAULTS
        self.scenarios = scenarios
        self.args = args
        self.data_source = self._get_data_source()
        self.results = {}
        self.features = None
        # Ensure random_state is always set for reproducibility
        if random_state is None:
            self.random_state = np.random.randint(0, 2**31 - 1) # Generate a random seed if none provided
            logger.info(f"No random seed provided. Using generated seed: {self.random_state}.")
        else:
            self.random_state = random_state
        np.random.seed(self.random_state)
        logger.info(f"Numpy random seed set to {self.random_state}.")
        # Parallelism / early-stop settings come from CLI
        self.n_jobs             = getattr(args, "n_jobs", 1)
        self.early_stop_patience = getattr(args, "early_stop_patience", 10)
        logger.info("Backtester initialized.")

    def _get_data_source(self):
        ds = self.global_config.get("data_source", "yfinance").lower()
        if ds == "stooq":
            logger.debug("Using StooqDataSource.")
            return StooqDataSource()
        elif ds == "yfinance":
            logger.debug("Using YFinanceDataSource.")
            return YFinanceDataSource()
        else:
            logger.error(f"Unsupported data source: {self.global_config['data_source']}")
            raise ValueError(f"Unsupported data source: {self.global_config['data_source']}")

    def _get_strategy(self, strategy_name, params):
        # Convert strategy_name (e.g., "vams_momentum") to class name (e.g., "VamsMomentumStrategy")
        class_name = "".join(word.capitalize() for word in strategy_name.split('_')) + "Strategy"
        # Special handling for VAMS strategies due to inconsistent naming convention
        if strategy_name == "vams_momentum":
            class_name = "VAMSMomentumStrategy"
        elif strategy_name == "vams_no_downside":
            class_name = "VAMSNoDownsideStrategy"
        
        strategy_class = getattr(strategies, class_name, None)
        
        if strategy_class:
            logger.debug(f"Using {class_name} with params: {params}")
            return strategy_class(params)
        else:
            logger.error(f"Unsupported strategy: {strategy_name}")
            raise ValueError(f"Unsupported strategy: {strategy_name}")

    def run_scenario(
        self,
        scenario_config,
        price_data_monthly: pd.DataFrame,
        price_data_daily: pd.DataFrame,
        rets_daily: pd.DataFrame | None = None,
        features: dict | None = None,
        verbose: bool = True,
    ):
        if verbose:
            logger.info(f"Running scenario: {scenario_config['name']}")

        if rets_daily is None:
            rets_daily = price_data_daily.pct_change().fillna(0)
        
        strategy = self._get_strategy(
            scenario_config["strategy"], scenario_config["strategy_params"]
        )
        
        # Get the universe from the strategy and extract only the tickers
        universe_tickers = [item[0] for item in strategy.get_universe(self.global_config)]
        
        # ------------------------------------------------------------------
        # 1) Signal generation uses *monthly* data
        # ------------------------------------------------------------------
        strategy_data_monthly = price_data_monthly[universe_tickers]
        benchmark_data_monthly = price_data_monthly[self.global_config["benchmark"]]
        
        signals = strategy.generate_signals(
            strategy_data_monthly,
            features,
            benchmark_data_monthly,
        )
        if verbose:
            logger.debug("Signals generated.")
            logger.info(f"Signals head:\n{signals.head()}")
            logger.info(f"Signals tail:\n{signals.tail()}")

        sized_signals = signals # Initialize sized_signals with raw signals
        if scenario_config["position_sizer"] == "equal_weight":
            sized_signals = equal_weight_sizer(signals)
            if verbose:
                logger.debug("Positions sized using equal_weight_sizer.")
                logger.info(f"Sized signals head:\n{sized_signals.head()}")
                logger.info(f"Sized signals tail:\n{sized_signals.tail()}")
        elif scenario_config["position_sizer"] != "none": # Add other sizers here if they exist
            logger.warning(f"Unsupported position sizer: {scenario_config['position_sizer']}. Using raw signals.")

        # ------------------------------------------------------------------
        # 2) Rebalance (monthly) and expand weights to daily frequency
        # ------------------------------------------------------------------
        weights_monthly = rebalance(
            sized_signals, scenario_config["rebalance_frequency"]
        )

        # Align monthly weights to the daily price index and forward-fill
        weights_daily = (
            weights_monthly.reindex(price_data_daily.index, method="ffill").fillna(0.0)
        )

        # ------------------------------------------------------------------
        # 3) Portfolio P&L on daily data (use previous-day weights)
        # ------------------------------------------------------------------
        aligned_rets_daily = rets_daily.loc[weights_daily.index]
        if verbose:
            logger.info(f"Aligned daily rets head:\n{aligned_rets_daily.head()}")
            logger.info(f"Weights (daily) head:\n{weights_daily.head()}")

        portfolio_rets_gross = (
            weights_daily.shift(1).fillna(0.0) * aligned_rets_daily
        ).sum(axis=1)
        if verbose:
            logger.info(f"Portfolio rets gross head:\n{portfolio_rets_gross.head()}")
            logger.info(f"Portfolio rets gross tail:\n{portfolio_rets_gross.tail()}")
        turnover = (weights_daily - weights_daily.shift(1)).abs().sum(axis=1)
        transaction_costs = turnover * (
            scenario_config["transaction_costs_bps"] / 10000
        )
        portfolio_rets_net = (
            portfolio_rets_gross - transaction_costs
        ).reindex(price_data_daily.index).fillna(0)
        if verbose:
            logger.info(f"Portfolio returns calculated for {scenario_config['name']}. First few returns: {portfolio_rets_net.head().to_dict()}")
            logger.info(f"portfolio_rets_gross index: {portfolio_rets_gross.index.min()} to {portfolio_rets_gross.index.max()}")
            logger.info(f"aligned_rets_daily index: {aligned_rets_daily.index.min()} to {aligned_rets_daily.index.max()}")
            logger.info(f"portfolio_rets_net index: {portfolio_rets_net.index.min()} to {portfolio_rets_net.index.max()}")

        return portfolio_rets_net

    def run(self):
        logger.info("Starting backtest data retrieval.")
        # ------------------------------------------------------------------
        # Fetch daily price data (no resampling).  We will keep two versions:
        #   • daily_data   – used for portfolio P&L & risk statistics (realistic)
        #   • monthly_data – last business-day of each month, used for signal
        #     generation and feature calculations that are specified using
        #     "month" look-back windows.
        # ------------------------------------------------------------------
        daily_data = self.data_source.get_data(
            tickers=self.global_config["universe"] + [self.global_config["benchmark"]],
            start_date=self.global_config["start_date"],
            end_date=self.global_config["end_date"]
        )

        # Ensure we drop any rows that are completely NaN (e.g. market holidays)
        daily_data.dropna(how="all", inplace=True)

        # Business-month-end prices for signal calculations (use 'BME' –
        # 'BM' alias is deprecated)
        monthly_data = daily_data.resample("BME").last()

        logger.info("Backtest data retrieved (daily) and business-month-end snapshot generated.")

        strategy_registry = {
            "calmar_momentum": strategies.CalmarMomentumStrategy,
            "vams_no_downside": strategies.VAMSNoDownsideStrategy,
            "momentum": strategies.MomentumStrategy,
            "sharpe_momentum": strategies.SharpeMomentumStrategy,
            "sortino_momentum": strategies.SortinoMomentumStrategy,
            "vams_momentum": strategies.VAMSMomentumStrategy,
        }
        
        required_features = get_required_features_from_scenarios(self.scenarios, strategy_registry)
        
        # Pre-compute features on *monthly* data so that a lookback of "11 months"
        # really means 11 month-end observations and not 11 trading days.
        self.features = precompute_features(
            monthly_data,
            required_features,
            monthly_data[self.global_config["benchmark"]]
        )
        logger.info("All features pre-computed.")

        # Daily return series used for portfolio P&L
        rets_full = daily_data.pct_change().fillna(0)

        if self.args.mode == "optimize":
            self._run_optimize_mode(self.scenarios[0], monthly_data, daily_data, rets_full)  # pass both data sets
        elif self.args.mode == "backtest":
            self._run_backtest_mode(self.scenarios[0], monthly_data, daily_data, rets_full)
        elif self.args.mode == "monte_carlo":
            self._run_monte_carlo_mode(self.scenarios[0], monthly_data, daily_data, rets_full)

        if self.args.mode != "monte_carlo":
            logger.info("All scenarios completed. Displaying results.")
            self.display_results(daily_data)

    def _run_backtest_mode(self, scenario_config, monthly_data, daily_data, rets_full):
        logger.info(f"Running backtest for scenario: {scenario_config['name']}")
        
        if self.args.study_name:
            try:
                study = optuna.load_study(study_name=self.args.study_name, storage="sqlite:///optuna_studies.db")
                optimal_params = scenario_config["strategy_params"].copy()
                optimal_params.update(study.best_params)
                scenario_config["strategy_params"] = optimal_params
                logger.info(f"Loaded best parameters from study '{self.args.study_name}': {optimal_params}")
            except KeyError:
                logger.warning(f'Study \'{self.args.study_name}\' not found. Using default parameters for scenario \'{scenario_config["name"]}\'.')
            except Exception as e:
                logger.error(f"Error loading Optuna study: {e}. Using default parameters.")

        rets = self.run_scenario(scenario_config, monthly_data, daily_data, rets_full, self.features)
        train_end_date = pd.to_datetime(scenario_config.get("train_end_date", "2018-12-31"))
        self.results[scenario_config["name"]] = {"returns": rets, "display_name": scenario_config["name"], "train_end_date": train_end_date}

    def _run_optimize_mode(self, scenario_config, monthly_data, daily_data, rets_full):
        logger.info(f"Running optimization for scenario: {scenario_config['name']}")
        
        # Step 1: Find optimal parameters on the training set
        optimal_params, actual_num_trials = self.run_optimization(scenario_config, monthly_data, daily_data, rets_full)
        
        # Step 2: Run a full backtest with the optimal parameters
        optimized_scenario = scenario_config.copy()
        optimized_scenario["strategy_params"] = optimal_params
        
        logger.info(f"Running full backtest with optimal parameters: {optimal_params}")
        full_rets = self.run_scenario(optimized_scenario, monthly_data, daily_data, rets_full, self.features)
        
        # Step 3: Store results for display, including actual number of trials for DSR
        optimized_name = f'{scenario_config["name"]} (Optimized)'
        self.results[optimized_name] = {
            "returns": full_rets, 
            "display_name": optimized_name,
            "num_trials_for_dsr": actual_num_trials,
            "train_end_date": pd.to_datetime(scenario_config.get("train_end_date", "2018-12-31"))
        }
        logger.info(f"Full backtest with optimized parameters completed for {scenario_config['name']}.")

    def run_optimization(self, scenario_config, monthly_data, daily_data, rets_full):
        logger.info(f"Running optimization for scenario: {scenario_config['name']} with simple train/test split.")

        # Define the train/test split point
        train_end_date = pd.to_datetime(scenario_config.get("train_end_date", "2018-12-31"))
        
        train_data_monthly = monthly_data[monthly_data.index <= train_end_date]
        train_data_daily   = daily_data[daily_data.index <= train_end_date]
        
        train_rets_sliced = rets_full[rets_full.index <= train_end_date]

        train_features_sliced = {}
        if self.features:
            train_features_sliced = {
                name: f[f.index <= train_end_date]
                for name, f in self.features.items()
            }

        logger.info(f"Training period: {train_data_daily.index.min()} to {train_data_daily.index.max()}")

        # --- Optuna Integration for finding best parameters on training set ---
        if self.args.storage_url:
            storage = self.args.storage_url
            db_path = storage.replace("sqlite:///", "")
        else:
            # Use JournalStorage for better file-based parallelization
            journal_dir = "optuna_journal"
            os.makedirs(journal_dir, exist_ok=True)
            db_path = os.path.join(journal_dir, f"{self.args.study_name or scenario_config['name']}.log")
            storage = JournalStorage(JournalFileBackend(file_path=db_path, lock_obj=JournalFileOpenLock(db_path)))

        study_name_base = f"{scenario_config['name']}_train_test"
        if self.args.study_name:
            study_name_base = f"{self.args.study_name}_{study_name_base}"

        if self.args.random_seed is not None:
            study_name = f"{study_name_base}_seed_{self.random_state}"
        else:
            study_name = study_name_base
        
        # --- Sampler selection ---
        optimization_specs = scenario_config.get("optimize", [])
        param_types = [
            self.global_config.get("optimizer_parameter_defaults", {}).get(spec["parameter"], {}).get("type")
            for spec in optimization_specs
        ]

        is_grid_search = all(pt == "int" for pt in param_types)
        
        if is_grid_search:
            search_space = {}
            for spec in optimization_specs:
                param_name = spec["parameter"]
                low = spec["min_value"]
                high = spec["max_value"]
                step = spec.get("step", 1)
                search_space[param_name] = list(range(low, high + 1, step))
            
            sampler = optuna.samplers.GridSampler(search_space)
            n_trials = reduce(mul, [len(v) for v in search_space.values()], 1)
            logger.info(f"Using GridSampler for optimization with search space: {search_space}. Total trials: {n_trials}")
        else:
            sampler = optuna.samplers.TPESampler(seed=self.random_state)
            n_trials = self.args.optuna_trials
            logger.info("Using TPESampler for optimization.")

        pruner = optuna.pruners.MedianPruner()

        if self.args.random_seed is not None: # If a random seed is provided, force a new study
            db_path = storage_url.replace("sqlite:///", "")
            if os.path.exists(db_path):
                logger.info(f"Attempting to delete existing Optuna study database at {db_path}.")
                try:
                    os.remove(db_path)
                    logger.info(f"Successfully deleted existing Optuna study database at {db_path}.")
                except Exception as e:
                    logger.error(f"Failed to delete Optuna study database at {db_path}. Reason: {e}")
            else:
                logger.info(f"No existing Optuna study database found at {db_path}. A new one will be created.")

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            pruner=pruner,
            direction="maximize",
            load_if_exists=(self.args.random_seed is None) # Only load if no explicit random seed is provided
        )

        objective = build_objective(
            self.global_config,
            scenario_config,
            train_data_monthly,
            train_data_daily,
            train_rets_sliced,
            daily_data[self.global_config["benchmark"]],
            train_features_sliced,
            metric=scenario_config["optimize"][0]["metric"]
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=self.args.optuna_timeout_sec,
            n_jobs=self.n_jobs
        )
        
        optimal_params = scenario_config["strategy_params"].copy()
        optimal_params.update(study.best_params)
        logger.info(f"Best parameters found on training set: {study.best_params}")
        
        return optimal_params, study.best_trial.number # Return optimal_params and number of trials

    def _run_monte_carlo_mode(self, scenario_config, monthly_data, daily_data, rets_full):
        logger.info(f"Running Monte Carlo simulation for scenario: {scenario_config['name']}")
        
        # Step 1: Find optimal parameters on the training set
        logger.info("Step 1: Finding optimal parameters...")
        optimal_params, actual_num_trials = self.run_optimization(scenario_config, monthly_data, daily_data, rets_full)
        
        optimized_scenario = scenario_config.copy()
        optimized_scenario["strategy_params"] = optimal_params

        # Step 2: Run on test data to get returns for MC simulation
        logger.info("Step 2: Generating test returns for Monte Carlo simulation...")
        train_end_date = pd.to_datetime(scenario_config.get("train_end_date", "2018-12-31"))
        test_data = monthly_data[monthly_data.index > train_end_date]
        test_rets_sliced = rets_full[rets_full.index > train_end_date]
        test_features_sliced = {name: f[f.index > train_end_date] for name, f in self.features.items()} if self.features else {}
        
        test_returns = self.run_scenario(optimized_scenario, test_data, daily_data[daily_data.index > train_end_date], test_rets_sliced, test_features_sliced, verbose=False)

        if test_returns is None or test_returns.empty:
            logger.error("Cannot run Monte Carlo simulation because test returns could not be generated.")
            return

        # Step 3: Run the Monte Carlo simulation
        logger.info("Step 3: Running Monte Carlo simulation...")
        n_simulations = scenario_config.get("mc_simulations", self.args.mc_simulations)
        n_years = scenario_config.get("mc_years", self.args.mc_years)
        mc_sims = run_monte_carlo_simulation(
            test_returns, 
            n_simulations=n_simulations,
            n_years=n_years
        )
        
        # Step 4: Run a full backtest with optimal parameters for the main report
        logger.info("Step 4: Running full backtest with optimal parameters for final report...")
        full_rets = self.run_scenario(optimized_scenario, monthly_data, daily_data, rets_full, self.features)
        
        # Store full backtest results
        optimized_name = f'{scenario_config["name"]} (Optimized)'
        self.results[optimized_name] = {
            "returns": full_rets, 
            "display_name": optimized_name,
            "num_trials_for_dsr": actual_num_trials
        }
        
        # Plot the MC results
        plot_monte_carlo_results(mc_sims, title=f"Monte Carlo Simulation: {scenario_config['name']} (Optimized)")
        logger.info("Monte Carlo simulation finished.")

    def display_results(self, data):
        logger.info("Generating performance report.")
        
        console = Console()
        
        # Get the train_end_date from the first scenario if available
        first_scenario_name = list(self.results.keys())[0]
        train_end_date = self.results[first_scenario_name].get("train_end_date")

        # --- Helper to generate and print a table for a specific period ---
        def generate_table(period_re_turns, bench_period_rets, title, num_trials_map):
            table = Table(title=title)
            table.add_column("Metric", style="cyan", no_wrap=True)

            # Calculate benchmark metrics for the period
            bench_metrics = calculate_metrics(bench_period_rets, bench_period_rets, self.global_config["benchmark"], name=self.global_config["benchmark"], num_trials=1)
            
            all_metrics = {self.global_config["benchmark"]: bench_metrics}
            
            # Add columns for each strategy
            for name in self.results.keys():
                display_name = self.results[name]["display_name"]
                table.add_column(display_name, style="magenta")

            table.add_column(self.global_config["benchmark"], style="green")

            # Calculate and add metrics for each strategy
            for name, result_data in self.results.items():
                rets = period_re_turns[name]
                display_name = result_data["display_name"]
                strategy_num_trials = num_trials_map.get(name, 1)
                
                metrics = calculate_metrics(rets, bench_period_rets, self.global_config["benchmark"], name=display_name, num_trials=strategy_num_trials)
                all_metrics[display_name] = metrics

            # Define which metrics are percentages and which need higher precision
            percentage_metrics = ["Total Return", "Ann. Return"]
            high_precision_metrics = ["ADF p-value"]

            for metric_name in bench_metrics.index:
                row = [metric_name]
                # Strategy metrics
                for name in all_metrics.keys():
                    if name != self.global_config["benchmark"]:
                        value = all_metrics[name].loc[metric_name]
                        if metric_name in percentage_metrics:
                            row.append(f"{value:.2%}")
                        elif metric_name in high_precision_metrics:
                            row.append(f"{value:.6f}")
                        else:
                            row.append(f"{value:.4f}")
                
                # Benchmark metrics
                bench_value = bench_metrics.loc[metric_name]
                if metric_name in percentage_metrics:
                    row.append(f"{bench_value:.2%}")
                elif metric_name in high_precision_metrics:
                    row.append(f"{bench_value:.6f}")
                else:
                    row.append(f"{bench_value:.4f}")
                table.add_row(*row)
            
            console.print(table)

        # --- Prepare data for different periods ---
        bench_rets = data[self.global_config["benchmark"]].pct_change(fill_method=None).fillna(0)
        
        # Full period data
        full_period_returns = {name: res["returns"] for name, res in self.results.items()}
        num_trials_full = {name: res.get("num_trials_for_dsr", 1) for name, res in self.results.items()}

        # In-sample and out-of-sample data
        if train_end_date:
            in_sample_returns = {name: res["returns"][res["returns"].index <= train_end_date] for name, res in self.results.items()}
            out_of_sample_returns = {name: res["returns"][res["returns"].index > train_end_date] for name, res in self.results.items()}
            
            bench_in_sample = bench_rets[bench_rets.index <= train_end_date]
            bench_out_of_sample = bench_rets[bench_rets.index > train_end_date]
            
            # For in-sample and out-of-sample, DSR is not applicable in the same way, so we use 1
            num_trials_split = {name: 1 for name in self.results.keys()}

            # --- Generate and display tables ---
            generate_table(in_sample_returns, bench_in_sample, "In-Sample Performance (Net of Costs)", num_trials_split)
            generate_table(out_of_sample_returns, bench_out_of_sample, "Out-of-Sample Performance (Net of Costs)", num_trials_split)
        
        generate_table(full_period_returns, bench_rets, "Full Period Performance (Net of Costs)", num_trials_full)

        logger.info("Performance tables displayed.")

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

        # Plot Cumulative Returns (P&L)
        ax1.set_title("Cumulative Returns (Net of Costs)", fontsize=16)
        ax1.set_ylabel("Cumulative Returns (Log Scale)", fontsize=12)
        ax1.set_yscale('log')
        
        # Calculate cumulative returns for all strategies and benchmark
        all_cumulative_returns = []
        for name, result_data in self.results.items():
            cumulative_strategy_returns = (1 + result_data["returns"]).cumprod()
            cumulative_strategy_returns.plot(ax=ax1, label=result_data["display_name"])
            all_cumulative_returns.append(cumulative_strategy_returns)
        
        cumulative_bench_returns = (1 + bench_rets).cumprod()
        cumulative_bench_returns.plot(ax=ax1, label=self.global_config["benchmark"], linestyle='--')
        all_cumulative_returns.append(cumulative_bench_returns)

        # Determine the maximum cumulative return across all series for setting y-limit
        max_cumulative_return = pd.concat(all_cumulative_returns).max()
        ax1.set_ylim(bottom=0.9, top=max_cumulative_return * 1.1) # Ensure initial flat line and add 10% buffer at top

        ax1.legend()
        ax1.grid(True, which="both", ls="-", alpha=0.5)

        # Add vertical line for train/test split
        if self.args.mode == "optimize" or self.args.mode == "backtest":
            # Get train_end_date from the first scenario's result_data if available
            first_scenario_name = list(self.results.keys())[0]
            first_scenario_data = self.results[first_scenario_name]
            if "train_end_date" in first_scenario_data:
                train_end_date = first_scenario_data["train_end_date"]
                ax1.axvline(train_end_date, color='gray', linestyle='--', lw=2, label='Train/Test Split')

        # Plot Drawdown
        ax2.set_ylabel("Drawdown", fontsize=12)
        ax2.set_xlabel("Date", fontsize=12)

        def calculate_drawdown(returns):
            cumulative_returns = (1 + returns).cumprod()
            peak = cumulative_returns.expanding(min_periods=1).max()
            drawdown = (cumulative_returns / peak) - 1
            return drawdown

        for name, result_data in self.results.items():
            drawdown = calculate_drawdown(result_data["returns"])
            drawdown.plot(ax=ax2, label=result_data["display_name"])
        
        bench_drawdown = calculate_drawdown(bench_rets)
        bench_drawdown.plot(ax=ax2, label=self.global_config["benchmark"], linestyle='--')

        ax2.legend()
        ax2.grid(True, which="both", ls="-", alpha=0.5)
        ax2.fill_between(bench_drawdown.index, 0, bench_drawdown, color='gray', alpha=0.2)

        plt.tight_layout()
        plt.show()
        logger.info("Cumulative returns and drawdown plots displayed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run portfolio backtester.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level.")
    parser.add_argument("--mode", type=str, required=True, choices=["backtest", "optimize", "monte_carlo"], help="Mode to run the backtester in.")
    parser.add_argument("--scenario-name", type=str, help="Name of the scenario to run/optimize from BACKTEST_SCENARIOS. Required for all modes.")
    parser.add_argument("--study-name", type=str, help="Name of the Optuna study to use for optimization or to load best parameters from.")
    parser.add_argument("--storage-url", type=str, help="Optuna storage URL. If not provided, SQLite will be used.")
    parser.add_argument("--random-seed", type=int, default=None, help="Set a random seed for reproducibility.")
    parser.add_argument("--optimize-min-positions", type=int, default=10, help="Minimum number of positions to consider during optimization of num_holdings.")
    parser.add_argument("--optimize-max-positions", type=int, default=30, help="Maximum number of positions to consider during optimization of num_holdings.")
    parser.add_argument("--top-n-params", type=int, default=3,
                        help="Number of top performing parameter values to keep per grid.")
    parser.add_argument("--n-jobs", type=int, default=8,
                        help="Parallel worker processes to use (-1 ⇒ all cores).")
    parser.add_argument("--early-stop-patience", type=int, default=10,
                        help="Stop optimisation after N successive ~zero-return evaluations.")
    parser.add_argument("--optuna-trials", type=int, default=200,
                        help="Maximum trials per optimization.")
    parser.add_argument("--optuna-timeout-sec", type=int, default=None,
                        help="Time budget per optimization (seconds).")
    # Monte Carlo specific arguments
    parser.add_argument("--mc-simulations", type=int, default=1000, help="Number of simulations for Monte Carlo analysis.")
    parser.add_argument("--mc-years", type=int, default=10, help="Number of years to project in Monte Carlo analysis.")

    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    # Only require --scenario-name for 'optimize' mode
    if args.mode == "optimize" and args.scenario_name is None:
        parser.error("--scenario-name is required for 'optimize' mode.")

    if args.scenario_name is not None:
        selected_scenarios = [s for s in BACKTEST_SCENARIOS if s["name"] == args.scenario_name]
        if not selected_scenarios:
            parser.error(f"Scenario '{args.scenario_name}' not found in BACKTEST_SCENARIOS.")
    else:
        selected_scenarios = BACKTEST_SCENARIOS

    backtester = Backtester(GLOBAL_CONFIG, selected_scenarios, args, random_state=args.random_seed)
    backtester.run()
