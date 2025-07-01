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

from .config import GLOBAL_CONFIG, BACKTEST_SCENARIOS
from .data_sources.yfinance_data_source import YFinanceDataSource
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ZERO_RET_EPS = 1e-8   # |returns| below this is considered "zero"

class Backtester:
    def __init__(self, global_config, scenarios, args, random_state=None):
        self.global_config = global_config
        self.scenarios = scenarios
        self.args = args
        self.data_source = self._get_data_source()
        self.results = {}
        self.features = None
        self.random_state = random_state
        if self.random_state is not None:
            np.random.seed(self.random_state)
            logger.info(f"Numpy random seed set to {self.random_state}.")
        # Parallelism / early-stop settings come from CLI
        self.n_jobs             = getattr(args, "n_jobs", 1)
        self.early_stop_patience = getattr(args, "early_stop_patience", 10)
        logger.info("Backtester initialized.")

    def _get_data_source(self):
        if self.global_config["data_source"] == "yfinance":
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

    def run_scenario(self, scenario_config, data, rets=None, features=None, verbose=True):
        if verbose:
            logger.info(f"Running scenario: {scenario_config['name']}")

        if rets is None:
            rets = data.pct_change().fillna(0)
        
        strategy = self._get_strategy(scenario_config["strategy"], scenario_config["strategy_params"])
        
        # Pass only the relevant portion of data to generate_signals
        strategy_data = data.loc[rets.index]
        benchmark_data = strategy_data[self.global_config["benchmark"]]
        
        signals = strategy.generate_signals(
            strategy_data.drop(columns=[self.global_config["benchmark"]]),
            features,
            benchmark_data
        )
        if verbose:
            logger.debug("Signals generated.")

        if scenario_config["position_sizer"] == "equal_weight":
            sized_signals = equal_weight_sizer(signals)
            if verbose:
                logger.debug("Positions sized using equal_weight_sizer.")
        else:
            logger.error(f"Unsupported position sizer: {scenario_config['position_sizer']}")
            raise ValueError(f"Unsupported position sizer: {scenario_config['position_sizer']}")

        weights = rebalance(sized_signals, scenario_config["rebalance_frequency"])
        if verbose:
            logger.debug("Portfolio rebalanced.")

        # Align rets with weights index
        aligned_rets = rets.loc[weights.index]
        
        portfolio_rets_gross = (weights.shift(1) * aligned_rets).sum(axis=1).dropna()
        turnover = (weights - weights.shift(1)).abs().sum(axis=1)
        transaction_costs = turnover * (scenario_config["transaction_costs_bps"] / 10000)
        portfolio_rets_net = (portfolio_rets_gross - transaction_costs).reindex(portfolio_rets_gross.index).fillna(0)
        if verbose:
            logger.info(f"Portfolio returns calculated for {scenario_config['name']}. First few returns: {portfolio_rets_net.head().to_dict()}")

        return portfolio_rets_net

    def run(self):
        logger.info("Starting backtest data retrieval.")
        data = self.data_source.get_data(
            tickers=self.global_config["universe"] + [self.global_config["benchmark"]],
            start_date=self.global_config["start_date"],
            end_date=self.global_config["end_date"]
        ).resample("ME").last()
        logger.info("Backtest data retrieved and resampled.")

        strategy_registry = {
            "calmar_momentum": strategies.CalmarMomentumStrategy,
            "vams_no_downside": strategies.VAMSNoDownsideStrategy,
            "momentum": strategies.MomentumStrategy,
            "sharpe_momentum": strategies.SharpeMomentumStrategy,
            "sortino_momentum": strategies.SortinoMomentumStrategy,
            "vams_momentum": strategies.VAMSMomentumStrategy,
        }
        
        required_features = get_required_features_from_scenarios(self.scenarios, strategy_registry)
        
        self.features = precompute_features(
            data,
            required_features,
            data[self.global_config["benchmark"]]
        )
        logger.info("All features pre-computed.")

        # Pre-calculate returns for the entire dataset
        rets_full = data.pct_change().fillna(0)

        for scenario in self.scenarios:
            if "optimize" in scenario:
                self.run_optimization(scenario, data, rets_full)
            else:
                # Pass the full returns dataframe to run_scenario
                rets = self.run_scenario(scenario, data, rets_full, self.features)
                self.results[scenario["name"]] = {"returns": rets, "display_name": scenario["name"]}

        logger.info("All scenarios completed. Displaying results.")
        self.display_results(data)

    def run_optimization(self, scenario_config, data, rets_full):
        logger.info(f"Running walk-forward optimization for scenario: {scenario_config['name']}")

        train_window = scenario_config["train_window_months"]
        test_window = scenario_config["test_window_months"]
        
        all_test_rets = []
        
        num_steps = (len(data) - train_window) // test_window
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=Console()
        ) as progress:
            wfa_task = progress.add_task(f"[green]Running Walk-Forward Analysis for {scenario_config['name']}...", total=num_steps)

            for i in range(num_steps):
                train_start_idx = i * test_window
                train_end_idx = train_start_idx + train_window
                test_start_idx = train_end_idx
                test_end_idx = test_start_idx + test_window

                if train_end_idx > len(data) or test_end_idx > len(data):
                    break

                train_data = data.iloc[train_start_idx:train_end_idx]
                test_data = data.iloc[test_start_idx:test_end_idx]
                
                train_rets_sliced = rets_full.iloc[train_start_idx:train_end_idx]
                test_rets_sliced = rets_full.iloc[test_start_idx:test_end_idx]
                
                train_features_sliced = {
                    name: f.iloc[train_start_idx:train_end_idx]
                    for name, f in self.features.items()
                }

                progress.update(wfa_task, description=f"[green]Optimizing period {i+1}/{num_steps}...")

                # --- Optuna Integration ---
                storage_url = "sqlite:///optuna_studies.db"
                study_name = f"{scenario_config['name']}_wfa_slice_{i}"
                
                sampler = optuna.samplers.TPESampler(seed=self.random_state)
                pruner = optuna.pruners.MedianPruner()
                
                study = optuna.create_study(
                    study_name=study_name,
                    storage=storage_url,
                    sampler=sampler,
                    pruner=pruner,
                    direction="maximize",
                    load_if_exists=True
                )

                objective = build_objective(
                    self.global_config,
                    scenario_config,
                    train_data,
                    train_rets_sliced,
                    train_data[self.global_config["benchmark"]],
                    train_features_sliced,
                    metric=scenario_config["optimize"][0]["metric"]
                )

                study.optimize(
                    objective,
                    n_trials=self.args.optuna_trials,
                    timeout=self.args.optuna_timeout_sec,
                    n_jobs=self.n_jobs
                )
                
                optimal_params = scenario_config["strategy_params"].copy()
                optimal_params.update(study.best_params)
                # --- End Optuna Integration ---

                test_scenario = scenario_config.copy()
                test_scenario["strategy_params"] = optimal_params
                
                test_features_sliced = {
                    name: f.iloc[test_start_idx:test_end_idx]
                    for name, f in self.features.items()
                }
                
                test_rets = self.run_scenario(test_scenario, test_data, test_rets_sliced, test_features_sliced, verbose=False)
                all_test_rets.append(test_rets)
                
                progress.advance(wfa_task)

        if not all_test_rets:
            logger.warning(
                f"No optimization periods were run for {scenario_config['name']}. "
                "The dataset might be too small for the specified train/test windows. "
                "Executing base strategy run instead."
            )
            base_rets = self.run_scenario(scenario_config, data, rets_full, self.features, verbose=False)
            self.results[scenario_config['name']] = {
                "returns": base_rets,
                "display_name": scenario_config["name"]
            }
            return

        final_returns = pd.concat(all_test_rets)
        
        optimized_name = f'{scenario_config["name"]} (Walk-Forward Optimized)'
        self.results[optimized_name] = {"returns": final_returns, "display_name": optimized_name}
        logger.info(f"Walk-forward optimization completed for {scenario_config['name']}.")

    def display_results(self, data):
        logger.info("Generating performance report.")
        bench_rets = data[self.global_config["benchmark"]].pct_change().fillna(0)
        bench_metrics = calculate_metrics(bench_rets, bench_rets, self.global_config["benchmark"], name=self.global_config["benchmark"])

        console = Console()
        table = Table(title="Strategy Performance Comparison (Net of Costs)")
        table.add_column("Metric", style="cyan", no_wrap=True)

        all_metrics = {self.global_config["benchmark"]: bench_metrics}
        for name, result_data in self.results.items():
            rets = result_data["returns"]
            display_name = result_data["display_name"]
            metrics = calculate_metrics(rets, bench_rets, self.global_config["benchmark"], name=display_name)
            all_metrics[display_name] = metrics
            table.add_column(display_name, style="magenta")
        table.add_column(self.global_config["benchmark"], style="green")

        for metric_name in bench_metrics.index:
            row = [metric_name]
            for name, metrics in all_metrics.items():
                if name != self.global_config["benchmark"]:
                    value = metrics.loc[metric_name]
                    row.append(f"{value:.2%}" if metric_name == "Total Return" else f"{value:.4f}")
            
            bench_value = bench_metrics.loc[metric_name]
            row.append(f"{bench_value:.2%}" if metric_name == "Total Return" else f"{bench_value:.4f}")
            table.add_row(*row)

        console.print(table)
        logger.info("Performance table displayed.")

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

        # Plot Cumulative Returns (P&L)
        ax1.set_title("Cumulative Returns (Net of Costs)", fontsize=16)
        ax1.set_ylabel("Cumulative Returns (Log Scale)", fontsize=12)
        ax1.set_yscale('log')

        for name, result_data in self.results.items():
            (1 + result_data["returns"]).cumprod().plot(ax=ax1, label=result_data["display_name"])
        (1 + bench_rets).cumprod().plot(ax=ax1, label=self.global_config["benchmark"], linestyle='--')

        ax1.legend()
        ax1.grid(True, which="both", ls="-", alpha=0.5)

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
    parser.add_argument("--portfolios", type=str, help="Comma-separated list of portfolio scenario names to run. If omitted, all scenarios (including the baseline 'Momentum_Unfiltered') will be executed.")
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
                        help="Maximum trials per WFA slice.")
    parser.add_argument("--optuna-timeout-sec", type=int, default=None,
                        help="Time budget per WFA slice (seconds).")
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    selected_scenarios = []
    if args.portfolios:
        portfolio_names = [name.strip().strip('"') for name in args.portfolios.split(',')]
        for scenario in BACKTEST_SCENARIOS:
            if scenario["name"] in portfolio_names:
                selected_scenarios.append(scenario)
    else:
        # No specific portfolios supplied: run the full suite and include the baseline
        selected_scenarios = BACKTEST_SCENARIOS.copy()

        # Ensure the baseline "Momentum_Unfiltered" is present (should already be, but be explicit)
        unfiltered_scenario = next((s for s in BACKTEST_SCENARIOS if s["name"] == "Momentum_Unfiltered"), None)
        if unfiltered_scenario and unfiltered_scenario not in selected_scenarios:
            selected_scenarios.insert(0, unfiltered_scenario)

    backtester = Backtester(GLOBAL_CONFIG, selected_scenarios, args, random_state=args.random_seed)
    backtester.run()

    reset_history_cache()          # flush old in-memory state
    print(get_top_weight_sp500_components("2024-07-01", 10))
