import setuptools # Ensure setuptools is imported before pandas_datareader
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
from typing import Any, Dict, Tuple, List
from functools import reduce
from operator import mul
from datetime import datetime

from .config_loader import GLOBAL_CONFIG, BACKTEST_SCENARIOS, OPTIMIZER_PARAMETER_DEFAULTS
from .config_initializer import populate_default_optimizations
from . import strategies
from .portfolio.position_sizer import get_position_sizer
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
from .constants import ZERO_RET_EPS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, global_config, scenarios, args, random_state=None):
        self.global_config = global_config
        # Add optimizer defaults to global_config for easy access
        self.global_config["optimizer_parameter_defaults"] = OPTIMIZER_PARAMETER_DEFAULTS
        self.scenarios = scenarios
        logger.debug(f"Backtester initialized with scenario strategy_params: {self.scenarios[0].get('strategy_params')}")
        populate_default_optimizations(self.scenarios, OPTIMIZER_PARAMETER_DEFAULTS) # Call after scenarios are loaded
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
        """Instantiate the data source lazily to avoid heavy imports during test
        collection."""
        ds = self.global_config.get("data_source", "yfinance").lower()
        if ds == "stooq":
            logger.debug("Using StooqDataSource.")
            from .data_sources.stooq_data_source import StooqDataSource
            return StooqDataSource()
        elif ds == "yfinance":
            logger.debug("Using YFinanceDataSource.")
            from .data_sources.yfinance_data_source import YFinanceDataSource
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
            rets_daily = price_data_daily.pct_change(fill_method=None).fillna(0)
        
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

        sizer_name = scenario_config.get("position_sizer", "equal_weight")
        sizer_func = get_position_sizer(sizer_name)
        
        # Start with all strategy_params as potential sizer parameters
        # This ensures 'target_volatility' and 'target_return' are included if present
        filtered_sizer_params = scenario_config.get("strategy_params", {}).copy()
        logger.debug(f"Initial filtered_sizer_params: {filtered_sizer_params}")
        
        # Define mappings for parameters that need to be renamed for the sizer function
        sizer_param_mapping = {
            "sizer_sharpe_window": "window",
            "sizer_sortino_window": "window",
            "sizer_beta_window": "window",
            "sizer_corr_window": "window",
            "sizer_dvol_window": "window",
            "sizer_target_return": "target_return", # For Sortino sizer, if it's ever mapped
            "sizer_max_leverage": "max_leverage", # Add mapping for max_leverage
        }
        
        # Apply remapping: if an old_key exists, rename it to new_key and remove old_key
        for old_key, new_key in sizer_param_mapping.items():
            if old_key in filtered_sizer_params:
                filtered_sizer_params[new_key] = filtered_sizer_params.pop(old_key)
        logger.debug(f"Filtered_sizer_params after remapping: {filtered_sizer_params}")

        # Extract 'window' if it's present, as it's a required positional argument for some sizers
        window_param = filtered_sizer_params.pop("window", None)
        logger.debug(f"window_param extracted: {window_param}")
        logger.debug(f"Filtered_sizer_params after window pop: {filtered_sizer_params}")

        # Extract 'target_return' if it's present, as it's a required positional argument for some sizers
        target_return_param = filtered_sizer_params.pop("target_return", None)
        logger.debug(f"target_return_param extracted: {target_return_param}")
        logger.debug(f"Filtered_sizer_params after target_return pop: {filtered_sizer_params}")

        # Extract 'max_leverage' if it's present, as it's a keyword argument for some sizers
        max_leverage_param = filtered_sizer_params.pop("max_leverage", None)
        logger.debug(f"max_leverage_param extracted: {max_leverage_param}")
        logger.debug(f"Filtered_sizer_params after max_leverage pop: {filtered_sizer_params}")

        # Prepare arguments for the sizer function
        sizer_args = [signals, strategy_data_monthly, benchmark_data_monthly]
        
        # Pass daily data for volatility calculation if the sizer needs it
        if sizer_name == "rolling_downside_volatility":
            # Ensure daily_data is sliced to universe_tickers for the sizer
            daily_prices_for_vol = price_data_daily[universe_tickers]
            sizer_args.append(daily_prices_for_vol)

        if sizer_name in ["rolling_sharpe", "rolling_sortino", "rolling_beta", "rolling_benchmark_corr", "rolling_downside_volatility"]:
            if window_param is None:
                raise ValueError(f"Sizer '{sizer_name}' requires a 'window' parameter, but it was not found in strategy_params.")
            sizer_args.append(window_param)
        
        # Add target_return for Sortino sizer if applicable
        if sizer_name == "rolling_sortino":
            if target_return_param is None:
                # Use default from sizer function if not provided in config
                sizer_args.append(0.0) # Default target_return for rolling_sortino_sizer
            else:
                sizer_args.append(target_return_param)
        
        # Add max_leverage for rolling_downside_volatility sizer if applicable
        if sizer_name == "rolling_downside_volatility" and max_leverage_param is not None:
            filtered_sizer_params["max_leverage"] = max_leverage_param

        logger.debug(f"Sizer arguments prepared: {sizer_args}")
        logger.debug(f"Final keyword arguments for sizer: {filtered_sizer_params}")

        sized_signals = sizer_func(
            *sizer_args,
            **filtered_sizer_params,
        )
        if verbose:
            logger.debug(f"Positions sized using {sizer_name}.")
            logger.info(f"Sized signals head:\n{sized_signals.head()}")
            logger.info(f"Sized signals tail:\n{sized_signals.tail()}")

        # ------------------------------------------------------------------
        # 2) Rebalance (monthly) and expand weights to daily frequency
        # ------------------------------------------------------------------
        weights_monthly = rebalance(
            sized_signals, scenario_config["rebalance_frequency"]
        )

        # ------------------------------------------------------------------
        # 3) Calculate daily weights and P&L
        # ------------------------------------------------------------------
        # Ensure weights_monthly has the same columns as universe_tickers, filling missing with 0
        weights_monthly = weights_monthly.reindex(columns=universe_tickers).fillna(0.0)

        # Expand monthly weights to daily frequency (forward fill)
        weights_daily = weights_monthly.reindex(price_data_daily.index, method="ffill")
        weights_daily = weights_daily.shift(1).fillna(0.0) # Shift to avoid look-ahead bias

        # Ensure rets_daily is aligned with price_data_daily.index and contains all universe_tickers
        aligned_rets_daily = rets_daily.reindex(price_data_daily.index).fillna(0.0)

        # Calculate daily portfolio returns
        daily_portfolio_returns_gross = (weights_daily * aligned_rets_daily[universe_tickers]).sum(axis=1)

        # Calculate turnover and transaction costs
        turnover = (weights_daily - weights_daily.shift(1)).abs().sum(axis=1).fillna(0.0)
        transaction_costs = turnover * (scenario_config.get("transaction_costs_bps", 0) / 10000.0)

        portfolio_rets_net = (daily_portfolio_returns_gross - transaction_costs).fillna(0.0)

        if verbose:
            logger.info(f"Portfolio net returns calculated for {scenario_config['name']}. First few net returns: {portfolio_rets_net.head().to_dict()}")
            logger.info(f"Net returns index: {portfolio_rets_net.index.min()} to {portfolio_rets_net.index.max()}")

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
            "momentum_dvol_sizer": strategies.MomentumDvolSizerStrategy, # Added this line
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
        rets_full = daily_data.pct_change(fill_method=None).fillna(0)

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
        """Optimize parameters using walk-forward train/test splits."""
        logger.info(
            f"Running optimization for scenario: {scenario_config['name']} with walk-forward splits."
        )

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

        logger.info(f"Generated {len(windows)} walk-forward windows using '{wf_type}' splits.")

        # --- Optuna integration ---
        if self.args.storage_url:
            storage = self.args.storage_url
            db_path = storage.replace("sqlite:///", "")
        else:
            journal_dir = "optuna_journal"
            os.makedirs(journal_dir, exist_ok=True)
            db_path = os.path.join(journal_dir, f"{self.args.study_name or scenario_config['name']}.log")
            storage = JournalStorage(JournalFileBackend(file_path=db_path, lock_obj=JournalFileOpenLock(db_path)))

        study_name_base = f"{scenario_config['name']}_walk_forward"
        if self.args.study_name:
            study_name_base = f"{self.args.study_name}_{study_name_base}"
        
        study, n_trials = self._setup_optuna_study(scenario_config, storage, study_name_base)

        metrics_to_optimize = [t["name"] for t in scenario_config.get("optimization_targets", [])] or \
                              [scenario_config.get("optimization_metric", "Calmar")]
        is_multi_objective = len(metrics_to_optimize) > 1

        # Define the objective function for Optuna
        def objective(trial: optuna.trial.Trial):
            # Suggest parameters for the current trial
            current_params = self._suggest_optuna_params(trial, scenario_config["strategy_params"], scenario_config.get("optimize", []))

            # Create a scenario configuration for the current trial
            trial_scenario_config = scenario_config.copy()
            trial_scenario_config["strategy_params"] = current_params

            # Evaluate parameters across all walk-forward windows
            return self._evaluate_params_walk_forward(
                trial,  # Pass the trial object
                trial_scenario_config, windows, monthly_data, daily_data, rets_full,
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
                if zero_streak > self.early_stop_patience:
                    study.stop()

            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=self.args.optuna_timeout_sec,
                n_jobs=self.n_jobs,
                callbacks=[callback]
            )
        
        optimal_params = scenario_config["strategy_params"].copy()
        optimal_params.update(study.best_params)
        logger.info(f"Best parameters found on training set: {study.best_params}")
        
        return optimal_params, study.best_trial.number # Return optimal_params and number of trials

    def _setup_optuna_study(self, scenario_config, storage, study_name_base: str) -> Tuple[optuna.Study, int]:
        """Sets up and returns an Optuna study object and the number of trials."""
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
            logger.info(f"Using GridSampler with search space: {search_space}. Total trials: {n_trials_actual}")
        else:
            if is_grid_search and self.n_jobs > 1:
                logger.warning("Grid search is not supported with n_jobs > 1. Using TPESampler instead.")
            sampler = optuna.samplers.TPESampler(seed=self.random_state)
            logger.info(f"Using TPESampler with {n_trials_actual} trials.")

        if self.args.pruning_enabled:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=self.args.pruning_n_startup_trials,
                n_warmup_steps=self.args.pruning_n_warmup_steps,
                interval_steps=self.args.pruning_interval_steps
            )
            logger.info(f"MedianPruner enabled with n_startup_trials={self.args.pruning_n_startup_trials}, n_warmup_steps={self.args.pruning_n_warmup_steps}, interval_steps={self.args.pruning_interval_steps}.")
        else:
            # If pruning is not enabled via CLI, use a pruner that never prunes.
            pruner = optuna.pruners.NopPruner()
            logger.info("Pruning disabled (NopPruner used).")

        if self.args.random_seed is not None:
            try:
                optuna.delete_study(study_name=study_name, storage=storage)
                logger.info(f"Deleted existing Optuna study '{study_name}' for fresh start with random seed.")
            except KeyError: # Study doesn't exist
                pass
            except Exception as e:
                logger.warning(f"Could not delete existing Optuna study '{study_name}': {e}")

        optimization_targets_config = scenario_config.get("optimization_targets", [])
        study_directions = [t.get("direction", "maximize").lower() for t in optimization_targets_config] or ["maximize"]
        for i, d in enumerate(study_directions):
            if d not in ["maximize", "minimize"]:
                logger.warning(f"Invalid direction '{d}' for target. Defaulting to 'maximize'.")
                study_directions[i] = "maximize"

        study = optuna.create_study(
            study_name=study_name, storage=storage, sampler=sampler, pruner=pruner,
            directions=study_directions if len(study_directions) > 1 else None,
            direction=study_directions[0] if len(study_directions) == 1 else None,
            load_if_exists=(self.args.random_seed is None)
        )
        return study, n_trials_actual

    def _suggest_optuna_params(self, trial: optuna.trial.Trial, base_params: Dict[str, Any], opt_specs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Suggests parameters for an Optuna trial based on optimization specifications."""
        params = base_params.copy()
        for spec in opt_specs:
            pname = spec["parameter"]
            opt_def = self.global_config.get("optimizer_parameter_defaults", {}).get(pname, {})
            ptype = opt_def.get("type", spec.get("type")) # Fallback to spec type

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
                    logger.warning(f"Categorical parameter '{pname}' has no choices defined or choices are invalid. Skipping suggestion.")
                    continue # Skip this parameter
                params[pname] = trial.suggest_categorical(pname, choices)
            else:
                logger.warning(f"Unsupported parameter type '{ptype}' for {pname}. Skipping suggestion.")
        return params

    def _evaluate_params_walk_forward(self, trial: optuna.trial.Trial, scenario_config: Dict[str, Any], windows: list,
                                     monthly_data: pd.DataFrame, daily_data: pd.DataFrame, rets_full: pd.DataFrame,
                                     metrics_to_optimize: List[str], is_multi_objective: bool) -> Any:
        """Evaluates a set of parameters across walk-forward windows, with intermediate reporting for pruning."""
        metric_sums = np.zeros(len(metrics_to_optimize))
        num_valid_windows = 0
        processed_steps_for_pruning = 0 # Tracks steps for pruning interval

        # Get pruning configuration from args (will be properly set in Step 3 of the plan)
        pruning_enabled = getattr(self.args, "pruning_enabled", False) # Default to False for now
        pruning_interval_steps = getattr(self.args, "pruning_interval_steps", 1) # Default to 1 for now

        for window_idx, (tr_start, tr_end, te_start, te_end) in enumerate(windows):
            # Slice data for the current window
            m_slice = monthly_data.loc[tr_start:te_end]
            d_slice = daily_data.loc[tr_start:te_end] # daily_data is used for benchmark in calculate_metrics
            r_slice = rets_full.loc[tr_start:te_end]

            # Slice features to the current combined train+test window (m_slice.index covers this)
            # Ensure that all feature dataframes in f_slice share the same index as m_slice
            # and that all required features are present, possibly with NaNs if they don't cover the full window.
            f_slice = {}
            if self.features:
                for name, feat_df in self.features.items():
                    # Align feature's index with m_slice.index (which spans tr_start to te_end for monthly)
                    f_slice[name] = feat_df.reindex(m_slice.index)


            # Run scenario for the current window's training and testing period combined
            # The scenario internally might use only training data for signals if designed that way.
            # For evaluation, we need returns over the test period.
            window_returns = self.run_scenario(scenario_config, m_slice, d_slice, r_slice, f_slice, verbose=False)

            if window_returns is None or window_returns.empty:
                logger.warning(f"No returns generated for window {tr_start}-{te_end}. Skipping.")
                continue

            # Extract test period returns
            test_rets = window_returns.loc[te_start:te_end]
            if test_rets.empty:
                logger.debug(f"Test returns empty for window {tr_start}-{te_end} with params {scenario_config['strategy_params']}.")
                # For safety, if test_rets is empty, we might return NaN or a very bad score
                # For now, let's assume this means the strategy didn't trade or something went wrong.
                # Returning NaN will likely cause Optuna to prune or ignore this trial.
                if is_multi_objective:
                    return tuple([float("nan")] * len(metrics_to_optimize))
                return float("nan")


            # Check for near-zero returns to enable early stopping
            if abs(test_rets.mean()) < ZERO_RET_EPS and abs(test_rets.std()) < ZERO_RET_EPS:
                # trial is passed as an argument to this function
                if trial: # Should always be true when called from Optuna objective
                    trial.set_user_attr("zero_returns", True)
                    logger.debug(f"Trial {trial.number}, window {window_idx+1}: Marked with zero_returns.")


            bench_ser = d_slice[self.global_config["benchmark"]].loc[te_start:te_end]
            bench_period_rets = bench_ser.pct_change(fill_method=None).fillna(0)

            # Calculate metrics for the test period
            # Pass benchmark_returns to calculate_metrics
            metrics = calculate_metrics(test_rets, bench_period_rets, self.global_config["benchmark"])

            current_metrics = np.array([metrics.get(m, np.nan) for m in metrics_to_optimize], dtype=float)

            if np.isnan(current_metrics).any():
                logger.warning(f"NaN metric found for window {tr_start}-{te_end}. Params: {scenario_config['strategy_params']}")
                # If any metric is NaN, this set of parameters might be problematic for this window.
                # Depending on strategy, could skip this window or penalize. For now, let's add NaNs.

            metric_sums += np.nan_to_num(current_metrics) # Convert NaNs to 0 for sum, count separately
            if not np.isnan(current_metrics).all(): # Count if at least one metric was not NaN
                num_valid_windows +=1
                processed_steps_for_pruning += 1

            # Intermediate reporting and pruning logic
            if pruning_enabled and num_valid_windows > 0 and processed_steps_for_pruning % pruning_interval_steps == 0:
                # For MedianPruner, report a single float.
                # If multi-objective, Optuna's pruners typically use the first objective by default.
                # We use metric_sums[0] which corresponds to the first metric in metrics_to_optimize.
                intermediate_value = metric_sums[0] / num_valid_windows

                # If intermediate_value is NaN or Inf, MedianPruner might handle it,
                # but providing a very bad value consistent with optimization direction is safer.
                if not np.isfinite(intermediate_value):
                    # Access directions from trial.study. Optuna pruners typically work on the first metric.
                    first_metric_direction = trial.study.directions[0]
                    if first_metric_direction == optuna.study.StudyDirection.MAXIMIZE:
                        intermediate_value = -1e12 # A very small number for maximization
                    else: # MINIMIZE
                        intermediate_value = 1e12  # A very large number for minimization
                    logger.debug(f"Trial {trial.number}, window {window_idx+1}: intermediate metric {metrics_to_optimize[0]} was non-finite. Reporting {intermediate_value}")

                trial.report(intermediate_value, window_idx + 1) # Use window_idx + 1 as step, since steps are 1-indexed
                current_trial_number_for_logs = trial.number # Store for logging if pruned

                if trial.should_prune():
                    logger.info(f"Trial {current_trial_number_for_logs} pruned at window {window_idx + 1} (step {window_idx + 1}) with intermediate value for '{metrics_to_optimize[0]}': {intermediate_value:.4f}")
                    raise optuna.exceptions.TrialPruned()

        if num_valid_windows == 0:
            logger.warning(f"No valid windows produced results for params: {scenario_config['strategy_params']}. Returning NaN.")
            return tuple([float("nan")] * len(metrics_to_optimize)) if is_multi_objective else float("nan")

        metric_avgs = metric_sums / num_valid_windows
        metric_avgs = np.where(np.isfinite(metric_avgs), metric_avgs, float("nan")) # Ensure non-finite results are NaN

        return tuple(metric_avgs) if is_multi_objective else metric_avgs[0]

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
        plot_monte_carlo_results(
            mc_sims,
            title=f"Monte Carlo Simulation: {scenario_config['name']} (Optimized)",
            scenario_name=scenario_config.get('name'),
            params=scenario_config.get('strategy_params'),
            interactive=getattr(self.args, 'interactive', False)
        )
        logger.info("Monte Carlo simulation finished.")

    def display_results(self, data):
        logger.info("Generating performance report.")
        console = Console()
        bench_rets_full = data[self.global_config["benchmark"]].pct_change(fill_method=None).fillna(0)

        # Determine if train/test split is applicable
        first_result_key = list(self.results.keys())[0]
        train_end_date = self.results[first_result_key].get("train_end_date")

        # Prepare data for different periods
        periods_data = []
        num_trials_full = {name: res.get("num_trials_for_dsr", 1) for name, res in self.results.items()}

        if train_end_date:
            periods_data.append({
                "title": "In-Sample Performance (Net of Costs)",
                "returns_map": {name: res["returns"][res["returns"].index <= train_end_date] for name, res in self.results.items()},
                "bench_returns": bench_rets_full[bench_rets_full.index <= train_end_date],
                "num_trials_map": {name: 1 for name in self.results.keys()} # DSR not typically applied to in-sample alone
            })
            periods_data.append({
                "title": "Out-of-Sample Performance (Net of Costs)",
                "returns_map": {name: res["returns"][res["returns"].index > train_end_date] for name, res in self.results.items()},
                "bench_returns": bench_rets_full[bench_rets_full.index > train_end_date],
                "num_trials_map": num_trials_full # OOS uses the num_trials from optimization
            })

        periods_data.append({
            "title": "Full Period Performance (Net of Costs)",
            "returns_map": {name: res["returns"] for name, res in self.results.items()},
            "bench_returns": bench_rets_full,
            "num_trials_map": num_trials_full
        })

        for period_data in periods_data:
            self._generate_performance_table(
                console,
                period_data["returns_map"],
                period_data["bench_returns"],
                period_data["title"],
                period_data["num_trials_map"]
            )
        
        logger.info("Performance tables displayed.")

        # Plotting
        self._plot_performance_summary(bench_rets_full, train_end_date)

    def _generate_performance_table(self, console: Console, period_returns: Dict[str, pd.Series],
                                    bench_period_rets: pd.Series, title: str,
                                    num_trials_map: Dict[str, int]):
        """Generates and prints a Rich table for performance metrics."""
        table = Table(title=title)
        table.add_column("Metric", style="cyan", no_wrap=True)

        # Calculate benchmark metrics for the period
        bench_metrics = calculate_metrics(bench_period_rets, bench_period_rets, self.global_config["benchmark"], name=self.global_config["benchmark"], num_trials=1)

        all_period_metrics = {self.global_config["benchmark"]: bench_metrics}

        # Add columns for each strategy result present in period_returns
        for name in period_returns.keys():
            display_name = self.results[name]["display_name"] # Get full display name from original results
            table.add_column(display_name, style="magenta")
        table.add_column(self.global_config["benchmark"], style="green") # Benchmark column

        # Calculate and store metrics for each strategy for the current period
        for name, rets in period_returns.items():
            display_name = self.results[name]["display_name"]
            strategy_num_trials = num_trials_map.get(name, 1)
            metrics = calculate_metrics(rets, bench_period_rets, self.global_config["benchmark"], name=display_name, num_trials=strategy_num_trials)
            all_period_metrics[display_name] = metrics

        percentage_metrics = ["Total Return", "Ann. Return"]
        high_precision_metrics = ["ADF p-value"]

        # Populate rows for each metric
        if not bench_metrics.empty:
            for metric_name in bench_metrics.index:
                row_values = [metric_name]
                # Strategy metrics (in the order they were added to table columns)
                for strategy_name_key in period_returns.keys(): # Iterate in consistent order
                    display_name = self.results[strategy_name_key]["display_name"]
                    value = all_period_metrics[display_name].loc[metric_name]
                    if metric_name in percentage_metrics:
                        row_values.append(f"{value:.2%}")
                    elif metric_name in high_precision_metrics:
                        row_values.append(f"{value:.6f}")
                    else:
                        row_values.append(f"{value:.4f}")
                
                # Benchmark metrics last
                bench_value = bench_metrics.loc[metric_name]
                if metric_name in percentage_metrics:
                    row_values.append(f"{bench_value:.2%}")
                elif metric_name in high_precision_metrics:
                    row_values.append(f"{bench_value:.6f}")
                else:
                    row_values.append(f"{bench_value:.4f}")
                table.add_row(*row_values)
        
        console.print(table)

    def _plot_performance_summary(self, bench_rets_full: pd.Series, train_end_date: pd.Timestamp | None):
        """Generates and saves/shows the performance summary plot."""
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

        # Plot Cumulative Returns (P&L)
        ax1.set_title("Cumulative Returns (Net of Costs)", fontsize=16)
        ax1.set_ylabel("Cumulative Returns (Log Scale)", fontsize=12)
        ax1.set_yscale('log')
        
        all_cumulative_returns_plotting = []
        for name, result_data in self.results.items():
            cumulative_strategy_returns = (1 + result_data["returns"]).cumprod()
            cumulative_strategy_returns.plot(ax=ax1, label=result_data["display_name"])
            all_cumulative_returns_plotting.append(cumulative_strategy_returns)
        
        cumulative_bench_returns = (1 + bench_rets_full).cumprod()
        cumulative_bench_returns.plot(ax=ax1, label=self.global_config["benchmark"], linestyle='--')
        all_cumulative_returns_plotting.append(cumulative_bench_returns)

        if all_cumulative_returns_plotting:
            combined_cumulative_returns = pd.concat(all_cumulative_returns_plotting)
            max_val = combined_cumulative_returns.max().max() # Get scalar max from potentially multi-column Series
            min_val = combined_cumulative_returns.min().min() # Get scalar min from potentially multi-column Series
            # Ensure bottom is slightly less than 1 for log scale if min_val is positive
            ax1.set_ylim(bottom=min(0.9, min_val * 0.9) if min_val > 0 else 0.1, top=max_val * 1.1)

        ax1.legend()
        ax1.grid(True, which="both", ls="-", alpha=0.5)

        if train_end_date and (self.args.mode == "optimize" or self.args.mode == "backtest"):
            ax1.axvline(train_end_date, color='gray', linestyle='--', lw=2, label='Train/Test Split')
            ax1.legend() # Re-call legend to include the vline label if applicable

        # Plot Drawdown
        ax2.set_ylabel("Drawdown", fontsize=12)
        ax2.set_xlabel("Date", fontsize=12)

        def calculate_drawdown(returns_series):
            cumulative = (1 + returns_series).cumprod()
            peak = cumulative.expanding(min_periods=1).max()
            drawdown = (cumulative / peak) - 1
            return drawdown

        for name, result_data in self.results.items():
            drawdown = calculate_drawdown(result_data["returns"])
            drawdown.plot(ax=ax2, label=result_data["display_name"])
        
        bench_drawdown = calculate_drawdown(bench_rets_full)
        bench_drawdown.plot(ax=ax2, label=self.global_config["benchmark"], linestyle='--')

        ax2.legend()
        ax2.grid(True, which="both", ls="-", alpha=0.5)
        ax2.fill_between(bench_drawdown.index, 0, bench_drawdown, color='gray', alpha=0.2)

        plt.tight_layout()
        
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use the original scenario name from args if available, else derive from results
        base_filename = self.args.scenario_name if self.args.scenario_name else list(self.results.keys())[0]
        scenario_name_for_filename = base_filename.replace(" ", "_").replace("(", "").replace(")", "")

        filename = f"performance_summary_{scenario_name_for_filename}_{timestamp}.png"
        filepath = os.path.join(plots_dir, filename)
        
        plt.savefig(filepath)
        logger.info(f"Performance plot saved to: {filepath}")

        if getattr(self.args, 'interactive', False):
            plt.show(block=False)
            logger.info("Performance plots displayed interactively.")
        else:
            plt.close(fig)
            logger.info("Performance plots generated and saved.")

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

    # Pruning specific arguments
    parser.add_argument("--pruning-enabled", action="store_true", help="Enable trial pruning with MedianPruner. Default: False.")
    parser.add_argument("--pruning-n-startup-trials", type=int, default=5, help="MedianPruner: Number of trials to complete before pruning begins. Default: 5.")
    parser.add_argument("--pruning-n-warmup-steps", type=int, default=0, help="MedianPruner: Number of intermediate steps (walk-forward windows) to observe before pruning a trial. Default: 0.")
    parser.add_argument("--pruning-interval-steps", type=int, default=1, help="MedianPruner: Report intermediate value and check for pruning every X walk-forward windows. Default: 1.")

    # Monte Carlo specific arguments
    parser.add_argument("--mc-simulations", type=int, default=1000, help="Number of simulations for Monte Carlo analysis.")
    parser.add_argument("--mc-years", type=int, default=10, help="Number of years to project in Monte Carlo analysis.")
    parser.add_argument("--interactive", action="store_true", help="Show plots interactively (blocks execution). Default: off, only saves plots.")
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    # Reload configuration to ensure latest changes from YAML are picked up
    import src.portfolio_backtester.config_loader as config_loader_module
    config_loader_module.load_config()

    # Access the global variables directly from the reloaded module
    GLOBAL_CONFIG_RELOADED = config_loader_module.GLOBAL_CONFIG
    BACKTEST_SCENARIOS_RELOADED = config_loader_module.BACKTEST_SCENARIOS
    OPTIMIZER_PARAMETER_DEFAULTS_RELOADED = config_loader_module.OPTIMIZER_PARAMETER_DEFAULTS

    # Only require --scenario-name for 'optimize' mode
    if args.mode == "optimize" and args.scenario_name is None:
        parser.error("--scenario-name is required for 'optimize' mode.")

    if args.scenario_name is not None:
        scenario_name = args.scenario_name.strip("'\"")
        selected_scenarios = [s for s in BACKTEST_SCENARIOS_RELOADED if s["name"] == scenario_name]
        if not selected_scenarios:
            parser.error(f"Scenario '{scenario_name}' not found in BACKTEST_SCENARIOS_RELOADED.")
    else:
        selected_scenarios = BACKTEST_SCENARIOS_RELOADED

    backtester = Backtester(GLOBAL_CONFIG_RELOADED, selected_scenarios, args, random_state=args.random_seed)
    backtester.run()
