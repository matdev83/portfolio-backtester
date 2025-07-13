import setuptools
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import numpy as np
import logging
import argparse
import sys
from typing import Dict, Any
import types
import optuna
from .config_loader import GLOBAL_CONFIG, BACKTEST_SCENARIOS, OPTIMIZER_PARAMETER_DEFAULTS
from .config_initializer import populate_default_optimizations
from . import strategies
from .portfolio.position_sizer import get_position_sizer
from .portfolio.rebalancing import rebalance
# from .features.feature_helpers import get_required_features_from_scenarios # Removed
# from .feature_engineering import precompute_features # Removed
from .universe_data.spy_holdings import reset_history_cache, get_top_weight_sp500_components
from .utils import _resolve_strategy, INTERRUPTED as CENTRAL_INTERRUPTED_FLAG
from .backtester_logic.reporting import display_results
from .backtester_logic.optimization import run_optimization
from .backtester_logic.execution import run_backtest_mode, run_optimize_mode, _generate_optimization_report
# Legacy Monte Carlo mode removed - Monte Carlo is now handled in two stages:
# Stage 1: During optimization for parameter robustness
# Stage 2: During results display for strategy stress testing
from .constants import ZERO_RET_EPS
from .data_cache import get_global_cache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, global_config, scenarios, args, random_state=None):
        self.global_config = global_config
        self.global_config["optimizer_parameter_defaults"] = OPTIMIZER_PARAMETER_DEFAULTS
        self.scenarios = scenarios
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Backtester initialized with scenario strategy_params: {self.scenarios[0].get('strategy_params')}")
        populate_default_optimizations(self.scenarios, OPTIMIZER_PARAMETER_DEFAULTS)
        self.args = args
        self.data_source = self._get_data_source()
        self.results = {}
        # self.features: Dict[str, pd.DataFrame | pd.Series] | None = None # Removed
        self.monthly_data: pd.DataFrame | None = None # This is monthly closing prices
        self.daily_data_ohlc: pd.DataFrame | None = None # This will store daily OHLC data
        self.rets_full: pd.DataFrame | pd.Series | None = None
        if random_state is None:
            self.random_state = np.random.randint(0, 2**31 - 1)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"No random seed provided. Using generated seed: {self.random_state}.")
        else:
            self.random_state = random_state
        np.random.seed(self.random_state)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Numpy random seed set to {self.random_state}.")
        self.n_jobs = getattr(args, "n_jobs", 1)
        self.early_stop_patience = getattr(args, "early_stop_patience", 10)
        self.logger = logger
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Backtester initialized.")
        
        # Initialize data cache for performance optimization
        self.data_cache = get_global_cache()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Data preprocessing cache initialized")
        
        # Cache for daily date index (used by fast path)
        self._daily_index_cache = None
        
        # Initialize Monte Carlo components if enabled
        self.asset_replacement_manager = None
        self.synthetic_data_generator = None
        if self.global_config.get('enable_synthetic_data', False):
            self._initialize_monte_carlo_components()
            # Configure full data access for comprehensive statistical analysis
            if self.asset_replacement_manager is not None:
                self.asset_replacement_manager.set_full_data_source(self.data_source, self.global_config)

        # Assign methods from the logic modules to the instance
        self.run_optimization = types.MethodType(run_optimization, self)
        self.run_backtest_mode = types.MethodType(run_backtest_mode, self)
        self.run_optimize_mode = types.MethodType(run_optimize_mode, self)
        # Legacy Monte Carlo mode binding removed
        self.display_results = types.MethodType(display_results, self)
        self._generate_optimization_report = types.MethodType(_generate_optimization_report, self)
        
        # Import the deferred report generation method
        from .backtester_logic.execution import generate_deferred_report
        self.generate_deferred_report = types.MethodType(generate_deferred_report, self)

    # Duplicate __init__ method removed - keeping the original one above

    def _get_data_source(self):
        ds = self.global_config.get("data_source", "hybrid").lower()
        if ds == "stooq":
            logger.debug("Using StooqDataSource.")
            from .data_sources.stooq_data_source import StooqDataSource
            return StooqDataSource()
        elif ds == "yfinance":
            logger.debug("Using YFinanceDataSource.")
            from .data_sources.yfinance_data_source import YFinanceDataSource
            return YFinanceDataSource()
        elif ds == "hybrid":
            logger.debug("Using HybridDataSource with fail-tolerance workflow.")
            from .data_sources.hybrid_data_source import HybridDataSource
            prefer_stooq = self.global_config.get("prefer_stooq", True)
            return HybridDataSource(prefer_stooq=prefer_stooq)
        else:
            logger.error(f"Unsupported data source: {self.global_config['data_source']}")
            raise ValueError(f"Unsupported data source: {self.global_config['data_source']}")

    def _get_strategy(self, strategy_name, params):
        if strategy_name == "momentum":
            class_name = "MomentumStrategy"
        elif strategy_name == "momentum_unfiltered_atr":
            class_name = "MomentumUnfilteredAtrStrategy"
        elif strategy_name == "vams_momentum":
            class_name = "VAMSMomentumStrategy"
        elif strategy_name == "vams_no_downside":
            class_name = "VAMSNoDownsideStrategy"
        elif strategy_name == "ema_crossover":
            class_name = "EMAStrategy"
        else:
            class_name = "".join(word.capitalize() for word in strategy_name.split('_')) + "Strategy"
        
        strategy_class = getattr(strategies, class_name, None)
        
        if strategy_class:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Using {class_name} with params: {params}")
            return strategy_class(params)
        else:
            logger.error(f"Unsupported strategy: {strategy_name}")
            raise ValueError(f"Unsupported strategy: {strategy_name}")
    
    def _initialize_monte_carlo_components(self):
        """Initialize Monte Carlo components for synthetic data generation."""
        try:
            from .monte_carlo.asset_replacement import AssetReplacementManager
            
            # Initialize asset replacement manager, which in turn initializes SyntheticDataGenerator
            self.asset_replacement_manager = AssetReplacementManager(self.global_config)
            
            # Access the synthetic_data_generator instance from the asset_replacement_manager
            self.synthetic_data_generator = self.asset_replacement_manager.synthetic_generator
            
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("Monte Carlo components initialized successfully")
            
        except ImportError as e:
            if self.logger.isEnabledFor(logging.WARNING):
                self.logger.warning(f"Monte Carlo components not available: {e}")
            self.asset_replacement_manager = None
            self.synthetic_data_generator = None

    def run_scenario(
        self,
        scenario_config,
        price_data_monthly_closes: pd.DataFrame, # Monthly close prices, used for rebalance dates and sizers
        price_data_daily_ohlc: pd.DataFrame,   # Daily OHLCV data for strategies
        rets_daily: pd.DataFrame | None = None,
        # features: dict | None = None, # Removed features argument
        verbose: bool = True,
    ):
        if verbose:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Running scenario: {scenario_config['name']}")

        # Ensure rets_daily is calculated based on daily close prices if not provided
        # This requires extracting daily closes from price_data_daily_ohlc
        if rets_daily is None:
            if isinstance(price_data_daily_ohlc.columns, pd.MultiIndex) and \
               'Close' in price_data_daily_ohlc.columns.get_level_values(1):
                daily_closes_for_rets = price_data_daily_ohlc.xs('Close', level='Field', axis=1)
            elif not isinstance(price_data_daily_ohlc.columns, pd.MultiIndex):
                daily_closes_for_rets = price_data_daily_ohlc # Assume it's already close prices
            else:
                # Attempt to find 'Close' prices in a less structured MultiIndex or raise error
                try:
                    if 'Close' in price_data_daily_ohlc.columns.get_level_values(-1):
                        daily_closes_for_rets = price_data_daily_ohlc.xs('Close', level=-1, axis=1)
                    else:
                        raise ValueError("run_scenario: Could not reliably extract 'Close' prices for daily returns.")
                except Exception as e:
                    raise ValueError(f"run_scenario: Error extracting 'Close' prices for daily returns: {e}. Columns: {price_data_daily_ohlc.columns}")

            if daily_closes_for_rets.empty:
                 raise ValueError("run_scenario: Daily close prices for return calculation are empty.")
            
            # Try to get cached returns first
            window_start = daily_closes_for_rets.index.min()
            window_end = daily_closes_for_rets.index.max()
            cached_returns = self.data_cache.get_window_returns_by_dates(
                price_data_daily_ohlc, window_start, window_end
            )
            
            if cached_returns is not None:
                # Use cached returns
                rets_daily = cached_returns
                if isinstance(rets_daily, pd.Series):
                    rets_daily = rets_daily.to_frame()
            else:
                # Compute returns and cache them
                rets_daily = self.data_cache.get_cached_window_returns(
                    daily_closes_for_rets, window_start, window_end
                )
                if isinstance(rets_daily, pd.Series):
                    rets_daily = rets_daily.to_frame()

        strategy = self._get_strategy(
            scenario_config["strategy"], scenario_config["strategy_params"]
        )
        
        universe_tickers = [item[0] for item in strategy.get_universe(self.global_config)]
        benchmark_ticker = self.global_config["benchmark"]

        rebalance_dates = price_data_monthly_closes.index # Dates for generating signals (typically month-end)

        # --- Iterative Signal Generation ---
        all_monthly_weights = []

        # WFO dates from scenario_config (optional)
        wfo_start_date = pd.to_datetime(scenario_config.get("wfo_start_date", None))
        wfo_end_date = pd.to_datetime(scenario_config.get("wfo_end_date", None))

        for current_rebalance_date in rebalance_dates:
            if verbose:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Generating signals for date: {current_rebalance_date}")

            # Prepare historical data up to the current_rebalance_date
            # For assets in universe
            if isinstance(price_data_daily_ohlc.columns, pd.MultiIndex) and 'Ticker' in price_data_daily_ohlc.columns.names:
                asset_hist_data_cols = pd.MultiIndex.from_product([universe_tickers, list(price_data_daily_ohlc.columns.get_level_values('Field').unique())], names=['Ticker', 'Field'])
                asset_hist_data_cols = [col for col in asset_hist_data_cols if col in price_data_daily_ohlc.columns] # Ensure columns exist
                all_historical_data_for_strat = price_data_daily_ohlc.loc[price_data_daily_ohlc.index <= current_rebalance_date, asset_hist_data_cols]
            
                # For benchmark
                benchmark_hist_data_cols = pd.MultiIndex.from_product([[benchmark_ticker], list(price_data_daily_ohlc.columns.get_level_values('Field').unique())], names=['Ticker', 'Field'])
                benchmark_hist_data_cols = [col for col in benchmark_hist_data_cols if col in price_data_daily_ohlc.columns]
                benchmark_historical_data_for_strat = price_data_daily_ohlc.loc[price_data_daily_ohlc.index <= current_rebalance_date, benchmark_hist_data_cols]
            else: # Assuming flat columns, tickers are column names
                all_historical_data_for_strat = price_data_daily_ohlc.loc[price_data_daily_ohlc.index <= current_rebalance_date, universe_tickers]
                benchmark_historical_data_for_strat = price_data_daily_ohlc.loc[price_data_daily_ohlc.index <= current_rebalance_date, [benchmark_ticker]]


            # Call strategy's generate_signals
            # The strategy itself will handle WFO start/end date filtering if current_rebalance_date is outside.
            current_weights_df = strategy.generate_signals(
                all_historical_data=all_historical_data_for_strat,
                benchmark_historical_data=benchmark_historical_data_for_strat,
                current_date=current_rebalance_date,
                start_date=wfo_start_date,
                end_date=wfo_end_date
            )
            # current_weights_df should be a DataFrame with 1 row (current_rebalance_date) and asset columns
            all_monthly_weights.append(current_weights_df)

        if not all_monthly_weights:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(f"No signals generated for scenario {scenario_config['name']}. This might be due to WFO window or other issues.")
            # Create an empty DataFrame with expected structure for downstream processing
            signals = pd.DataFrame(columns=universe_tickers, index=rebalance_dates)
        else:
            signals = pd.concat(all_monthly_weights)
            signals = signals.reindex(rebalance_dates).fillna(0.0) # Ensure all rebalance dates are present

        if verbose:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"All signals generated for scenario: {scenario_config['name']}")
                logger.debug(f"Signals head:\n{signals.head()}")
                logger.debug(f"Signals tail:\n{signals.tail()}")
            if signals.empty:
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning("Generated signals DataFrame is empty.")

        # --- Sizing and Rebalancing (largely unchanged, uses the 'signals' DataFrame) ---
        sizer_name = scenario_config.get("position_sizer", "equal_weight")
        sizer_func = get_position_sizer(sizer_name)
        
        sizer_param_mapping = {
            "sizer_sharpe_window": "window",
            "sizer_sortino_window": "window",
            "sizer_beta_window": "window",
            "sizer_corr_window": "window",
            "sizer_dvol_window": "window",
            "sizer_target_return": "target_return",
            "sizer_max_leverage": "max_leverage",
        }

        filtered_sizer_params = {}
        strategy_params = scenario_config.get("strategy_params", {})

        window_param = None
        target_return_param = None
        max_leverage_param = None

        for key, value in strategy_params.items():
            if key in sizer_param_mapping:
                new_key = sizer_param_mapping[key]
                if new_key == "window":
                    window_param = value
                elif new_key == "target_return":
                    target_return_param = value
                elif new_key == "max_leverage":
                    max_leverage_param = value
                else:
                    filtered_sizer_params[new_key] = value

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Filtered_sizer_params: {filtered_sizer_params}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"window_param extracted: {window_param}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"target_return_param extracted: {target_return_param}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"max_leverage_param extracted: {max_leverage_param}")

        # Prepare data for sizers: they typically need monthly close prices
        strategy_monthly_closes = price_data_monthly_closes[universe_tickers]
        benchmark_monthly_closes = price_data_monthly_closes[benchmark_ticker]

        sizer_args = [signals, strategy_monthly_closes, benchmark_monthly_closes]
        
        if sizer_name == "rolling_downside_volatility":
            # This sizer needs daily prices, specifically closes for return calculation within sizer.
            if isinstance(price_data_daily_ohlc.columns, pd.MultiIndex) and \
               'Close' in price_data_daily_ohlc.columns.get_level_values(1):
                daily_closes_for_sizer = price_data_daily_ohlc.xs('Close', level='Field', axis=1)[universe_tickers]
            elif not isinstance(price_data_daily_ohlc.columns, pd.MultiIndex): # Assume it's already daily closes
                daily_closes_for_sizer = price_data_daily_ohlc[universe_tickers]
            else:
                raise ValueError("rolling_downside_volatility sizer: Could not extract daily close prices from price_data_daily_ohlc.")
            sizer_args.append(daily_closes_for_sizer)

        if sizer_name in ["rolling_sharpe", "rolling_sortino", "rolling_beta", "rolling_benchmark_corr", "rolling_downside_volatility"]:
            if window_param is None:
                raise ValueError(f"Sizer '{sizer_name}' requires a 'window' parameter, but it was not found in strategy_params.")
            sizer_args.append(window_param)
        
        if sizer_name == "rolling_sortino":
            if target_return_param is None:
                sizer_args.append(0.0)
            else:
                sizer_args.append(target_return_param)
        
        if sizer_name == "rolling_downside_volatility" and max_leverage_param is not None:
            filtered_sizer_params["max_leverage"] = max_leverage_param

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Sizer arguments prepared: {sizer_args}")
            logger.debug(f"Final keyword arguments for sizer: {filtered_sizer_params}")

        sized_signals = sizer_func(
            *sizer_args,
            **filtered_sizer_params,
        )
        if verbose:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Positions sized using {sizer_name}.")
                logger.debug(f"Sized signals head:\n{sized_signals.head()}")
                logger.debug(f"Sized signals tail:\n{signals.tail()}")

        weights_monthly = rebalance(
            sized_signals, scenario_config["rebalance_frequency"]
        )

        weights_monthly = weights_monthly.reindex(columns=universe_tickers).fillna(0.0)

        # Use the index from daily OHLC data for reindexing monthly weights to daily
        weights_daily = weights_monthly.reindex(price_data_daily_ohlc.index, method="ffill")
        weights_daily = weights_daily.shift(1).fillna(0.0) # Apply execution lag

        # Ensure daily returns are aligned with the daily OHLC data index
        # Ensure rets_daily is not None before reindexing
        if rets_daily is None:
            logger.error("rets_daily is None before reindexing in run_scenario.")
            return pd.Series(0.0, index=price_data_daily_ohlc.index) # Return zero series to avoid further errors
        aligned_rets_daily = rets_daily.reindex(price_data_daily_ohlc.index).fillna(0.0)

        # Ensure universe_tickers used for indexing aligned_rets_daily are present in its columns
        valid_universe_tickers_in_rets = [ticker for ticker in universe_tickers if ticker in aligned_rets_daily.columns]
        if len(valid_universe_tickers_in_rets) < len(universe_tickers):
            missing_tickers = set(universe_tickers) - set(valid_universe_tickers_in_rets)
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(f"Tickers {missing_tickers} not found in aligned_rets_daily columns. Portfolio calculations might be affected.")

        # Calculate gross returns using only valid tickers present in returns data
        if not valid_universe_tickers_in_rets:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning("No valid universe tickers found in daily returns. Gross portfolio returns will be zero.")
            daily_portfolio_returns_gross = pd.Series(0.0, index=weights_daily.index)
        else:
            daily_portfolio_returns_gross = (weights_daily[valid_universe_tickers_in_rets] * aligned_rets_daily[valid_universe_tickers_in_rets]).sum(axis=1)

        turnover = (weights_daily - weights_daily.shift(1)).abs().sum(axis=1).fillna(0.0)
        
        # Use realistic transaction costs for retail trading of liquid S&P 500 stocks
        from .transaction_costs import calculate_realistic_transaction_costs
        transaction_costs = calculate_realistic_transaction_costs(
            turnover=turnover,
            weights_daily=weights_daily,
            price_data=price_data_daily_ohlc,
            global_config=self.global_config
        )

        portfolio_rets_net = (daily_portfolio_returns_gross - transaction_costs).fillna(0.0)

        if verbose:
            if logger.isEnabledFor(logging.DEBUG):
                scenario_name = scenario_config['name']
                logger.debug(f"Portfolio net returns calculated for {scenario_name}. First few net returns: {portfolio_rets_net.head().to_dict()}")
                logger.debug(f"Net returns index: {portfolio_rets_net.index.min()} to {portfolio_rets_net.index.max()}")

        return portfolio_rets_net

    def _evaluate_walk_forward_fast(
        self,
        trial: Any,
        scenario_config: dict,
        windows: list,
        monthly_data_np: np.ndarray,
        daily_data_np: np.ndarray,
        rets_full_np: np.ndarray,
        metrics_to_optimize: list,
        is_multi_objective: bool,
    ) -> float | tuple[float, ...]:
        """Optional Numba-accelerated evaluation.

        The fast path is activated only when the environment variable
        ``ENABLE_NUMBA_WALKFORWARD`` is set to ``"1"``.  Otherwise we fall
        back to the legacy Pandas implementation – guaranteeing identical
        results while refactoring is in progress.
        """

        import os
        from .utils import _df_to_float32_array  # pylint: disable=import-error
        from . import numba_kernels as _nk  # pylint: disable=import-error

        use_fast = os.environ.get("ENABLE_NUMBA_WALKFORWARD", "0") == "1"

        if not use_fast:
            # Legacy path – convert back to DataFrames and delegate
            monthly_data = pd.DataFrame(monthly_data_np)
            daily_data = pd.DataFrame(daily_data_np)
            rets_full = pd.DataFrame(rets_full_np)
            return self._evaluate_params_walk_forward(
                trial,
                scenario_config,
                windows,
                monthly_data,
                daily_data,
                rets_full,
                metrics_to_optimize,
                is_multi_objective,
            )

        # ------------------------------------------------------------------
        # FAST PATH  (experimental)
        # ------------------------------------------------------------------
        try:
            # Convert to float32 NumPy arrays; assume daily_data_np already holds
            # daily *portfolio* returns per asset.  We first need aggregate
            # daily portfolio returns using equal weights (approx) as a proof-
            # of-concept.

            daily_returns = rets_full_np.astype(np.float32)

            # Aggregate across assets equally – later we will bring in sizer
            # logic inside the kernel.
            if daily_returns.size == 0:
                raise ValueError("Daily returns array is empty – cannot run fast path")

            port_rets = np.nanmean(daily_returns, axis=1).astype(np.float32)

            # Build start/end index arrays for test periods
            test_starts = np.asarray([np.searchsorted(self._daily_index_cache, te_start) for _, _, te_start, _ in windows], dtype=np.int64)
            test_ends = np.asarray([np.searchsorted(self._daily_index_cache, te_end) for _, _, _, te_end in windows], dtype=np.int64)

            metrics_mat = _nk.window_mean_std(port_rets, test_starts, test_ends)

            # Currently optimise on mean return (obj 0)
            avg_metric = float(np.nanmean(metrics_mat[:, 0]))
            return avg_metric if not is_multi_objective else (avg_metric,)

        except Exception as exc:  # noqa: BLE001
            logger.error("Fast walk-forward path failed – falling back to legacy: %s", exc)
            monthly_data = pd.DataFrame(monthly_data_np)
            daily_data = pd.DataFrame(daily_data_np)
            rets_full = pd.DataFrame(rets_full_np)
            return self._evaluate_params_walk_forward(
                trial,
                scenario_config,
                windows,
                monthly_data,
                daily_data,
                rets_full,
                metrics_to_optimize,
                is_multi_objective,
            )

    def _evaluate_params_walk_forward(
        self,
        trial: Any,
        scenario_config: dict,
        windows: list,
        monthly_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        rets_full: pd.DataFrame,
        metrics_to_optimize: list,
        is_multi_objective: bool,
    ) -> float | tuple[float, ...]:
        """
        Evaluate parameters using walk-forward optimization.
        
        Args:
            trial: Optuna trial object
            scenario_config: Configuration for the scenario
            windows: List of (tr_start, tr_end, te_start, te_end) tuples
            monthly_data: Monthly price data
            daily_data: Daily OHLC data
            rets_full: Full period returns data
            metrics_to_optimize: List of metrics to optimize
            is_multi_objective: Whether this is multi-objective optimization
            
        Returns:
            Single objective value or tuple of objective values
        """
        from .reporting.performance_metrics import calculate_metrics
        
        # Pre-compute returns for all windows if not already cached
        if not hasattr(self, '_windows_precomputed') or not self._windows_precomputed:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Pre-computing returns for all windows")
            self.data_cache.precompute_window_returns(daily_data, windows)
            self._windows_precomputed = True
        
        # Import here to avoid circular imports
        from .backtester_logic.optimization import _global_progress_tracker
        from .reporting.performance_metrics import calculate_metrics
        from .utils import calculate_stability_metrics, INTERRUPTED as CENTRAL_INTERRUPTED_FLAG
        
        metric_values_per_objective = [[] for _ in metrics_to_optimize]
        processed_steps_for_pruning = 0

        pruning_enabled = getattr(self.args, "pruning_enabled", False)
        pruning_interval_steps = getattr(self.args, "pruning_interval_steps", 1)

        # STAGE 1 MONTE CARLO: Lightweight robustness testing during optimization
        # Initialize Monte Carlo asset replacement manager for trial-level robustness
        monte_carlo_config = self.global_config.get('monte_carlo_config', {})
        asset_replacement_manager = None
        trial_synthetic_data = None
        
        # Check if Monte Carlo is enabled and not explicitly disabled during optimization
        mc_enabled = monte_carlo_config.get('enable_synthetic_data', False)
        mc_during_optimization = monte_carlo_config.get('enable_during_optimization', True)
        
        # SMART FEATURE FLAGS: Adaptive Monte Carlo based on optimization mode
        optimization_mode = monte_carlo_config.get('optimization_mode', 'balanced')
        trial_threshold = self._get_monte_carlo_trial_threshold(optimization_mode)
        
        # Only enable Monte Carlo after sufficient trials for adaptive performance
        trial_number = getattr(trial, 'number', 0) if trial else 0
        mc_adaptive_enabled = mc_enabled and mc_during_optimization and (trial_number >= trial_threshold)
        
        if mc_adaptive_enabled:
            from .monte_carlo.asset_replacement import AssetReplacementManager
            
            # Create optimized config for Stage 1 MC (faster generation during optimization)
            stage1_config = monte_carlo_config.copy()
            stage1_config['stage1_optimization'] = True  # Flag for optimization mode
            stage1_config['replacement_percentage'] = monte_carlo_config.get('replacement_percentage', 0.05)  # Keep lightweight
            
            # Override generation settings for speed
            stage1_config['generation_config'] = {
                'buffer_multiplier': 1.0,  # No buffer for speed
                'max_attempts': 1,         # Single attempt only
                'validation_tolerance': 1.0  # Very lenient validation
            }
            
            # Disable validation during optimization for speed
            stage1_config['validation_config'] = {
                'enable_validation': False
            }
            
            asset_replacement_manager = AssetReplacementManager(stage1_config)
            # Configure full data access for comprehensive statistical analysis
            if asset_replacement_manager is not None:
                asset_replacement_manager.set_full_data_source(self.data_source, self.global_config)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Stage 1 MC: Lightweight synthetic data enabled for optimization robustness (mode: {optimization_mode}, trial: {trial_number})")
        elif mc_enabled and not mc_during_optimization:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Stage 1 MC: Disabled during optimization for performance")
        elif mc_enabled and trial_number < trial_threshold:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Stage 1 MC: Waiting for trial {trial_threshold} before enabling (current: {trial_number}, mode: {optimization_mode})")
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Stage 1 MC: Monte Carlo disabled in configuration")

        # Store all window returns for this trial to create full P&L curve
        all_window_returns = []
        
        # Update progress description to show current trial
        if _global_progress_tracker and trial and hasattr(trial, 'number'):
            trial_num = trial.number + 1  # Optuna uses 0-based indexing
            total_trials = _global_progress_tracker['total_trials']
            _global_progress_tracker['progress'].update(
                _global_progress_tracker['task'], 
                description=f"[cyan]Trial {trial_num}/{total_trials} running ({len(windows)} windows/trial)..."
            )
        
        # Generate lightweight synthetic data ONCE per trial for Stage 1 MC
        # PERFORMANCE FIX: Only do expensive data preparation if Monte Carlo is actually enabled
        replacement_info = None
        if asset_replacement_manager is not None:
            try:
                # Get universe from scenario config or global config
                universe = scenario_config.get('universe', self.global_config.get('universe', []))
                
                # Find the overall data range across all windows
                all_start_dates = [tr_start for tr_start, _, _, _ in windows]
                all_end_dates = [te_end for _, _, _, te_end in windows]
                overall_start = min(all_start_dates)
                overall_end = max(all_end_dates)
                
                # Get full data range for synthetic generation
                full_data_slice = daily_data.loc[overall_start:overall_end]
                
                # Convert to dictionary format expected by asset replacement
                daily_data_dict = {}
                
                if isinstance(full_data_slice.columns, pd.MultiIndex):
                    # Handle MultiIndex columns (Ticker, Field)
                    for ticker in universe:
                        ticker_data = full_data_slice.xs(ticker, level='Ticker', axis=1, drop_level=True)
                        if not ticker_data.empty:
                            daily_data_dict[ticker] = ticker_data
                else:
                    # Handle simple column structure
                    for ticker in universe:
                        if ticker in full_data_slice.columns:
                            # Create OHLC structure from single price column
                            ticker_data = pd.DataFrame({
                                'Open': full_data_slice[ticker],
                                'High': full_data_slice[ticker],
                                'Low': full_data_slice[ticker],
                                'Close': full_data_slice[ticker]
                            })
                            daily_data_dict[ticker] = ticker_data
                
                # Generate random seed for this trial
                trial_seed = None
                if monte_carlo_config.get('random_seed') is not None:
                    trial_seed = monte_carlo_config['random_seed'] + getattr(trial, 'number', 0)
                
                # Stage 1 MC: Generate lightweight synthetic data for robustness testing
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Stage 1 MC: Generating lightweight synthetic data for trial {getattr(trial, 'number', 0)}")
                trial_synthetic_data, replacement_info = asset_replacement_manager.create_monte_carlo_dataset(
                    original_data=daily_data_dict,
                    universe=universe,
                    test_start=overall_start,
                    test_end=overall_end,
                    run_id=f"trial_{getattr(trial, 'number', 0)}",
                    random_seed=trial_seed
                )
                
                if replacement_info and replacement_info.selected_assets:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Stage 1 MC: Trial {getattr(trial, 'number', 0)} using synthetic data for {len(replacement_info.selected_assets)} assets")
                
            except Exception as e:
                logger.error(f"Stage 1 MC: Failed to generate synthetic data for trial {getattr(trial, 'number', 0)}: {e}")
                trial_synthetic_data = None
                replacement_info = None

        for window_idx, (tr_start, tr_end, te_start, te_end) in enumerate(windows):
            if CENTRAL_INTERRUPTED_FLAG:
                self.logger.warning("Evaluation interrupted by user via central flag.")
                break

            m_slice = monthly_data.loc[tr_start:tr_end]
            d_slice = daily_data.loc[tr_start:te_end]
            r_slice = rets_full.loc[tr_start:te_end]

            # d_slice is the original daily OHLC data for the current window
            current_daily_data_ohlc = d_slice  # Use view by default for performance

            # Stage 1 MC: Apply lightweight synthetic data to test period only (for robustness)
            if mc_adaptive_enabled and trial_synthetic_data is not None and replacement_info is not None:
                # PERFORMANCE: Only copy when we actually need to modify data
                # This prevents modifications from leaking into subsequent walk-forward windows.
                current_daily_data_ohlc = d_slice.copy()
                
                # Iterate through selected assets and replace their test period data
                for asset in replacement_info.selected_assets:
                    if asset in trial_synthetic_data:
                        synthetic_ohlc_for_asset = trial_synthetic_data[asset]
                        
                        # Get the test period slice for this asset from the synthetic data
                        window_synthetic_ohlc = synthetic_ohlc_for_asset.loc[te_start:te_end]
                        
                        if not window_synthetic_ohlc.empty:
                            # Replace data in current_daily_data_ohlc for the test period
                            if isinstance(current_daily_data_ohlc.columns, pd.MultiIndex):
                                # MultiIndex: (Ticker, Field)
                                for field in window_synthetic_ohlc.columns:
                                    if (asset, field) in current_daily_data_ohlc.columns:
                                        current_daily_data_ohlc.loc[window_synthetic_ohlc.index, (asset, field)] = window_synthetic_ohlc[field]
                            else:
                                # Flat columns: Ticker_Field
                                for field in window_synthetic_ohlc.columns:
                                    col_name = f"{asset}_{field}"
                                    if col_name in current_daily_data_ohlc.columns:
                                        current_daily_data_ohlc.loc[window_synthetic_ohlc.index, col_name] = window_synthetic_ohlc[field]
                                    elif field == 'Close' and asset in current_daily_data_ohlc.columns: # Fallback for single column per ticker
                                        current_daily_data_ohlc.loc[window_synthetic_ohlc.index, asset] = window_synthetic_ohlc[field]
                                    else:
                                        if logger.isEnabledFor(logging.WARNING):
                                            logger.warning(f"Could not find column {col_name} or {asset} for synthetic data replacement in current_daily_data_ohlc.")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Stage 1 MC: Applied synthetic data to current_daily_data_ohlc for window {window_idx+1}")
            
            # Pass the potentially modified daily OHLC data to run_scenario
            # Use cached returns if available to avoid recomputation
            cached_window_returns = self.data_cache.get_window_returns_by_dates(
                current_daily_data_ohlc, tr_start, te_end
            )
            
            window_returns = self.run_scenario(
                scenario_config, 
                m_slice, 
                current_daily_data_ohlc, 
                rets_daily=cached_window_returns, 
                verbose=False
            )

            if window_returns is None or window_returns.empty:
                if self.logger.isEnabledFor(logging.WARNING):
                    self.logger.warning(f"No returns generated for window {tr_start}-{te_end}. Skipping.")
                for i in range(len(metrics_to_optimize)):
                    metric_values_per_objective[i].append(np.nan)
                if _global_progress_tracker:
                    _global_progress_tracker['progress'].update(_global_progress_tracker['task'], advance=1)
                continue

            all_window_returns.append(window_returns)

            test_rets = window_returns.loc[te_start:te_end]
            if test_rets.empty:
                if self.logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Test returns empty for window {tr_start}-{te_end} with params {scenario_config['strategy_params']}.")
                if _global_progress_tracker:
                    _global_progress_tracker['progress'].update(_global_progress_tracker['task'], advance=1)
                if is_multi_objective:
                    return tuple([float("nan")] * len(metrics_to_optimize))
                return float("nan")

            if abs(test_rets.mean()) < 1e-9 and abs(test_rets.std()) < 1e-9:
                if trial and hasattr(trial, "set_user_attr"):
                    trial.set_user_attr("zero_returns", True)
                    if hasattr(trial, "number"):
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Trial {trial.number}, window {window_idx+1}: Marked with zero_returns.")

            bench_ser = d_slice[self.global_config["benchmark"]].loc[te_start:te_end]
            bench_period_rets = bench_ser.pct_change(fill_method=None).fillna(0)
            metrics = calculate_metrics(test_rets, bench_period_rets, self.global_config["benchmark"])
            current_metrics = np.array([metrics.get(m, np.nan) for m in metrics_to_optimize], dtype=float)

            for i, metric_val in enumerate(current_metrics):
                metric_values_per_objective[i].append(metric_val)

            # Update progress bar after processing this window
            if _global_progress_tracker:
                _global_progress_tracker['progress'].update(_global_progress_tracker['task'], advance=1)

            # Handle pruning if enabled
            processed_steps_for_pruning += 1
            if pruning_enabled and processed_steps_for_pruning % pruning_interval_steps == 0:
                if trial and hasattr(trial, "should_prune"):
                    current_score = np.nanmean(metric_values_per_objective[0])
                    if hasattr(trial, "report"):
                        trial.report(current_score, processed_steps_for_pruning)
                    if trial.should_prune():
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Trial {getattr(trial, 'number', 'N/A')} pruned at step {processed_steps_for_pruning}")
                        raise optuna.exceptions.TrialPruned()

        # Calculate and store stability metrics for this trial
        if trial and hasattr(trial, "set_user_attr"):
            if all_window_returns:
                try:
                    stability_metrics = calculate_stability_metrics(metric_values_per_objective, metrics_to_optimize, self.global_config)
                    trial.set_user_attr("stability_metrics", stability_metrics)
                    if hasattr(trial, "number"):
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Trial {trial.number} stability metrics: {stability_metrics}")
                except Exception as e:
                    if logger.isEnabledFor(logging.WARNING):
                        logger.warning(f"Failed to calculate stability metrics for trial {getattr(trial, 'number', 'N/A')}: {e}")
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Trial {getattr(trial, 'number', 'N/A')} has no window returns for stability metrics")

        # Store Monte Carlo replacement statistics if available
        if asset_replacement_manager is not None and trial and hasattr(trial, "set_user_attr"):
            replacement_stats = asset_replacement_manager.get_replacement_statistics()
            trial.set_user_attr("monte_carlo_replacement_stats", replacement_stats)

        metric_avgs = [np.nanmean(values) if not all(np.isnan(values)) else np.nan for values in metric_values_per_objective]

        if all(np.isnan(np.array(metric_avgs))):
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(f"No valid windows produced results for params: {scenario_config['strategy_params']}. Returning NaN.")
            full_pnl_returns = pd.Series(dtype=float)
            if is_multi_objective:
                return tuple([float("nan")] * len(metrics_to_optimize))
            else:
                return float("nan")

        # Concatenate all window returns into a single series for full P&L curve
        full_pnl_returns = pd.concat(all_window_returns) if all_window_returns else pd.Series(dtype=float)
        # Ensure the index is unique before saving to JSON
        full_pnl_returns = full_pnl_returns[~full_pnl_returns.index.duplicated(keep='first')].sort_index()

        # Store full_pnl_returns as a user attribute for later analysis
        if trial and hasattr(trial, "set_user_attr"):
            trial.set_user_attr("full_pnl_returns", full_pnl_returns.to_json())

        if is_multi_objective:
            return tuple(float(v) for v in metric_avgs)
        else:
            return float(metric_avgs[0])
    
    def _evaluate_single_window(self, window_config, scenario_config, shared_data):
        """
        Evaluate a single WFO window. This method is designed to be called by parallel workers.
        
        Args:
            window_config: Dictionary containing window configuration
            scenario_config: Scenario configuration
            shared_data: Shared data needed for evaluation
            
        Returns:
            Tuple of (metrics_array, window_returns)
        """
        from .reporting.performance_metrics import calculate_metrics
        
        # Extract window parameters
        window_idx = window_config['window_idx']
        tr_start = window_config['tr_start']
        tr_end = window_config['tr_end']
        te_start = window_config['te_start']
        te_end = window_config['te_end']
        
        # Extract shared data
        monthly_data = shared_data['monthly_data']
        daily_data = shared_data['daily_data']
        rets_full = shared_data['rets_full']
        trial_synthetic_data = shared_data.get('trial_synthetic_data')
        replacement_info = shared_data.get('replacement_info')
        mc_adaptive_enabled = shared_data.get('mc_adaptive_enabled', False)
        metrics_to_optimize = shared_data['metrics_to_optimize']
        global_config = shared_data['global_config']
        
        try:
            m_slice = monthly_data.loc[tr_start:tr_end]
            d_slice = daily_data.loc[tr_start:te_end]
            r_slice = rets_full.loc[tr_start:te_end]

            # d_slice is the original daily OHLC data for the current window
            current_daily_data_ohlc = d_slice  # Avoid unnecessary copy for performance
            
            # Stage 1 MC: Apply lightweight synthetic data to test period only (for robustness)
            if mc_adaptive_enabled and trial_synthetic_data is not None and replacement_info is not None:
                # Need to copy only if we're modifying the data
                current_daily_data_ohlc = d_slice.copy()
                
                # Iterate through selected assets and replace their test period data
                for asset in replacement_info.selected_assets:
                    if asset in trial_synthetic_data:
                        synthetic_ohlc_for_asset = trial_synthetic_data[asset]
                        
                        # Get the test period slice for this asset from the synthetic data
                        window_synthetic_ohlc = synthetic_ohlc_for_asset.loc[te_start:te_end]
                        
                        if not window_synthetic_ohlc.empty:
                            # Replace data in current_daily_data_ohlc for the test period
                            if isinstance(current_daily_data_ohlc.columns, pd.MultiIndex):
                                # MultiIndex: (Ticker, Field)
                                for field in window_synthetic_ohlc.columns:
                                    if (asset, field) in current_daily_data_ohlc.columns:
                                        current_daily_data_ohlc.loc[window_synthetic_ohlc.index, (asset, field)] = window_synthetic_ohlc[field]
                            else:
                                # Flat columns: Ticker_Field
                                for field in window_synthetic_ohlc.columns:
                                    col_name = f"{asset}_{field}"
                                    if col_name in current_daily_data_ohlc.columns:
                                        current_daily_data_ohlc.loc[window_synthetic_ohlc.index, col_name] = window_synthetic_ohlc[field]
                                    elif field == 'Close' and asset in current_daily_data_ohlc.columns:
                                        current_daily_data_ohlc.loc[window_synthetic_ohlc.index, asset] = window_synthetic_ohlc[field]
            
            # Pass the potentially modified daily OHLC data to run_scenario
            window_returns = self.run_scenario(scenario_config, m_slice, current_daily_data_ohlc, rets_daily=None, verbose=False)

            if window_returns is None or window_returns.empty:
                return [np.nan] * len(metrics_to_optimize), pd.Series(dtype=float)

            test_rets = window_returns.loc[te_start:te_end]
            if test_rets.empty:
                return [np.nan] * len(metrics_to_optimize), pd.Series(dtype=float)

            bench_ser = d_slice[global_config["benchmark"]].loc[te_start:te_end]
            bench_period_rets = bench_ser.pct_change(fill_method=None).fillna(0)
            metrics = calculate_metrics(test_rets, bench_period_rets, global_config["benchmark"])
            current_metrics = [metrics.get(m, np.nan) for m in metrics_to_optimize]

            return current_metrics, window_returns
            
        except Exception as e:
            logger.error(f"Error evaluating window {window_idx}: {e}")
            return [np.nan] * len(metrics_to_optimize), pd.Series(dtype=float)

    def _get_monte_carlo_trial_threshold(self, optimization_mode):
        """
        Get the trial threshold for enabling Monte Carlo based on optimization mode.
        
        Args:
        
        Returns:
            Trial number threshold for enabling Monte Carlo
        """
        thresholds = {
            'fast': 20,        # Enable MC after 20 trials for fast mode
            'balanced': 10,    # Enable MC after 10 trials for balanced mode  
            'comprehensive': 5 # Enable MC after 5 trials for comprehensive mode
        }
        return thresholds.get(optimization_mode, 10)  # Default to balanced

    def evaluate_fast(
        self,
        trial: Any,
        scenario_config: dict,
        windows: list,
        monthly_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        rets_full: pd.DataFrame,
        metrics_to_optimize: list,
        is_multi_objective: bool,
    ) -> tuple[float | tuple[float, ...], pd.Series]:
        """Public interface for fast evaluation with Monte-Carlo safeguards.
        
        This method ensures Monte-Carlo synthetic data injection (if enabled)
        occurs before NumPy conversion, then delegates to the fast kernel.
        Falls back to legacy path if any unsupported features are active.
        
        Returns:
            Tuple of (objective_value, full_pnl_returns) to match legacy interface
        """
        import os
        from .utils import _df_to_float32_array
        
        # Check if fast path is enabled
        use_fast = os.environ.get("ENABLE_NUMBA_WALKFORWARD", "0") == "1"
        if not use_fast:
            # Legacy path returns (objective_value, full_pnl_returns)
            objective_value = self._evaluate_params_walk_forward(
                trial, scenario_config, windows, monthly_data, daily_data, rets_full,
                metrics_to_optimize, is_multi_objective
            )
            # Extract full_pnl_returns from trial user_attrs if available
            full_pnl_returns = pd.Series(dtype=float)
            if trial and hasattr(trial, 'user_attrs') and 'full_pnl_returns' in trial.user_attrs:
                pnl_dict = trial.user_attrs['full_pnl_returns']
                if isinstance(pnl_dict, dict):
                    full_pnl_returns = pd.Series(pnl_dict)
                    full_pnl_returns.index = pd.to_datetime(full_pnl_returns.index)
            return objective_value, full_pnl_returns
        
        # Check for unsupported features that force legacy path
        monte_carlo_config = self.global_config.get('monte_carlo_config', {})
        mc_enabled = monte_carlo_config.get('enable_synthetic_data', False)
        mc_during_optimization = monte_carlo_config.get('enable_during_optimization', True)
        
        robustness_config = self.global_config.get("wfo_robustness_config", {})
        window_randomization = robustness_config.get("enable_window_randomization", False)
        start_randomization = robustness_config.get("enable_start_date_randomization", False)
        
        # Fall back if Monte-Carlo or randomization features are active
        if (mc_enabled and mc_during_optimization) or window_randomization or start_randomization:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Fast path disabled due to Monte-Carlo or randomization features - using legacy path")
            objective_value = self._evaluate_params_walk_forward(
                trial, scenario_config, windows, monthly_data, daily_data, rets_full,
                metrics_to_optimize, is_multi_objective
            )
            # Extract full_pnl_returns from trial user_attrs if available
            full_pnl_returns = pd.Series(dtype=float)
            if trial and hasattr(trial, 'user_attrs') and 'full_pnl_returns' in trial.user_attrs:
                pnl_dict = trial.user_attrs['full_pnl_returns']
                if isinstance(pnl_dict, dict):
                    full_pnl_returns = pd.Series(pnl_dict)
                    full_pnl_returns.index = pd.to_datetime(full_pnl_returns.index)
            return objective_value, full_pnl_returns
        
        try:
            # Cache daily index for fast lookups
            if self._daily_index_cache is None or not daily_data.index.equals(pd.Index(self._daily_index_cache)):
                self._daily_index_cache = daily_data.index.to_numpy()
            
            # Convert DataFrames to NumPy arrays
            monthly_data_np, _ = _df_to_float32_array(monthly_data)
            daily_data_np, _ = _df_to_float32_array(daily_data)
            rets_full_np, _ = _df_to_float32_array(rets_full)
            
            objective_value = self._evaluate_walk_forward_fast(
                trial, scenario_config, windows, monthly_data_np, daily_data_np, rets_full_np,
                metrics_to_optimize, is_multi_objective
            )
            
            # For now, return empty full_pnl_returns (fast path doesn't compute this yet)
            full_pnl_returns = pd.Series(dtype=float)
            
            return objective_value, full_pnl_returns
            
        except Exception as exc:
            logger.error("Fast evaluation failed - falling back to legacy: %s", exc)
            objective_value = self._evaluate_params_walk_forward(
                trial, scenario_config, windows, monthly_data, daily_data, rets_full,
                metrics_to_optimize, is_multi_objective
            )
            # Extract full_pnl_returns from trial user_attrs if available
            full_pnl_returns = pd.Series(dtype=float)
            if trial and hasattr(trial, 'user_attrs') and 'full_pnl_returns' in trial.user_attrs:
                pnl_dict = trial.user_attrs['full_pnl_returns']
                if isinstance(pnl_dict, dict):
                    full_pnl_returns = pd.Series(pnl_dict)
                    full_pnl_returns.index = pd.to_datetime(full_pnl_returns.index)
            return objective_value, full_pnl_returns

    def run(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Starting backtest data retrieval.")
        daily_data = self.data_source.get_data(
            tickers=self.global_config["universe"] + [self.global_config["benchmark"]],
            start_date=self.global_config["start_date"],
            end_date=self.global_config["end_date"]
        )

        if daily_data is None or daily_data.empty:
            logger.critical("No data fetched from data source. Aborting backtest run.")
            raise ValueError("daily_data is None after data source fetch. Cannot proceed.")

        daily_data.dropna(how="all", inplace=True)

        if isinstance(daily_data.columns, pd.MultiIndex) and daily_data.columns.names[0] != 'Ticker':
            daily_data_std_format = daily_data.stack(level=1).unstack(level=0)
        else:
            daily_data_std_format = daily_data

        # Ensure self.daily_data_ohlc is set up correctly.
        # It should contain daily OHLCV data for all universe tickers and the benchmark.
        # The existing logic for daily_data_std_format seems to prepare this.
        # Ensure daily_data_std_format is a DataFrame before assigning
        if isinstance(daily_data_std_format, pd.Series):
            self.daily_data_ohlc = daily_data_std_format.to_frame()
        else:
            self.daily_data_ohlc = daily_data_std_format
        
        if self.daily_data_ohlc is not None: # Add explicit check for Pylance
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Shape of self.daily_data_ohlc: {self.daily_data_ohlc.shape}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Columns of self.daily_data_ohlc: {self.daily_data_ohlc.columns}")


        # The section for preparing 'monthly_data_for_features' is no longer needed
        # as features are computed within strategies using daily_data_ohlc.
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Feature pre-computation step removed. Features will be calculated within strategies.")

        # Extract daily closes for return calculations and for monthly resampling if needed by other parts
        daily_closes = None
        if self.daily_data_ohlc is not None: # Ensure self.daily_data_ohlc is not None
            if isinstance(self.daily_data_ohlc.columns, pd.MultiIndex) and \
               'Close' in self.daily_data_ohlc.columns.get_level_values(1):
                # Assuming 'Ticker' is level 0 and 'Field' is level 1
                daily_closes = self.daily_data_ohlc.xs('Close', level='Field', axis=1)
            elif not isinstance(self.daily_data_ohlc.columns, pd.MultiIndex):
                # If daily_data_ohlc is just close prices (older format or specific data source output)
                daily_closes = self.daily_data_ohlc
            else:
                # Attempt to find 'Close' prices in a less structured MultiIndex or raise error
                try:
                    if 'Close' in self.daily_data_ohlc.columns.get_level_values(-1): # Check last level for 'Close'
                        daily_closes = self.daily_data_ohlc.xs('Close', level=-1, axis=1)
                    else:
                        raise ValueError("Could not reliably extract 'Close' prices from self.daily_data_ohlc due to unrecognized column structure.")
                except Exception as e:
                     raise ValueError(f"Error extracting 'Close' prices from self.daily_data_ohlc: {e}. Columns: {self.daily_data_ohlc.columns}")

        if daily_closes is None or daily_closes.empty:
            raise ValueError("Daily close prices could not be extracted or are empty.")

        if isinstance(daily_closes, pd.Series):
            daily_closes = daily_closes.to_frame()

        # monthly_closes will be used for determining rebalance dates and for sizers if they need monthly frequency.
        monthly_closes = daily_closes.resample("BME").last()
        # Ensure monthly_closes is a DataFrame before assigning
        self.monthly_data = monthly_closes.to_frame() if isinstance(monthly_closes, pd.Series) else monthly_closes # Store monthly closing prices

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Backtest data retrieved and prepared (daily OHLC, monthly closes).")

        # Removed strategy_registry and get_required_features_from_scenarios
        # Removed benchmark_monthly_closes (strategies will use benchmark_historical_data from daily_data_ohlc)
        # Removed precompute_features call and self.features initialization

        # Use data cache for expensive return calculations
        rets_full = self.data_cache.get_cached_returns(daily_closes, "full_period_returns")
        # Ensure rets_full is a DataFrame, even if it originates from a Series
        self.rets_full = rets_full.to_frame() if isinstance(rets_full, pd.Series) else rets_full # These are daily returns based on daily_closes

        # The arguments to run_optimize_mode, run_backtest_mode
        # might need adjustment if they directly used self.features or specific monthly data forms
        # that are no longer created.
        # `monthly_closes` is still available as self.monthly_data.
        # `daily_closes` is available.
        # `rets_full` (daily returns) is available.
        # `self.daily_data_ohlc` (full daily OHLCV) is the primary data source for strategies now.

        if self.args.mode == "optimize":
            # Pass self.daily_data_ohlc for strategies, monthly_closes for other potential uses by optimizer/scenario runner
            self.run_optimize_mode(self.scenarios[0], self.monthly_data, self.daily_data_ohlc, rets_full)
        elif self.args.mode == "backtest":
            self.run_backtest_mode(self.scenarios[0], self.monthly_data, self.daily_data_ohlc, rets_full)
            # Note: Monte Carlo robustness analysis (Stage 2) is automatically performed 
            # during results display if optimization reports are enabled
        
        # Moved logic from the module level into the class method's finally block
        # This ensures it always runs after backtest/optimization, or upon interruption.
        if CENTRAL_INTERRUPTED_FLAG:
            self.logger.warning("Operation interrupted by user. Skipping final results display and plotting.")
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("All scenarios completed. Displaying results.")
            # Display results and generate deferred reports if needed
            if self.args.mode == "backtest":
                self.display_results(daily_data) # daily_data is available here
            else:
                # For optimization mode, generate deferred reports if enabled
                try:
                    self.generate_deferred_report()
                except Exception as e:
                    self.logger.warning(f"Failed to generate deferred report: {e}")
                
                # Then display results
                self.display_results(self.daily_data_ohlc)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Optimization mode completed. Reports generated.")
        
# The following code block was incorrectly placed inside the class in the previous attempt.
# It should remain at the module level.
from .utils import register_signal_handler as register_central_signal_handler

if __name__ == "__main__":
    register_central_signal_handler()

    parser = argparse.ArgumentParser(description="Run portfolio backtester.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level.")
    parser.add_argument("--mode", type=str, required=True, choices=["backtest", "optimize"], help="Mode to run the backtester in.")
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
    parser.add_argument("--optimizer", type=str, default="optuna", choices=["optuna", "genetic"],
                        help="Optimizer to use ('optuna' or 'genetic'). Default: optuna.")
    parser.add_argument("--pruning-enabled", action="store_true", help="Enable trial pruning with MedianPruner (Optuna only). Default: False.")
    parser.add_argument("--pruning-n-startup-trials", type=int, default=5, help="MedianPruner: Number of trials to complete before pruning begins. Default: 5.")
    parser.add_argument("--pruning-n-warmup-steps", type=int, default=0, help="MedianPruner: Number of intermediate steps (walk-forward windows) to observe before pruning a trial. Default: 0.")
    parser.add_argument("--pruning-interval-steps", type=int, default=1, help="MedianPruner: Report intermediate value and check for pruning every X walk-forward windows. Default: 1.")
    parser.add_argument("--mc-simulations", type=int, default=1000, help="Number of simulations for Monte Carlo analysis.")
    parser.add_argument("--mc-years", type=int, default=10, help="Number of years to project in Monte Carlo analysis.")
    parser.add_argument("--interactive", action="store_true", help="Show plots interactively (blocks execution). Default: off, only saves plots.")
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    import src.portfolio_backtester.config_loader as config_loader_module
    config_loader_module.load_config()

    GLOBAL_CONFIG_RELOADED = config_loader_module.GLOBAL_CONFIG
    BACKTEST_SCENARIOS_RELOADED = config_loader_module.BACKTEST_SCENARIOS
    OPTIMIZER_PARAMETER_DEFAULTS_RELOADED = config_loader_module.OPTIMIZER_PARAMETER_DEFAULTS

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
    try:
        backtester.run()
    except Exception as e:
        logger.error(f"An unhandled exception occurred during backtester run: {e}", exc_info=True)
        if CENTRAL_INTERRUPTED_FLAG:
            logger.info("The error occurred after an interruption signal was received.")
            sys.exit(130)
        sys.exit(1)
    finally:
        if CENTRAL_INTERRUPTED_FLAG:
            logger.info("Backtester run finished or was terminated due to user interruption.")
            sys.exit(130)
        else:
            logger.info("Backtester run completed.")
            # Moved from outside the try-finally block
            if CENTRAL_INTERRUPTED_FLAG:
                logger.warning("Operation interrupted by user. Skipping final results display and plotting.")
            else:
                logger.info("All scenarios completed. Displaying results.")
                # Display results for both backtest and optimization modes
                backtester.display_results(backtester.daily_data_ohlc)
                if args.mode == "optimize":
                    logger.info("Optimization mode completed. Performance tables above show results with optimal parameters.")
