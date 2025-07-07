import setuptools
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import numpy as np
import logging
import argparse
import sys
from typing import Dict
import types
from .config_loader import GLOBAL_CONFIG, BACKTEST_SCENARIOS, OPTIMIZER_PARAMETER_DEFAULTS
from .config_initializer import populate_default_optimizations
from . import strategies
from .portfolio.position_sizer import get_position_sizer
from .portfolio.rebalancing import rebalance
# from .features.feature_helpers import get_required_features_from_scenarios # Removed
# from .feature_engineering import precompute_features # Removed
from .spy_holdings import reset_history_cache, get_top_weight_sp500_components
from .utils import _resolve_strategy, INTERRUPTED as CENTRAL_INTERRUPTED_FLAG
from .backtester_logic.reporting import display_results
from .backtester_logic.optimization import run_optimization
from .backtester_logic.execution import run_backtest_mode, run_optimize_mode
from .backtester_logic.monte_carlo import run_monte_carlo_mode
from .constants import ZERO_RET_EPS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, global_config, scenarios, args, random_state=None):
        self.global_config = global_config
        self.global_config["optimizer_parameter_defaults"] = OPTIMIZER_PARAMETER_DEFAULTS
        self.scenarios = scenarios
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
            logger.info(f"No random seed provided. Using generated seed: {self.random_state}.")
        else:
            self.random_state = random_state
        np.random.seed(self.random_state)
        logger.info(f"Numpy random seed set to {self.random_state}.")
        self.n_jobs = getattr(args, "n_jobs", 1)
        self.early_stop_patience = getattr(args, "early_stop_patience", 10)
        self.logger = logger
        logger.info("Backtester initialized.")

        # Assign methods from the logic modules to the instance
        self.run_optimization = types.MethodType(run_optimization, self)
        self.run_backtest_mode = types.MethodType(run_backtest_mode, self)
        self.run_optimize_mode = types.MethodType(run_optimize_mode, self)
        self.run_monte_carlo_mode = types.MethodType(run_monte_carlo_mode, self)
        self.display_results = types.MethodType(display_results, self)

    def _get_data_source(self):
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
        if strategy_name == "momentum":
            class_name = "MomentumStrategy"
        elif strategy_name == "vams_momentum":
            class_name = "VAMSMomentumStrategy"
        elif strategy_name == "vams_no_downside":
            class_name = "VAMSNoDownsideStrategy"
        else:
            class_name = "".join(word.capitalize() for word in strategy_name.split('_')) + "Strategy"
        
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
        price_data_monthly_closes: pd.DataFrame, # Monthly close prices, used for rebalance dates and sizers
        price_data_daily_ohlc: pd.DataFrame,   # Daily OHLCV data for strategies
        rets_daily: pd.DataFrame | None = None,
        # features: dict | None = None, # Removed features argument
        verbose: bool = True,
    ):
        if verbose:
            logger.info(f"Running scenario: {scenario_config['name']}")

        # Ensure rets_daily is calculated based on daily close prices if not provided
        # This requires extracting daily closes from price_data_daily_ohlc
        if rets_daily is None:
            if isinstance(price_data_daily_ohlc.columns, pd.MultiIndex) and \
               'Close' in price_data_daily_ohlc.columns.get_level_values(1):
                daily_closes_for_rets = price_data_daily_ohlc.xs('Close', level='Field', axis=1)
            elif not isinstance(price_data_daily_ohlc.columns, pd.MultiIndex):
                daily_closes_for_rets = price_data_daily_ohlc # Assume it's already close prices
            else:
                # Attempt to find 'Close' prices in a less structured MultiIndex
                try:
                    if 'Close' in price_data_daily_ohlc.columns.get_level_values(-1):
                        daily_closes_for_rets = price_data_daily_ohlc.xs('Close', level=-1, axis=1)
                    else:
                        raise ValueError("run_scenario: Could not reliably extract 'Close' prices for daily returns.")
                except Exception as e:
                    raise ValueError(f"run_scenario: Error extracting 'Close' prices for daily returns: {e}. Columns: {price_data_daily_ohlc.columns}")

            if daily_closes_for_rets.empty:
                 raise ValueError("run_scenario: Daily close prices for return calculation are empty.")
            rets_daily = daily_closes_for_rets.pct_change(fill_method=None).fillna(0)

        strategy = self._get_strategy(
            scenario_config["strategy"], scenario_config["strategy_params"]
        )
        
        universe_tickers = [item[0] for item in strategy.get_universe(self.global_config)]
        benchmark_ticker = self.global_config["benchmark"]

        # --- Iterative Signal Generation ---
        all_monthly_weights = []
        rebalance_dates = price_data_monthly_closes.index # Dates for generating signals (typically month-end)

        # WFO dates from scenario_config (optional)
        wfo_start_date = pd.to_datetime(scenario_config.get("wfo_start_date", None))
        wfo_end_date = pd.to_datetime(scenario_config.get("wfo_end_date", None))

        for current_rebalance_date in rebalance_dates:
            if verbose:
                logger.debug(f"Generating signals for date: {current_rebalance_date}")

            # Prepare historical data up to the current_rebalance_date
            # For assets in universe
            if isinstance(price_data_daily_ohlc.columns, pd.MultiIndex) and 'Ticker' in price_data_daily_ohlc.columns.names:
                asset_hist_data_cols = pd.MultiIndex.from_product([universe_tickers, price_data_daily_ohlc.columns.get_level_values('Field').unique()], names=['Ticker', 'Field'])
                asset_hist_data_cols = [col for col in asset_hist_data_cols if col in price_data_daily_ohlc.columns] # Ensure columns exist
                all_historical_data_for_strat = price_data_daily_ohlc.loc[price_data_daily_ohlc.index <= current_rebalance_date, asset_hist_data_cols]

                # For benchmark
                benchmark_hist_data_cols = pd.MultiIndex.from_product([[benchmark_ticker], price_data_daily_ohlc.columns.get_level_values('Field').unique()], names=['Ticker', 'Field'])
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
            logger.warning(f"No signals generated for scenario {scenario_config['name']}. This might be due to WFO window or other issues.")
            # Create an empty DataFrame with expected structure for downstream processing
            signals = pd.DataFrame(columns=universe_tickers, index=rebalance_dates)
        else:
            signals = pd.concat(all_monthly_weights)
            signals = signals.reindex(rebalance_dates).fillna(0.0) # Ensure all rebalance dates are present

        if verbose:
            logger.debug(f"All signals generated for scenario: {scenario_config['name']}")
            logger.info(f"Signals head:\n{signals.head()}")
            logger.info(f"Signals tail:\n{signals.tail()}")
            if signals.empty:
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

        logger.debug(f"Filtered_sizer_params: {filtered_sizer_params}")
        logger.debug(f"window_param extracted: {window_param}")
        logger.debug(f"target_return_param extracted: {target_return_param}")
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

        weights_monthly = rebalance(
            sized_signals, scenario_config["rebalance_frequency"]
        )

        weights_monthly = weights_monthly.reindex(columns=universe_tickers).fillna(0.0)

        # Use the index from daily OHLC data for reindexing monthly weights to daily
        weights_daily = weights_monthly.reindex(price_data_daily_ohlc.index, method="ffill")
        weights_daily = weights_daily.shift(1).fillna(0.0) # Apply execution lag

        # Ensure daily returns are aligned with the daily OHLC data index
        aligned_rets_daily = rets_daily.reindex(price_data_daily_ohlc.index).fillna(0.0)

        # Ensure universe_tickers used for indexing aligned_rets_daily are present in its columns
        valid_universe_tickers_in_rets = [ticker for ticker in universe_tickers if ticker in aligned_rets_daily.columns]
        if len(valid_universe_tickers_in_rets) < len(universe_tickers):
            missing_tickers = set(universe_tickers) - set(valid_universe_tickers_in_rets)
            logger.warning(f"Tickers {missing_tickers} not found in aligned_rets_daily columns. Portfolio calculations might be affected.")

        # Calculate gross returns using only valid tickers present in returns data
        if not valid_universe_tickers_in_rets:
            logger.warning("No valid universe tickers found in daily returns. Gross portfolio returns will be zero.")
            daily_portfolio_returns_gross = pd.Series(0.0, index=weights_daily.index)
        else:
            daily_portfolio_returns_gross = (weights_daily[valid_universe_tickers_in_rets] * aligned_rets_daily[valid_universe_tickers_in_rets]).sum(axis=1)

        turnover = (weights_daily - weights_daily.shift(1)).abs().sum(axis=1).fillna(0.0)
        transaction_costs = turnover * (scenario_config.get("transaction_costs_bps", 0) / 10000.0)

        portfolio_rets_net = (daily_portfolio_returns_gross - transaction_costs).fillna(0.0)

        if verbose:
            logger.info(f"Portfolio net returns calculated for {scenario_config['name']}. First few net returns: {portfolio_rets_net.head().to_dict()}")
            logger.info(f"Net returns index: {portfolio_rets_net.index.min()} to {portfolio_rets_net.index.max()}")

        return portfolio_rets_net

    def run(self):
        logger.info("Starting backtest data retrieval.")
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
        self.daily_data_ohlc = daily_data_std_format.copy()
        logger.info(f"Shape of self.daily_data_ohlc: {self.daily_data_ohlc.shape}")
        logger.debug(f"Columns of self.daily_data_ohlc: {self.daily_data_ohlc.columns}")


        # The section for preparing 'monthly_data_for_features' is no longer needed
        # as features are computed within strategies using daily_data_ohlc.
        logger.info("Feature pre-computation step removed. Features will be calculated within strategies.")

        # Extract daily closes for return calculations and for monthly resampling if needed by other parts
        if isinstance(self.daily_data_ohlc.columns, pd.MultiIndex) and \
           'Close' in self.daily_data_ohlc.columns.get_level_values(1):
            # Assuming 'Ticker' is level 0 and 'Field' is level 1
            daily_closes = self.daily_data_ohlc.xs('Close', level='Field', axis=1)
        elif not isinstance(self.daily_data_ohlc.columns, pd.MultiIndex):
            # If daily_data_ohlc is just close prices (older format or specific data source output)
            daily_closes = self.daily_data_ohlc
        else:
            # Attempt to find 'Close' prices in a less structured MultiIndex or raise error
            # This part might need adjustment based on actual data structures if not (Ticker, Field)
            try:
                if 'Close' in self.daily_data_ohlc.columns.get_level_values(-1): # Check last level for 'Close'
                    daily_closes = self.daily_data_ohlc.xs('Close', level=-1, axis=1)
                else:
                    raise ValueError("Could not reliably extract 'Close' prices from self.daily_data_ohlc due to unrecognized column structure.")
            except Exception as e:
                 raise ValueError(f"Error extracting 'Close' prices from self.daily_data_ohlc: {e}. Columns: {self.daily_data_ohlc.columns}")

        if daily_closes.empty:
            raise ValueError("Daily close prices could not be extracted or are empty.")

        # monthly_closes will be used for determining rebalance dates and for sizers if they need monthly frequency.
        monthly_closes = daily_closes.resample("BME").last()
        self.monthly_data = monthly_closes # Store monthly closing prices

        logger.info("Backtest data retrieved and prepared (daily OHLC, monthly closes).")

        # Removed strategy_registry and get_required_features_from_scenarios
        # Removed benchmark_monthly_closes (strategies will use benchmark_historical_data from daily_data_ohlc)
        # Removed precompute_features call and self.features initialization

        rets_full = daily_closes.pct_change(fill_method=None).fillna(0)
        self.rets_full = rets_full # These are daily returns based on daily_closes

        # The arguments to run_optimize_mode, run_backtest_mode, run_monte_carlo_mode
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
        elif self.args.mode == "monte_carlo":
            # Monte Carlo might need different data inputs or just portfolio returns.
            # Assuming it primarily works off portfolio returns generated by run_backtest_mode.
            # For now, keeping its inputs similar, but this might need review based on its internal logic.
            self.run_monte_carlo_mode(self.scenarios[0], self.monthly_data, self.daily_data_ohlc, rets_full)

        if self.args.mode != "monte_carlo":
            if CENTRAL_INTERRUPTED_FLAG:
                logger.warning("Operation interrupted by user. Skipping final results display and plotting.")
            else:
                logger.info("All scenarios completed. Displaying results.")
                self.display_results(daily_data)

from .utils import register_signal_handler as register_central_signal_handler

if __name__ == "__main__":
    register_central_signal_handler()

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
