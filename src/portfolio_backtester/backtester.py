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
from .features.feature_helpers import get_required_features_from_scenarios
from .feature_engineering import precompute_features
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
        self.features: Dict[str, pd.DataFrame | pd.Series] | None = None
        self.monthly_data: pd.DataFrame | None = None
        self.daily_data: pd.DataFrame | None = None
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
        
        universe_tickers = [item[0] for item in strategy.get_universe(self.global_config)]
        
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

        sizer_args = [signals, strategy_data_monthly, benchmark_data_monthly]
        
        if sizer_name == "rolling_downside_volatility":
            daily_prices_for_vol = price_data_daily[universe_tickers]
            sizer_args.append(daily_prices_for_vol)

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

        weights_daily = weights_monthly.reindex(price_data_daily.index, method="ffill")
        weights_daily = weights_daily.shift(1).fillna(0.0)

        aligned_rets_daily = rets_daily.reindex(price_data_daily.index).fillna(0.0)

        daily_portfolio_returns_gross = (weights_daily * aligned_rets_daily[universe_tickers]).sum(axis=1)

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

        self.daily_data_ohlc = daily_data_std_format.copy()

        monthly_data_for_features = pd.DataFrame(index=self.daily_data_ohlc.index.unique())

        if isinstance(self.daily_data_ohlc.columns, pd.MultiIndex) and \
           self.daily_data_ohlc.columns.nlevels == 2 and \
           self.daily_data_ohlc.columns.names[0] == 'Ticker' and \
           self.daily_data_ohlc.columns.names[1] == 'Field':

            logger.info("Daily OHLC data has (Ticker, Field) MultiIndex. Preparing monthly OHLC for features.")
            monthly_ohlc_for_features_list = []
            for ticker in self.daily_data_ohlc.columns.get_level_values('Ticker').unique():
                try:
                    ticker_ohlc_df = self.daily_data_ohlc[ticker]
                    required_cols = ['Open', 'High', 'Low', 'Close']
                    if all(col in ticker_ohlc_df.columns for col in required_cols):
                        resampled_ohlc = ticker_ohlc_df.resample("BME").agg(
                            {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
                        )
                        if 'Volume' in ticker_ohlc_df.columns:
                            resampled_ohlc['Volume'] = ticker_ohlc_df['Volume'].resample("BME").sum()

                        resampled_ohlc.columns = pd.MultiIndex.from_product([[ticker], list(resampled_ohlc.columns)], names=['Ticker', 'Field'])
                        monthly_ohlc_for_features_list.append(resampled_ohlc)
                    else:
                        logger.warning(f"Ticker {ticker} missing one or more required OHLC columns (Open, High, Low, Close) when processing (Ticker, Field) data. Skipping for monthly OHLC features.")
                except Exception as e:
                    logger.error(f"Error resampling OHLC for ticker {ticker} from (Ticker,Field) data: {e}")

            if monthly_ohlc_for_features_list:
                monthly_data_for_features = pd.concat(monthly_ohlc_for_features_list, axis=1)

        elif not isinstance(self.daily_data_ohlc.columns, pd.MultiIndex) and isinstance(self.daily_data_ohlc, pd.DataFrame):
            logger.warning("Daily data is in old format (close prices only). ATRFeature requires OHLC and will not compute meaningful values.")
            empty_cols = pd.MultiIndex.from_tuples([], names=['Ticker', 'Field'])
            idx_for_empty_df = self.daily_data_ohlc.index.unique() if not self.daily_data_ohlc.empty else pd.Index([])
            monthly_data_for_features = pd.DataFrame(index=idx_for_empty_df, columns=empty_cols)
        else:
            logger.error(f"Daily OHLC data is in an unrecognized format. Columns: {self.daily_data_ohlc.columns}. ATRFeature may fail.")
            empty_cols = pd.MultiIndex.from_tuples([], names=['Ticker', 'Field'])
            idx_for_empty_df = self.daily_data_ohlc.index.unique() if hasattr(self.daily_data_ohlc, 'index') and not self.daily_data_ohlc.empty else pd.Index([])
            monthly_data_for_features = pd.DataFrame(index=idx_for_empty_df, columns=empty_cols)

        if isinstance(self.daily_data_ohlc.columns, pd.MultiIndex) and 'Close' in self.daily_data_ohlc.columns.get_level_values(1):
            daily_closes = self.daily_data_ohlc.loc[:, self.daily_data_ohlc.columns.get_level_values(1) == 'Close']
        else:
            if not isinstance(daily_data.columns, pd.MultiIndex):
                daily_closes = daily_data
            else:
                raise ValueError("Daily data format for extracting close prices is not recognized.")

        monthly_closes = daily_closes.resample("BME").last()

        logger.info("Backtest data retrieved and prepared (daily OHLC, monthly OHLC for features, monthly closes).")

        strategy_registry = {
            "calmar_momentum": strategies.CalmarMomentumStrategy,
            "vams_no_downside": strategies.VAMSNoDownsideStrategy,
            "momentum": strategies.MomentumStrategy,
            "sharpe_momentum": strategies.SharpeMomentumStrategy,
            "sortino_momentum": strategies.SortinoMomentumStrategy,
            "vams_momentum": strategies.VAMSMomentumStrategy,
            "momentum_dvol_sizer": strategies.MomentumDvolSizerStrategy,
            "filtered_lagged_momentum": strategies.FilteredLaggedMomentumStrategy,
        }
        
        required_features = get_required_features_from_scenarios(self.scenarios, strategy_registry)
        
        benchmark_monthly_closes = monthly_closes[self.global_config["benchmark"]]

        if isinstance(monthly_closes, pd.Series):
            monthly_closes_df = monthly_closes.to_frame()
        else:
            monthly_closes_df = monthly_closes

        self.features = precompute_features(
            data=monthly_data_for_features,
            required_features=required_features,
            benchmark_data=benchmark_monthly_closes,
            legacy_monthly_closes=monthly_closes_df
        )
        logger.info("All features pre-computed.")

        rets_full = daily_closes.pct_change(fill_method=None).fillna(0)
        self.rets_full = rets_full

        if self.args.mode == "optimize":
            self.run_optimize_mode(self.scenarios[0], monthly_closes, daily_closes, rets_full)
        elif self.args.mode == "backtest":
            self.run_backtest_mode(self.scenarios[0], monthly_closes, daily_closes, rets_full)
        elif self.args.mode == "monte_carlo":
            self.run_monte_carlo_mode(self.scenarios[0], monthly_closes, daily_closes, rets_full)

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
