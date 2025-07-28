import logging
import time
import types
import sys
import argparse
from typing import Any

import numpy as np
import optuna
import pandas as pd

from . import strategies
from .strategies import enumerate_strategies_with_params
from .backtester_logic.execution import (
    _generate_optimization_report,
    run_backtest_mode,
    run_optimize_mode,
)
from .backtester_logic.optimization import run_optimization
from .backtester_logic.reporting import display_results
from .config_initializer import populate_default_optimizations
from .config_loader import OPTIMIZER_PARAMETER_DEFAULTS
from .data_cache import get_global_cache
from .portfolio.position_sizer import get_position_sizer, SIZER_PARAM_MAPPING
from .portfolio.rebalancing import rebalance
from .utils import INTERRUPTED as CENTRAL_INTERRUPTED_FLAG

logger = logging.getLogger(__name__)


class TimeoutManager:
    def __init__(self, timeout_seconds):
        self.timeout = timeout_seconds
        self.start_time = time.time()

    def check_timeout(self):
        if self.timeout is None:
            return False
        
        # CRITICAL: Handle Mock objects that might not support numeric comparison
        # Problem: In tests, Mock objects might be passed as timeout values, but Mock objects
        # don't support numeric operations like comparison or float() conversion
        # Solution: Use defensive programming with try-catch and type conversion
        try:
            # Ensure timeout is a number before comparison
            timeout_value = float(self.timeout) if self.timeout is not None else None
            if timeout_value is None:
                return False
                
            elapsed_time = time.time() - self.start_time
            if elapsed_time > timeout_value:
                logger.warning(f"Timeout of {timeout_value} seconds exceeded. Elapsed time: {elapsed_time:.2f} seconds.")
                print(f"Warning: Timeout of {timeout_value} seconds exceeded.")
                return True
        except (TypeError, ValueError, AttributeError):
            # TRICKY: If timeout is a Mock object or invalid type, assume no timeout
            # This prevents crashes during testing when Mock objects are used
            # Mock objects will raise TypeError on float() conversion
            return False
        
        return False


class Backtester:
    def __init__(self, global_config, scenarios, args, random_state=None):
        self.global_config = global_config
        self.global_config["optimizer_parameter_defaults"] = OPTIMIZER_PARAMETER_DEFAULTS
        self.scenarios = scenarios
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Backtester initialized with scenario strategy_params: {self.scenarios[0].get('strategy_params')}")
        populate_default_optimizations(self.scenarios, OPTIMIZER_PARAMETER_DEFAULTS)
        self.args = args
        self.timeout_manager = TimeoutManager(args.timeout)
        self.strategy_map = {name: klass for name, klass in enumerate_strategies_with_params().items()}
        self.data_source = self._get_data_source()
        self.results = {}
        self.monthly_data: pd.DataFrame | None = None
        self.daily_data_ohlc: pd.DataFrame | None = None
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
        
        self.data_cache = get_global_cache()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Data preprocessing cache initialized")
        
        self._daily_index_cache = None
        
        self.asset_replacement_manager = None
        self.synthetic_data_generator = None
        if self.global_config.get('enable_synthetic_data', False):
            self._initialize_monte_carlo_components()
            if self.asset_replacement_manager is not None:
                self.asset_replacement_manager.set_full_data_source(self.data_source, self.global_config)

        self.run_optimization = types.MethodType(run_optimization, self)
        self.run_backtest_mode = types.MethodType(run_backtest_mode, self)
        self.run_optimize_mode = types.MethodType(run_optimize_mode, self)
        self.display_results = types.MethodType(display_results, self)
        self._generate_optimization_report = types.MethodType(_generate_optimization_report, self)
        
        from .backtester_logic.execution import generate_deferred_report
        self.generate_deferred_report = types.MethodType(generate_deferred_report, self)

    @property
    def has_timed_out(self):
        return self.timeout_manager.check_timeout()

    def _get_data_source(self):
        from .data_sources.stooq_data_source import StooqDataSource
        from .data_sources.yfinance_data_source import YFinanceDataSource
        from .data_sources.hybrid_data_source import HybridDataSource

        data_source_map = {
            "stooq": StooqDataSource,
            "yfinance": YFinanceDataSource,
            "hybrid": HybridDataSource,
        }

        ds_name = self.global_config.get("data_source", "hybrid").lower()
        data_source_class = data_source_map.get(ds_name)

        if data_source_class:
            logger.debug(f"Using {data_source_class.__name__}.")
            if ds_name == "hybrid":
                prefer_stooq = self.global_config.get("prefer_stooq", True)
                return HybridDataSource(prefer_stooq=prefer_stooq)
            return data_source_class()
        else:
            logger.error(f"Unsupported data source: {ds_name}")
            raise ValueError(f"Unsupported data source: {ds_name}")

    def _get_strategy(self, strategy_name, params):
        strategy_class = self.strategy_map.get(strategy_name)
        
        if strategy_class:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Using {strategy_class.__name__} with params: {params}")
            return strategy_class(params)
        else:
            logger.error(f"Unsupported strategy: {strategy_name}")
            raise ValueError(f"Unsupported strategy: {strategy_name}")
    
    def _initialize_monte_carlo_components(self):
        """Initialize Monte Carlo components for synthetic data generation."""
        try:
            from .monte_carlo.asset_replacement import AssetReplacementManager
            
            self.asset_replacement_manager = AssetReplacementManager(self.global_config)
            
            self.synthetic_data_generator = self.asset_replacement_manager.synthetic_generator
            
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("Monte Carlo components initialized successfully")
            
        except ImportError as e:
            if self.logger.isEnabledFor(logging.WARNING):
                self.logger.warning(f"Monte Carlo components not available: {e}")
            self.asset_replacement_manager = None
            self.synthetic_data_generator = None

    def _prepare_scenario_data(self, price_data_daily_ohlc, universe_tickers, benchmark_ticker):
        daily_closes = None
        if price_data_daily_ohlc is not None:
            if isinstance(price_data_daily_ohlc.columns, pd.MultiIndex) and \
               'Close' in price_data_daily_ohlc.columns.get_level_values(1):
                daily_closes = price_data_daily_ohlc.xs('Close', level='Field', axis=1)
            elif not isinstance(price_data_daily_ohlc.columns, pd.MultiIndex):
                daily_closes = price_data_daily_ohlc
            else:
                try:
                    if 'Close' in price_data_daily_ohlc.columns.get_level_values(-1):
                        daily_closes = price_data_daily_ohlc.xs('Close', level=-1, axis=1)
                    else:
                        raise ValueError("Could not reliably extract 'Close' prices from price_data_daily_ohlc due to unrecognized column structure.")
                except Exception as e:
                     raise ValueError(f"Error extracting 'Close' prices from price_data_daily_ohlc: {e}. Columns: {price_data_daily_ohlc.columns}")

        if daily_closes is None or daily_closes.empty:
            raise ValueError("Daily close prices could not be extracted or are empty.")

        if isinstance(daily_closes, pd.Series):
            daily_closes = daily_closes.to_frame()

        monthly_closes = daily_closes.resample("BME").last()
        price_data_monthly_closes = monthly_closes.to_frame() if isinstance(monthly_closes, pd.Series) else monthly_closes

        rets_daily = self.data_cache.get_cached_returns(daily_closes, "full_period_returns")
        rets_daily = rets_daily.to_frame() if isinstance(rets_daily, pd.Series) else rets_daily

        return price_data_monthly_closes, rets_daily


    def _generate_signals(self, strategy, scenario_config, price_data_daily_ohlc, universe_tickers, benchmark_ticker):
        timing_controller = strategy.get_timing_controller()
        timing_controller.reset_state()

        start_date = price_data_daily_ohlc.index.min()
        end_date = price_data_daily_ohlc.index.max()

        wfo_start_date = pd.to_datetime(scenario_config.get("wfo_start_date", None))
        wfo_end_date = pd.to_datetime(scenario_config.get("wfo_end_date", None))

        if wfo_start_date is not None:
            start_date = max(start_date, wfo_start_date)
        if wfo_end_date is not None:
            end_date = min(end_date, wfo_end_date)

        rebalance_dates = timing_controller.get_rebalance_dates(
            start_date=start_date,
            end_date=end_date,
            available_dates=price_data_daily_ohlc.index,
            strategy_context=strategy
        )

        all_monthly_weights = []

        for current_rebalance_date in rebalance_dates:
            if self.has_timed_out:
                logger.warning("Timeout reached during scenario run. Halting signal generation.")
                break

            should_generate = timing_controller.should_generate_signal(
                current_date=current_rebalance_date,
                strategy_context=strategy
            )

            if not should_generate:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Timing controller skipped signal generation for date: {current_rebalance_date}")
                continue

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Generating signals for date: {current_rebalance_date}")

            if isinstance(price_data_daily_ohlc.columns, pd.MultiIndex) and 'Ticker' in price_data_daily_ohlc.columns.names:
                asset_hist_data_cols = pd.MultiIndex.from_product([universe_tickers, list(price_data_daily_ohlc.columns.get_level_values('Field').unique())], names=['Ticker', 'Field'])
                asset_hist_data_cols = [col for col in asset_hist_data_cols if col in price_data_daily_ohlc.columns]
                all_historical_data_for_strat = price_data_daily_ohlc.loc[price_data_daily_ohlc.index <= current_rebalance_date, asset_hist_data_cols]

                benchmark_hist_data_cols = pd.MultiIndex.from_product([[benchmark_ticker], list(price_data_daily_ohlc.columns.get_level_values('Field').unique())], names=['Ticker', 'Field'])
                benchmark_hist_data_cols = [col for col in benchmark_hist_data_cols if col in price_data_daily_ohlc.columns]
                benchmark_historical_data_for_strat = price_data_daily_ohlc.loc[price_data_daily_ohlc.index <= current_rebalance_date, benchmark_hist_data_cols]
            else:
                all_historical_data_for_strat = price_data_daily_ohlc.loc[price_data_daily_ohlc.index <= current_rebalance_date, universe_tickers]
                benchmark_historical_data_for_strat = price_data_daily_ohlc.loc[price_data_daily_ohlc.index <= current_rebalance_date, [benchmark_ticker]]

            non_universe_tickers = strategy.get_non_universe_data_requirements()
            non_universe_historical_data_for_strat = pd.DataFrame()
            if non_universe_tickers:
                if isinstance(price_data_daily_ohlc.columns, pd.MultiIndex) and 'Ticker' in price_data_daily_ohlc.columns.names:
                    non_universe_hist_data_cols = pd.MultiIndex.from_product([non_universe_tickers, list(price_data_daily_ohlc.columns.get_level_values('Field').unique())], names=['Ticker', 'Field'])
                    non_universe_hist_data_cols = [col for col in non_universe_hist_data_cols if col in price_data_daily_ohlc.columns]
                    non_universe_historical_data_for_strat = price_data_daily_ohlc.loc[price_data_daily_ohlc.index <= current_rebalance_date, non_universe_hist_data_cols]
                else:
                    non_universe_historical_data_for_strat = price_data_daily_ohlc.loc[price_data_daily_ohlc.index <= current_rebalance_date, non_universe_tickers]

            import inspect
            sig = inspect.signature(strategy.generate_signals)
            if 'non_universe_historical_data' in sig.parameters:
                current_weights_df = strategy.generate_signals(
                    all_historical_data=all_historical_data_for_strat,
                    benchmark_historical_data=benchmark_historical_data_for_strat,
                    non_universe_historical_data=non_universe_historical_data_for_strat,
                    current_date=current_rebalance_date,
                    start_date=wfo_start_date,
                    end_date=wfo_end_date
                )
            else:
                current_weights_df = strategy.generate_signals(
                    all_historical_data=all_historical_data_for_strat,
                    benchmark_historical_data=benchmark_historical_data_for_strat,
                    current_date=current_rebalance_date,
                    start_date=wfo_start_date,
                    end_date=wfo_end_date
                )

            if current_weights_df is not None and not current_weights_df.empty:
                if len(current_weights_df) > 0:
                    current_weights_series = current_weights_df.iloc[0]
                    timing_controller.update_signal_state(current_rebalance_date, current_weights_series)

                    try:
                        if isinstance(price_data_daily_ohlc.columns, pd.MultiIndex) and 'Close' in price_data_daily_ohlc.columns.get_level_values(1):
                            current_prices = price_data_daily_ohlc.loc[current_rebalance_date].xs('Close', level='Field')
                        elif not isinstance(price_data_daily_ohlc.columns, pd.MultiIndex):
                            current_prices = price_data_daily_ohlc.loc[current_rebalance_date]
                        else:
                            try:
                                current_prices = price_data_daily_ohlc.loc[current_rebalance_date].xs('Close', level=-1)
                            except:
                                current_prices = price_data_daily_ohlc.loc[current_rebalance_date].iloc[:len(universe_tickers)]

                        universe_prices = current_prices.reindex(universe_tickers).ffill()

                        timing_controller.update_position_state(
                            current_rebalance_date, 
                            current_weights_series, 
                            universe_prices
                        )

                    except Exception as e:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Could not update position state for {current_rebalance_date}: {e}")

            all_monthly_weights.append(current_weights_df)

        if not all_monthly_weights:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(f"No signals generated for scenario {scenario_config['name']}. This might be due to WFO window or other issues.")
            signals = pd.DataFrame(columns=universe_tickers, index=rebalance_dates)
        else:
            signals = pd.concat(all_monthly_weights)
            signals = signals.reindex(rebalance_dates).fillna(0.0)

        return signals

    def _size_positions(self, signals, scenario_config, price_data_monthly_closes, price_data_daily_ohlc, universe_tickers, benchmark_ticker):
        sizer_name = scenario_config.get("position_sizer", "equal_weight")
        sizer_func = get_position_sizer(sizer_name)

        sizer_param_mapping = SIZER_PARAM_MAPPING

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

        strategy_monthly_closes = price_data_monthly_closes[universe_tickers]
        benchmark_monthly_closes = price_data_monthly_closes[benchmark_ticker]

        sizer_args = [signals, strategy_monthly_closes, benchmark_monthly_closes]

        if sizer_name == "rolling_downside_volatility":
            if isinstance(price_data_daily_ohlc.columns, pd.MultiIndex) and \
               'Close' in price_data_daily_ohlc.columns.get_level_values(1):
                daily_closes_for_sizer = price_data_daily_ohlc.xs('Close', level='Field', axis=1)[universe_tickers]
            elif not isinstance(price_data_daily_ohlc.columns, pd.MultiIndex):
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

        sized_signals = sizer_func(
            *sizer_args,
            **filtered_sizer_params,
        )

        return sized_signals

    def _calculate_portfolio_returns(self, sized_signals, scenario_config, price_data_daily_ohlc, rets_daily, universe_tickers):
        weights_monthly = rebalance(
            sized_signals, scenario_config["rebalance_frequency"]
        )

        weights_monthly = weights_monthly.reindex(columns=universe_tickers).fillna(0.0)

        weights_daily = weights_monthly.reindex(price_data_daily_ohlc.index, method="ffill")
        weights_daily = weights_daily.shift(1).fillna(0.0)

        if rets_daily is None:
            logger.error("rets_daily is None before reindexing in run_scenario.")
            return pd.Series(0.0, index=price_data_daily_ohlc.index)

        aligned_rets_daily = rets_daily.reindex(price_data_daily_ohlc.index).fillna(0.0)

        valid_universe_tickers_in_rets = [ticker for ticker in universe_tickers if ticker in aligned_rets_daily.columns]
        if len(valid_universe_tickers_in_rets) < len(universe_tickers):
            missing_tickers = set(universe_tickers) - set(valid_universe_tickers_in_rets)
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(f"Tickers {missing_tickers} not found in aligned_rets_daily columns. Portfolio calculations might be affected.")

        if not valid_universe_tickers_in_rets:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning("No valid universe tickers found in daily returns. Gross portfolio returns will be zero.")
            daily_portfolio_returns_gross = pd.Series(0.0, index=weights_daily.index)
        else:
            daily_portfolio_returns_gross = (weights_daily[valid_universe_tickers_in_rets] * aligned_rets_daily[valid_universe_tickers_in_rets]).sum(axis=1)

        turnover = (weights_daily - weights_daily.shift(1)).abs().sum(axis=1).fillna(0.0)

        from .trading.transaction_costs import get_transaction_cost_model
        
        tx_cost_model = get_transaction_cost_model(self.global_config)
        transaction_costs, _ = tx_cost_model.calculate(
            turnover=turnover,
            weights_daily=weights_daily,
            price_data=price_data_daily_ohlc,
            portfolio_value=self.global_config.get("portfolio_value", 100000.0)
        )

        portfolio_rets_net = (daily_portfolio_returns_gross - transaction_costs).fillna(0.0)

        return portfolio_rets_net

    def run_scenario(
        self,
        scenario_config,
        price_data_monthly_closes: pd.DataFrame,
        price_data_daily_ohlc: pd.DataFrame,
        rets_daily: pd.DataFrame | None = None,
        verbose: bool = True,
    ):
        if verbose:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Running scenario: {scenario_config['name']}")

        strategy = self._get_strategy(
            scenario_config["strategy"], scenario_config["strategy_params"]
        )
        
        if "universe" in scenario_config:
            universe_tickers = scenario_config["universe"]
        else:
            universe_tickers = [item[0] for item in strategy.get_universe(self.global_config)]

        missing_cols = [t for t in universe_tickers if t not in price_data_monthly_closes.columns]
        if missing_cols:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    f"Tickers {missing_cols} not found in price data; they will be skipped for this run."
                )
            universe_tickers = [t for t in universe_tickers if t not in missing_cols]

        if not universe_tickers:
            logger.warning("No universe tickers remain after filtering for missing data. Skipping scenario.")
            return pd.Series(dtype=float)

        benchmark_ticker = self.global_config["benchmark"]

        price_data_monthly_closes, rets_daily = self._prepare_scenario_data(price_data_daily_ohlc, universe_tickers, benchmark_ticker)

        signals = self._generate_signals(strategy, scenario_config, price_data_daily_ohlc, universe_tickers, benchmark_ticker)

        sized_signals = self._size_positions(signals, scenario_config, price_data_monthly_closes, price_data_daily_ohlc, universe_tickers, benchmark_ticker)

        portfolio_rets_net = self._calculate_portfolio_returns(sized_signals, scenario_config, price_data_daily_ohlc, rets_daily, universe_tickers)

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
        import os
        from . import numba_kernels as _nk

        use_fast = os.environ.get("ENABLE_NUMBA_WALKFORWARD", "0") == "1"

        if use_fast:
            try:
                daily_returns = rets_full_np.astype(np.float32)

                if daily_returns.size == 0:
                    raise ValueError("Daily returns array is empty – cannot run fast path")

                port_rets = np.nanmean(daily_returns, axis=1).astype(np.float32)

                if self._daily_index_cache is None:
                    raise ValueError("_daily_index_cache is not initialized")
                test_starts = np.asarray([np.searchsorted(self._daily_index_cache, te_start) for _, _, te_start, _ in windows], dtype=np.int64)
                test_ends = np.asarray([np.searchsorted(self._daily_index_cache, te_end) for _, _, _, te_end in windows], dtype=np.int64)

                metrics_mat = _nk.window_mean_std(port_rets, test_starts, test_ends)

                avg_metric = float(np.nanmean(metrics_mat[:, 0]))
                return avg_metric if not is_multi_objective else (avg_metric,)

            except Exception as exc:
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
        from .reporting.metrics import calculate_metrics
        
        if not hasattr(self, '_windows_precomputed') or not self._windows_precomputed:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Pre-computing returns for all windows")
            self.data_cache.precompute_window_returns(daily_data, windows)
            self._windows_precomputed = True
        
        from .backtester_logic.optimization import _global_progress_tracker
        from .utils import calculate_stability_metrics, INTERRUPTED as CENTRAL_INTERRUPTED_FLAG
        
        metric_values_per_objective = [[] for _ in metrics_to_optimize]
        processed_steps_for_pruning = 0

        pruning_enabled = getattr(self.args, "pruning_enabled", False)
        pruning_interval_steps = getattr(self.args, "pruning_interval_steps", 1)

        monte_carlo_config = self.global_config.get('monte_carlo_config', {})
        asset_replacement_manager = None
        trial_synthetic_data = None
        
        mc_enabled = monte_carlo_config.get('enable_synthetic_data', False)
        mc_during_optimization = monte_carlo_config.get('enable_during_optimization', True)
        
        optimization_mode = monte_carlo_config.get('optimization_mode', 'balanced')
        trial_threshold = self._get_monte_carlo_trial_threshold(optimization_mode)
        
        trial_number = getattr(trial, 'number', 0) if trial else 0
        strategy = self._get_strategy(scenario_config["strategy"], scenario_config["strategy_params"])
        mc_adaptive_enabled = mc_enabled and mc_during_optimization and (trial_number >= trial_threshold) and strategy.get_synthetic_data_requirements()
        
        if mc_adaptive_enabled and 'asset_replacement_manager' not in scenario_config:
            from .monte_carlo.asset_replacement import AssetReplacementManager
            
            stage1_config = monte_carlo_config.copy()
            stage1_config['stage1_optimization'] = True
            stage1_config['replacement_percentage'] = monte_carlo_config.get('replacement_percentage', 0.05)
            
            stage1_config['generation_config'] = {
                'buffer_multiplier': 1.0, 'max_attempts': 1, 'validation_tolerance': 1.0
            }
            stage1_config['validation_config'] = {'enable_validation': False}
            
            asset_replacement_manager = AssetReplacementManager(stage1_config)
            asset_replacement_manager.set_full_data_source(self.data_source, self.global_config)
            scenario_config['asset_replacement_manager'] = asset_replacement_manager
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Stage 1 MC: Initialized AssetReplacementManager for optimization robustness (mode: {optimization_mode})")

        asset_replacement_manager = scenario_config.get('asset_replacement_manager')

        all_window_returns = []
        
        if _global_progress_tracker and trial and hasattr(trial, 'number'):
            trial_num = trial.number + 1
            total_trials = _global_progress_tracker['total_trials']
            _global_progress_tracker['progress'].update(
                _global_progress_tracker['task'], 
                description=f"[cyan]Trial {trial_num}/{total_trials} running ({len(windows)} windows/trial)..."
            )
        
        replacement_info = None
        if asset_replacement_manager is not None:
            try:
                universe = scenario_config.get('universe', self.global_config.get('universe', []))
                
                all_start_dates = [tr_start for tr_start, _, _, _ in windows]
                all_end_dates = [te_end for _, _, _, te_end in windows]
                overall_start = min(all_start_dates)
                overall_end = max(all_end_dates)
                
                full_data_slice = daily_data.loc[overall_start:overall_end]
                
                daily_data_dict = {}
                
                if isinstance(full_data_slice.columns, pd.MultiIndex):
                    for ticker in universe:
                        ticker_data = full_data_slice.xs(ticker, level='Ticker', axis=1, drop_level=True)
                        if not ticker_data.empty:
                            daily_data_dict[ticker] = ticker_data
                else:
                    for ticker in universe:
                        if ticker in full_data_slice.columns:
                            ticker_data = pd.DataFrame({
                                'Open': full_data_slice[ticker],
                                'High': full_data_slice[ticker],
                                'Low': full_data_slice[ticker],
                                'Close': full_data_slice[ticker]
                            })
                            daily_data_dict[ticker] = ticker_data
                
                trial_seed = None
                if monte_carlo_config.get('random_seed') is not None:
                    trial_seed = monte_carlo_config['random_seed'] + getattr(trial, 'number', 0)
                
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
            if self.has_timed_out:
                logger.warning("Timeout reached during walk-forward evaluation. Stopping further windows.")
                break
            if CENTRAL_INTERRUPTED_FLAG:
                self.logger.warning("Evaluation interrupted by user via central flag.")
                break

            m_slice = monthly_data.loc[tr_start:tr_end]
            d_slice = daily_data.loc[tr_start:te_end]

            current_daily_data_ohlc = d_slice

            if mc_adaptive_enabled and trial_synthetic_data is not None and replacement_info is not None:
                current_daily_data_ohlc = d_slice.copy()
                
                for asset in replacement_info.selected_assets:
                    if asset in trial_synthetic_data:
                        synthetic_ohlc_for_asset = trial_synthetic_data[asset]
                        
                        window_synthetic_ohlc = synthetic_ohlc_for_asset.loc[te_start:te_end]
                        
                        if not window_synthetic_ohlc.empty:
                            if isinstance(current_daily_data_ohlc.columns, pd.MultiIndex):
                                for field in window_synthetic_ohlc.columns:
                                    if (asset, field) in current_daily_data_ohlc.columns:
                                        current_daily_data_ohlc.loc[window_synthetic_ohlc.index, (asset, field)] = window_synthetic_ohlc[field]
                            else:
                                for field in window_synthetic_ohlc.columns:
                                    col_name = f"{asset}_{field}"
                                    if col_name in current_daily_data_ohlc.columns:
                                        current_daily_data_ohlc.loc[window_synthetic_ohlc.index, col_name] = window_synthetic_ohlc[field]
                                    elif field == 'Close' and asset in current_daily_data_ohlc.columns:
                                        current_daily_data_ohlc.loc[window_synthetic_ohlc.index, asset] = window_synthetic_ohlc[field]
                                    else:
                                        if logger.isEnabledFor(logging.WARNING):
                                            logger.warning(f"Could not find column {col_name} or {asset} for synthetic data replacement in current_daily_data_ohlc.")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Stage 1 MC: Applied synthetic data to current_daily_data_ohlc for window {window_idx+1}")
            
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

            if _global_progress_tracker:
                _global_progress_tracker['progress'].update(_global_progress_tracker['task'], advance=1)

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

        if asset_replacement_manager is not None and trial and hasattr(trial, "set_user_attr"):
            replacement_stats = asset_replacement_manager.get_replacement_statistics()
            trial.set_user_attr("monte_carlo_replacement_stats", replacement_stats)

        if all_window_returns:
            full_pnl_returns = pd.concat(all_window_returns).sort_index()
            full_pnl_returns = full_pnl_returns[~full_pnl_returns.index.duplicated(keep='first')]
            
            bench_ser = daily_data[self.global_config["benchmark"]].loc[full_pnl_returns.index]
            bench_period_rets = bench_ser.pct_change(fill_method=None).fillna(0)
            
            final_metrics = calculate_metrics(full_pnl_returns, bench_period_rets, self.global_config["benchmark"])
            metric_avgs = [final_metrics.get(m, np.nan) for m in metrics_to_optimize]

        else:
            full_pnl_returns = pd.Series(dtype=float)
            metric_avgs = [np.nan for _ in metrics_to_optimize]


        if all(np.isnan(np.array(metric_avgs))):
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(f"No valid windows produced results for params: {scenario_config['strategy_params']}. Returning NaN.")
            if is_multi_objective:
                return tuple([float("nan")] * len(metrics_to_optimize))
            else:
                return float("nan")

        if trial and hasattr(trial, "set_user_attr"):
            trial.set_user_attr("full_pnl_returns", full_pnl_returns.to_json())

        if is_multi_objective:
            return tuple(float(v) for v in metric_avgs)
        else:
            return float(metric_avgs[0])
    
    def _evaluate_single_window(self, window_config, scenario_config, shared_data):
        from .reporting.performance_metrics import calculate_metrics
        
        window_idx = window_config['window_idx']
        tr_start = window_config['tr_start']
        tr_end = window_config['tr_end']
        te_start = window_config['te_start']
        te_end = window_config['te_end']
        
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

            current_daily_data_ohlc = d_slice
            
            if mc_adaptive_enabled and trial_synthetic_data is not None and replacement_info is not None:
                current_daily_data_ohlc = d_slice.copy()
                
                for asset in replacement_info.selected_assets:
                    if asset in trial_synthetic_data:
                        synthetic_ohlc_for_asset = trial_synthetic_data[asset]
                        
                        window_synthetic_ohlc = synthetic_ohlc_for_asset.loc[te_start:te_end]
                        
                        if not window_synthetic_ohlc.empty:
                            if isinstance(current_daily_data_ohlc.columns, pd.MultiIndex):
                                for field in window_synthetic_ohlc.columns:
                                    if (asset, field) in current_daily_data_ohlc.columns:
                                        current_daily_data_ohlc.loc[window_synthetic_ohlc.index, (asset, field)] = window_synthetic_ohlc[field]
                            else:
                                for field in window_synthetic_ohlc.columns:
                                    col_name = f"{asset}_{field}"
                                    if col_name in current_daily_data_ohlc.columns:
                                        current_daily_data_ohlc.loc[window_synthetic_ohlc.index, col_name] = window_synthetic_ohlc[field]
                                    elif field == 'Close' and asset in current_daily_data_ohlc.columns:
                                        current_daily_data_ohlc.loc[window_synthetic_ohlc.index, asset] = window_synthetic_ohlc[field]
            
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
        thresholds = {
            'fast': 20,
            'balanced': 10,    
            'comprehensive': 5
        }
        return thresholds.get(optimization_mode, 10)

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
        import os
        from .utils import _df_to_float32_array
        
        use_fast = os.environ.get("ENABLE_NUMBA_WALKFORWARD", "0") == "1"
        if not use_fast:
            objective_value = self._evaluate_params_walk_forward(
                trial, scenario_config, windows, monthly_data, daily_data, rets_full,
                metrics_to_optimize, is_multi_objective
            )
            full_pnl_returns = pd.Series(dtype=float)
            if trial and hasattr(trial, 'user_attrs') and 'full_pnl_returns' in trial.user_attrs:
                pnl_dict = trial.user_attrs['full_pnl_returns']
                if isinstance(pnl_dict, dict):
                    full_pnl_returns = pd.Series(pnl_dict)
                    full_pnl_returns.index = pd.to_datetime(full_pnl_returns.index)
            return objective_value, full_pnl_returns
        
        monte_carlo_config = self.global_config.get('monte_carlo_config', {})
        mc_enabled = monte_carlo_config.get('enable_synthetic_data', False)
        mc_during_optimization = monte_carlo_config.get('enable_during_optimization', True)
        
        robustness_config = self.global_config.get("wfo_robustness_config", {})
        window_randomization = robustness_config.get("enable_window_randomization", False)
        start_randomization = robustness_config.get("enable_start_date_randomization", False)
        
        if (mc_enabled and mc_during_optimization) or window_randomization or start_randomization:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Fast path disabled due to Monte-Carlo or randomization features - using legacy path")
            objective_value = self._evaluate_params_walk_forward(
                trial, scenario_config, windows, monthly_data, daily_data, rets_full,
                metrics_to_optimize, is_multi_objective
            )
            full_pnl_returns = pd.Series(dtype=float)
            if trial and hasattr(trial, 'user_attrs') and 'full_pnl_returns' in trial.user_attrs:
                pnl_dict = trial.user_attrs['full_pnl_returns']
                if isinstance(pnl_dict, dict):
                    full_pnl_returns = pd.Series(pnl_dict)
                    full_pnl_returns.index = pd.to_datetime(full_pnl_returns.index)
            return objective_value, full_pnl_returns
        
        try:
            if self._daily_index_cache is None or not daily_data.index.equals(pd.Index(self._daily_index_cache)):
                self._daily_index_cache = daily_data.index.to_numpy()
            
            monthly_data_np, _ = _df_to_float32_array(monthly_data)
            daily_data_np, _ = _df_to_float32_array(daily_data)
            rets_full_np, _ = _df_to_float32_array(rets_full)
            
            objective_value = self._evaluate_walk_forward_fast(
                trial, scenario_config, windows, monthly_data_np, daily_data_np, rets_full_np,
                metrics_to_optimize, is_multi_objective
            )
            
            full_pnl_returns = pd.Series(dtype=float)
            
            return objective_value, full_pnl_returns
            
        except Exception as exc:
            logger.error("Fast evaluation failed - falling back to legacy: %s", exc)
            objective_value = self._evaluate_params_walk_forward(
                trial, scenario_config, windows, monthly_data, daily_data, rets_full,
                metrics_to_optimize, is_multi_objective
            )
            full_pnl_returns = pd.Series(dtype=float)
            if trial and hasattr(trial, 'user_attrs') and 'full_pnl_returns' in trial.user_attrs:
                pnl_dict = trial.user_attrs['full_pnl_returns']
                if isinstance(pnl_dict, dict):
                    full_pnl_returns = pd.Series(pnl_dict)
                    full_pnl_returns.index = pd.to_datetime(full_pnl_returns.index)
            return objective_value, full_pnl_returns

    def run(self):
        if self.has_timed_out:
            logger.warning("Timeout reached before starting the backtest run.")
            return
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Starting backtest data retrieval.")

        all_tickers = set(self.global_config["universe"])
        all_tickers.add(self.global_config["benchmark"])

        for scenario_config in self.scenarios:
            if "universe" in scenario_config:
                all_tickers.update(scenario_config["universe"])
            strategy = self._get_strategy(
                scenario_config["strategy"], scenario_config["strategy_params"]
            )
            non_universe_tickers = strategy.get_non_universe_data_requirements()
            all_tickers.update(non_universe_tickers)

        daily_data = self.data_source.get_data(
            tickers=list(all_tickers),
            start_date=self.global_config["start_date"],
            end_date=self.global_config["end_date"],
        )

        if daily_data is None or daily_data.empty:
            logger.critical("No data fetched from data source. Aborting backtest run.")
            raise ValueError("daily_data is None after data source fetch. Cannot proceed.")

        daily_data.dropna(how="all", inplace=True)

        if isinstance(daily_data.columns, pd.MultiIndex) and daily_data.columns.names[0] != 'Ticker':
            daily_data_std_format = daily_data.stack(level=1).unstack(level=0)
        else:
            daily_data_std_format = daily_data

        if isinstance(daily_data_std_format, pd.Series):
            self.daily_data_ohlc = daily_data_std_format.to_frame()
        else:
            self.daily_data_ohlc = daily_data_std_format
        
        if self.daily_data_ohlc is not None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Shape of self.daily_data_ohlc: {self.daily_data_ohlc.shape}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Columns of self.daily_data_ohlc: {self.daily_data_ohlc.columns}")


        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Feature pre-computation step removed. Features will be calculated within strategies.")

        daily_closes = None
        if self.daily_data_ohlc is not None:
            if isinstance(self.daily_data_ohlc.columns, pd.MultiIndex) and \
               'Close' in self.daily_data_ohlc.columns.get_level_values(1):
                daily_closes = self.daily_data_ohlc.xs('Close', level='Field', axis=1)
            elif not isinstance(self.daily_data_ohlc.columns, pd.MultiIndex):
                daily_closes = self.daily_data_ohlc
            else:
                try:
                    if 'Close' in self.daily_data_ohlc.columns.get_level_values(-1):
                        daily_closes = self.daily_data_ohlc.xs('Close', level=-1, axis=1)
                    else:
                        raise ValueError("Could not reliably extract 'Close' prices from self.daily_data_ohlc due to unrecognized column structure.")
                except Exception as e:
                     raise ValueError(f"Error extracting 'Close' prices from self.daily_data_ohlc: {e}. Columns: {self.daily_data_ohlc.columns}")

        if daily_closes is None or daily_closes.empty:
            raise ValueError("Daily close prices could not be extracted or are empty.")

        if isinstance(daily_closes, pd.Series):
            daily_closes = daily_closes.to_frame()

        monthly_closes = daily_closes.resample("BME").last()
        self.monthly_data = monthly_closes.to_frame() if isinstance(monthly_closes, pd.Series) else monthly_closes

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Backtest data retrieved and prepared (daily OHLC, monthly closes).")

        rets_full = self.data_cache.get_cached_returns(daily_closes, "full_period_returns")
        self.rets_full = rets_full.to_frame() if isinstance(rets_full, pd.Series) else rets_full

        if self.args.mode == "optimize":
            self.run_optimize_mode(self.scenarios[0], self.monthly_data, self.daily_data_ohlc, rets_full)
        elif self.args.mode == "backtest":
            self.run_backtest_mode(self.scenarios[0], self.monthly_data, self.daily_data_ohlc, rets_full)
        
        if CENTRAL_INTERRUPTED_FLAG:
            self.logger.warning("Operation interrupted by user. Skipping final results display and plotting.")
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("All scenarios completed. Displaying results.")
            if self.args.mode == "backtest":
                self.display_results(daily_data)
            else:
                try:
                    self.generate_deferred_report()
                except Exception as e:
                    self.logger.warning(f"Failed to generate deferred report: {e}")
                
                self.display_results(self.daily_data_ohlc)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Optimization mode completed. Reports generated.")


