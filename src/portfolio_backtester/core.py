import logging
import time
import types
import sys
import argparse
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import optuna
import pandas as pd

from . import strategies
from .strategies.base.base_strategy import BaseStrategy
from .strategies import enumerate_strategies_with_params
from .backtester_logic.execution import run_backtest_mode
from .backtester_logic.strategy_logic import generate_signals, size_positions
from .config_initializer import populate_default_optimizations
from .config_loader import OPTIMIZER_PARAMETER_DEFAULTS
from .data_cache import get_global_cache
from .portfolio.position_sizer import get_position_sizer, SIZER_PARAM_MAPPING
from .backtester_logic.portfolio_logic import calculate_portfolio_returns
from .backtester_logic.data_manager import get_data_source, prepare_scenario_data
# Import display function for user-visible results
from .backtester_logic.reporting import display_results
from .api_stability import api_stable
# Import the optimization report generator and alias for internal use
from .backtester_logic.reporting_logic import generate_optimization_report as _generate_optimization_report
from .utils import INTERRUPTED as CENTRAL_INTERRUPTED_FLAG
from .utils.timeout import TimeoutManager
from .optimization.results import OptimizationData

logger = logging.getLogger(__name__)





class Backtester:
    @api_stable(version="1.0", strict_params=True, strict_return=False)
    def __init__(
        self,
        global_config: Dict[str, Any],
        scenarios: List[Dict[str, Any]],
        args: argparse.Namespace,
        random_state: Optional[int] = None
    ) -> None:
        self.global_config: Dict[str, Any] = global_config
        self.global_config["optimizer_parameter_defaults"] = OPTIMIZER_PARAMETER_DEFAULTS
        self.scenarios: List[Dict[str, Any]] = scenarios
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Backtester initialized with scenario strategy_params: {self.scenarios[0].get('strategy_params')}")
        populate_default_optimizations(self.scenarios, OPTIMIZER_PARAMETER_DEFAULTS)
        self.args: argparse.Namespace = args
        # Always use a wall-clock reference for timeout, for multiprocessing safety
        self._timeout_start_time: float = time.time()
        self.timeout_manager: TimeoutManager = TimeoutManager(args.timeout, start_time=self._timeout_start_time)
        self.strategy_map: Dict[str, type] = {name: klass for name, klass in enumerate_strategies_with_params().items()}
        self.data_source: Any = get_data_source(self.global_config)
        self.results: Dict[str, Any] = {}
        self.monthly_data: Optional[pd.DataFrame] = None
        self.daily_data_ohlc: Optional[pd.DataFrame] = None
        self.rets_full: Optional[Union[pd.DataFrame, pd.Series]] = None
        if random_state is None:
            self.random_state = np.random.randint(0, 2**31 - 1)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"No random seed provided. Using generated seed: {self.random_state}.")
        else:
            self.random_state = random_state
        self._windows_precomputed: bool = False
        np.random.seed(self.random_state)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Numpy random seed set to {self.random_state}.")
        self.n_jobs: int = getattr(args, "n_jobs", 1)
        self.early_stop_patience: int = getattr(args, "early_stop_patience", 10)
        self.logger = logger
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Backtester initialized.")

        self.data_cache: Any = get_global_cache()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Data preprocessing cache initialized")

        self._daily_index_cache: Optional[np.ndarray] = None
        self._daily_prices_np_cache: Optional[np.ndarray] = None

        self.asset_replacement_manager: Any = None
        self.synthetic_data_generator: Any = None
        if self.global_config.get('enable_synthetic_data', False):
            self._initialize_monte_carlo_components()
            if self.asset_replacement_manager is not None:
                self.asset_replacement_manager.set_full_data_source(self.data_source, self.global_config)

        # Remove dynamic method binding to fix multiprocessing pickling issues
        # Methods will be called directly from their modules


    
    @property
    def has_timed_out(self):
        return self.timeout_manager.check_timeout()

    

    def _get_strategy(self, strategy_name: str, params: Dict[str, Any]) -> BaseStrategy:
        strategy_class = self.strategy_map.get(strategy_name)
        if strategy_class is None:
            logger.error(f"Unsupported strategy: {strategy_name}")
            raise ValueError(f"Unsupported strategy: {strategy_name}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Using {strategy_class.__name__} with params: {params}")
        result = strategy_class(params)
        if not isinstance(result, BaseStrategy):
            raise TypeError(f"Strategy class {strategy_class} did not return a BaseStrategy instance.")
        return result
    
    def _initialize_monte_carlo_components(self) -> None:
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

    


    

    

    

    @api_stable(version="1.0", strict_params=True, strict_return=True)
    def run_scenario(
        self,
        scenario_config: Dict[str, Any],
        price_data_monthly_closes: pd.DataFrame,
        price_data_daily_ohlc: pd.DataFrame,
        rets_daily: Optional[pd.DataFrame] = None,
        verbose: bool = True,
    ) -> Optional[pd.Series]:
        """
        [API STABILITY NOTE]
        This method is protected by the @api_stable decorator to ensure its signature remains stable for critical workflows.
        
        TEMPORARY CHANGE (for test): The 'price_data_daily_ohlc' parameter was commented out to simulate a breaking change and test the API stability protection system. This change has now been reverted.
        
        If you need to restore the original method, ensure the signature includes:
            price_data_daily_ohlc: pd.DataFrame
        as the third parameter, as shown above.
        
        Any changes to this signature should be made with caution and must be validated by the API stability test suite.
        """
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
            return None

        benchmark_ticker = self.global_config["benchmark"]

        price_data_monthly_closes, rets_daily = prepare_scenario_data(price_data_daily_ohlc, self.data_cache)

        # Pass a callable to lazily check timeout status during signal generation
        signals = generate_signals(
            strategy,
            scenario_config,
            price_data_daily_ohlc,
            universe_tickers,
            benchmark_ticker,
            lambda: self.has_timed_out,
        )

        sized_signals = size_positions(signals, scenario_config, price_data_monthly_closes, price_data_daily_ohlc, universe_tickers, benchmark_ticker)

        # Calculate portfolio returns (no trade tracking for optimization)
        result = calculate_portfolio_returns(sized_signals, scenario_config, price_data_daily_ohlc, rets_daily, universe_tickers, self.global_config, track_trades=False)
        
        # Handle both old and new return formats
        if isinstance(result, tuple):
            portfolio_rets_net, _ = result
        else:
            portfolio_rets_net = result

        if verbose:
            if logger.isEnabledFor(logging.DEBUG):
                scenario_name = scenario_config['name']
                logger.debug(f"Portfolio net returns calculated for {scenario_name}. First few net returns: {portfolio_rets_net.head().to_dict()}")
                logger.debug(f"Net returns index: {portfolio_rets_net.index.min()} to {portfolio_rets_net.index.max()}")

        if not isinstance(portfolio_rets_net, (pd.Series, type(None))):
            raise TypeError("run_scenario must return a pd.Series or None")
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
        signals: np.ndarray,
        strategy_instance: BaseStrategy,
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
                metrics_mat = _nk.run_backtest_fast(port_rets, test_starts, test_ends, signals)
                avg_metric = float(np.nanmean(metrics_mat))
                return avg_metric if not is_multi_objective else (avg_metric,)
            except Exception as exc:
                logger.error("Fast walk-forward path failed – falling back to legacy: %s", exc)
        # Fallback: use new architecture components
        from .backtesting.strategy_backtester import StrategyBacktester
        from .optimization.results import OptimizationData
        from .optimization.evaluator import BacktestEvaluator
        
        monthly_data = pd.DataFrame(monthly_data_np) if monthly_data_np is not None else pd.DataFrame()
        daily_data = pd.DataFrame(daily_data_np) if daily_data_np is not None else pd.DataFrame()
        rets_full = pd.DataFrame(rets_full_np) if rets_full_np is not None else pd.DataFrame()
        
        # Create new architecture components
        strategy_backtester = StrategyBacktester(self.global_config, self.data_source)
        evaluator = BacktestEvaluator(
            metrics_to_optimize=metrics_to_optimize,
            is_multi_objective=is_multi_objective
        )
        
        # Create optimization data
        optimization_data = OptimizationData(
            monthly=monthly_data,
            daily=daily_data,
            returns=rets_full,
            windows=windows
        )
        
        # Extract parameters from trial
        parameters = scenario_config.get('strategy_params', {}).copy()
        if hasattr(trial, 'params') and trial.params is not None:
            parameters.update(trial.params)
        elif hasattr(trial, 'user_attrs') and 'parameters' in trial.user_attrs:
            parameters.update(trial.user_attrs['parameters'])
        
        # Evaluate using new architecture
        evaluation_result = evaluator.evaluate_parameters(
            parameters, scenario_config, optimization_data, strategy_backtester
        )
        
        return evaluation_result.objective_value


    
    def _evaluate_single_window(self, window_config: Dict[str, Any], scenario_config: Dict[str, Any], shared_data: Dict[str, Any]) -> Tuple[List[float], pd.Series]:
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

    def _get_monte_carlo_trial_threshold(self, optimization_mode: str) -> int:
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
            # Use new architecture components
            from .backtesting.strategy_backtester import StrategyBacktester
            from .optimization.results import OptimizationData
            from .optimization.evaluator import BacktestEvaluator
            
            # Create new architecture components
            strategy_backtester = StrategyBacktester(self.global_config, self.data_source)
            evaluator = BacktestEvaluator(
                metrics_to_optimize=metrics_to_optimize,
                is_multi_objective=is_multi_objective
            )
            
            # Create optimization data
            optimization_data = OptimizationData(
                monthly=monthly_data,
                daily=daily_data,
                returns=rets_full,
                windows=windows
            )
            
            # Extract parameters from trial
            parameters = scenario_config.get('strategy_params', {}).copy()
            if hasattr(trial, 'params') and trial.params is not None:
                parameters.update(trial.params)
            elif hasattr(trial, 'user_attrs') and 'parameters' in trial.user_attrs:
                parameters.update(trial.user_attrs['parameters'])
            
            # Evaluate using new architecture
            evaluation_result = evaluator.evaluate_parameters(
                parameters, scenario_config, optimization_data, strategy_backtester
            )
            
            objective_value = evaluation_result.objective_value
            full_pnl_returns = pd.Series(dtype=float)
            if trial and hasattr(trial, 'user_attrs') and 'full_pnl_returns' in trial.user_attrs:
                pnl_dict = trial.user_attrs['full_pnl_returns']
                if isinstance(pnl_dict, dict):
                    full_pnl_returns = pd.Series(pnl_dict)
                    full_pnl_returns.index = pd.to_datetime(full_pnl_returns.index)
            return objective_value, full_pnl_returns
        
        try:
            # --------------------------------------------------------------
            # Cache daily index and price matrix so that repeated calls (one
            # per walk-forward window) in the *same process* avoid expensive
            # pandas-to-NumPy conversions.
            # --------------------------------------------------------------
            # Attributes are now always present from __init__

            # Update cache if DataFrame index changed (different data slice)
            if self._daily_index_cache is None or not daily_data.index.equals(pd.Index(self._daily_index_cache)):
                # Convert to NumPy datetime64 for fast, type-stable searchsorted
                self._daily_index_cache = daily_data.index.values.astype('datetime64[ns]')

                # Handle Multi-Index vs single-level columns only when cache invalidated
                if isinstance(daily_data.columns, pd.MultiIndex):
                    self._daily_prices_np_cache, _ = _df_to_float32_array(daily_data, field='Close')
                else:
                    self._daily_prices_np_cache, _ = _df_to_float32_array(daily_data)

            prices_np = self._daily_prices_np_cache
                
            strategy_class = self.strategy_map.get(scenario_config["strategy"])
            if strategy_class is None:
                logger.error(f"Unsupported strategy: {scenario_config['strategy']}")
                return np.nan, pd.Series(dtype=float)
            strategy_instance = strategy_class(scenario_config["strategy_params"])
            # ----------------------------------------------------------
            # Build a *full-length* signal matrix aligned with the daily
            # price DataFrame – same logic as evaluate_fast_numba.
            # ----------------------------------------------------------

            if isinstance(daily_data.columns, pd.MultiIndex):
                price_df_for_cols = daily_data.xs('Close', level='Field', axis=1)
            else:
                price_df_for_cols = daily_data

            tickers = list(price_df_for_cols.columns)

            signals_df = pd.DataFrame(
                data=np.nan,
                index=daily_data.index,
                columns=tickers,
                dtype=np.float32,
            )

            for _, _, te_start, _ in windows:
                try:
                    w_sig = strategy_instance.generate_signals(
                        monthly_data,
                        daily_data,
                        rets_full,
                        te_start,
                        None,
                        None,
                    )
                    if w_sig is not None and not w_sig.empty:
                        for col in w_sig.columns:
                            if col in signals_df.columns:
                                signals_df.at[te_start, col] = w_sig.iloc[0][col]
                except Exception as sig_exc:
                    logger.error(
                        "Signal generation failed for window start %s: %s",
                        te_start,
                        sig_exc,
                    )

            signals_df.ffill(inplace=True)
            signals_df.fillna(0.0, inplace=True)

            signals = signals_df

            # Convert DataFrames to numpy arrays for fast evaluation
            from .utils import _df_to_float32_array
            monthly_data_np, _ = _df_to_float32_array(monthly_data)
            daily_data_np, _ = _df_to_float32_array(daily_data)
            rets_full_np, _ = _df_to_float32_array(rets_full)

            objective_value = self._evaluate_walk_forward_fast(
                trial, scenario_config, windows, monthly_data_np, daily_data_np, rets_full_np,
                metrics_to_optimize, is_multi_objective, signals.to_numpy(), strategy_instance
            )

            full_pnl_returns = pd.Series(dtype=float)

            return objective_value, full_pnl_returns
            
        except Exception as exc:
            logger.error("Fast evaluation failed - falling back to new architecture: %s", exc)
            # Use new architecture components as fallback
            from .backtesting.strategy_backtester import StrategyBacktester
            from .optimization.results import OptimizationData
            from .optimization.evaluator import BacktestEvaluator
            
            # Create new architecture components
            strategy_backtester = StrategyBacktester(self.global_config, self.data_source)
            evaluator = BacktestEvaluator(
                metrics_to_optimize=metrics_to_optimize,
                is_multi_objective=is_multi_objective
            )
            
            # Create optimization data
            optimization_data = OptimizationData(
                monthly=monthly_data,
                daily=daily_data,
                returns=rets_full,
                windows=windows
            )
            
            # Extract parameters from trial
            parameters = scenario_config.get('strategy_params', {}).copy()
            if hasattr(trial, 'params') and trial.params is not None:
                parameters.update(trial.params)
            elif hasattr(trial, 'user_attrs') and 'parameters' in trial.user_attrs:
                parameters.update(trial.user_attrs['parameters'])
            
            # Evaluate using new architecture
            evaluation_result = evaluator.evaluate_parameters(
                parameters, scenario_config, optimization_data, strategy_backtester
            )
            
            objective_value = evaluation_result.objective_value
            full_pnl_returns = pd.Series(dtype=float)
            if trial and hasattr(trial, 'user_attrs') and 'full_pnl_returns' in trial.user_attrs:
                pnl_dict = trial.user_attrs['full_pnl_returns']
                if isinstance(pnl_dict, dict):
                    full_pnl_returns = pd.Series(pnl_dict)
                    full_pnl_returns.index = pd.to_datetime(full_pnl_returns.index)
            return objective_value, full_pnl_returns

    def evaluate_fast_numba(
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
        from .numba_kernels import run_backtest_numba
        from .utils import _df_to_float32_array

        try:
            # --------------------------------------------------------------
            # Cache daily index and price matrix so that repeated calls (one
            # per walk-forward window) in the *same process* avoid expensive
            # pandas-to-NumPy conversions.
            # --------------------------------------------------------------
            # Attributes are now always present from __init__

            # Update cache if DataFrame index changed (different data slice)
            if self._daily_index_cache is None or not daily_data.index.equals(pd.Index(self._daily_index_cache)):
                self._daily_index_cache = daily_data.index.to_numpy()

                # Handle Multi-Index vs single-level columns only when cache invalidated
                if isinstance(daily_data.columns, pd.MultiIndex):
                    self._daily_prices_np_cache, _ = _df_to_float32_array(daily_data, field='Close')
                else:
                    self._daily_prices_np_cache, _ = _df_to_float32_array(daily_data)

            prices_np = self._daily_prices_np_cache
            
            strategy_class = self.strategy_map.get(scenario_config["strategy"])
            if strategy_class is None:
                logger.error(f"Unsupported strategy: {scenario_config['strategy']}")
                return np.nan, pd.Series(dtype=float)
            strategy_instance = strategy_class(scenario_config["strategy_params"])

            # ----------------------------------------------------------
            # Build a *full-length* signals matrix that matches exactly
            # the shape of the price matrix used by the Numba kernel.
            # We only calculate a signal on each walk-forward test window
            # start date (``te_start``) and then forward-fill so the
            # positions remain constant until the next re-optimisation
            # point.  Any days prior to the first window carry zero
            # exposure.
            # ----------------------------------------------------------

            # Determine the list of tickers (same order as the price
            # matrix extracted via ``_df_to_float32_array``).
            logger.debug("Determining tickers for signal generation.")
            if isinstance(daily_data.columns, pd.MultiIndex):
                # Extract just the *Close* field to obtain single-level
                # columns that represent tickers only.
                price_df_for_cols = daily_data.xs('Close', level='Field', axis=1)
            else:
                price_df_for_cols = daily_data

            tickers = list(price_df_for_cols.columns)
            logger.debug(f"Tickers for signal generation: {tickers}")

            # Initialise signals DataFrame with NaNs so we can detect
            # unfilled periods explicitly (filled with 0.0 later).
            logger.debug("Initializing signals DataFrame.")
            signals_df = pd.DataFrame(
                data=np.nan,
                index=daily_data.index,
                columns=tickers,
                dtype=np.float32,
            )
            logger.debug("Successfully initialized signals DataFrame.")

            # Generate signals only on each test window *start* date.
            logger.debug("Starting signal generation loop.")
            for _, _, te_start, _ in windows:
                try:
                    logger.debug(f"Generating signal for window starting at {te_start}")
                    window_signal = strategy_instance.generate_signals(
                        monthly_data,
                        daily_data,
                        rets_full,
                        te_start,
                        None,
                        None,
                    )
                    logger.debug(f"Successfully generated signal for window starting at {te_start}")

                    if window_signal is not None and not window_signal.empty:
                        logger.debug(f"Signal for {te_start}:\n{window_signal}")
                        # Align columns – some strategies may return a
                        # subset of tickers; missing tickers default to 0.
                        for col in window_signal.columns:
                            if col in signals_df.columns:
                                signals_df.at[te_start, col] = window_signal.iloc[0][col]
                        logger.debug(f"Successfully assigned signal for window starting at {te_start}")
                except Exception as sig_exc:
                    logger.error(
                        "Signal generation failed for window start %s: %s",
                        te_start,
                        sig_exc,
                    )
            logger.debug("Finished signal generation loop.")

            # Forward-fill to make positions persistent until next signal
            # update, then replace any leading NaNs (prior to first
            # window) with zero exposure.
            signals_df.ffill(inplace=True)
            signals_df.fillna(0.0, inplace=True)

            # Convert to float32 NumPy array for Numba kernel.
            signals_np, _ = _df_to_float32_array(signals_df)
             
            start_indices = np.asarray([np.searchsorted(self._daily_index_cache, np.datetime64(w[2])) for w in windows], dtype=np.int64)
            end_indices = np.asarray([np.searchsorted(self._daily_index_cache, np.datetime64(w[3])) for w in windows], dtype=np.int64)

            if prices_np is None or signals_np is None:
                logger.error("prices_np or signals_np is None. Cannot run numba backtest.")
                return np.nan, pd.Series(dtype=float)
            
            # Ensure correct dtype for Numba kernel
            prices_np = prices_np.astype(np.float32)
            signals_np = signals_np.astype(np.float32)
            
            portfolio_returns = run_backtest_numba(prices_np, signals_np, start_indices, end_indices)
            
            # For now, we'll just return the mean of the portfolio returns as the objective value
            objective_value: float | tuple[float, ...] = float(np.nanmean(portfolio_returns))
            full_pnl_returns = pd.Series(portfolio_returns, index=[w[2] for w in windows])
            return objective_value, full_pnl_returns

        except Exception as exc:
            logger.error("Numba evaluation failed: %s", exc)
            return np.nan, pd.Series(dtype=float)

    @api_stable(version="1.0", strict_params=True, strict_return=False)
    def run(self) -> None:
        if self.has_timed_out:
            logger.warning("Timeout reached before starting the backtest run.")
            return
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Starting backtest data retrieval.")

        if self.args.scenario_name:
            scenarios_to_run = [s for s in self.scenarios if s["name"] == self.args.scenario_name]
            if not scenarios_to_run:
                self.logger.error(f"Scenario '{self.args.scenario_name}' not found in the loaded scenarios.")
                return
        else:
            scenarios_to_run = self.scenarios

        all_tickers = set(self.global_config.get("universe", []))
        all_tickers.add(self.global_config["benchmark"])

        for scenario_config in scenarios_to_run:
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

        # Determine mode and use appropriate architecture
        if self.args and hasattr(self.args, 'mode') and self.args.mode == "optimize":
            self._run_optimize_mode_new_architecture(scenarios_to_run[0], self.monthly_data, self.daily_data_ohlc, rets_full)
        else:
            self._run_backtest_mode_new_architecture(scenarios_to_run[0], self.monthly_data, self.daily_data_ohlc, rets_full)
        
        if CENTRAL_INTERRUPTED_FLAG:
            self.logger.warning("Operation interrupted by user. Skipping final results display and plotting.")
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("All scenarios completed. Displaying results.")
            if self.args.mode == "backtest":
                display_results(self, daily_data)
            else:
                try:
                    from .backtester_logic.execution import generate_deferred_report
                    generate_deferred_report(self)
                except Exception as e:
                    self.logger.warning(f"Failed to generate deferred report: {e}")
                
                if self.daily_data_ohlc is not None:
                    display_results(self, self.daily_data_ohlc)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Optimization mode completed. Reports generated.")

    def _run_backtest_mode_new_architecture(
        self, 
        scenario_config: Dict[str, Any], 
        monthly_data: pd.DataFrame, 
        daily_data: pd.DataFrame, 
        rets_full: pd.DataFrame
    ) -> None:
        """Run backtest mode using the new StrategyBacktester architecture.
        
        This method uses the pure StrategyBacktester directly for backtesting.
        
        Args:
            scenario_config: Scenario configuration
            monthly_data: Monthly price data
            daily_data: Daily OHLC data
            rets_full: Full period returns data
        """
        from .backtesting.strategy_backtester import StrategyBacktester
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Running backtest using new architecture for scenario: {scenario_config['name']}")

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
                self.logger.warning(f"Study '{self.args.study_name}' not found. Using default parameters for scenario '{scenario_config['name']}'.")
            except Exception as e:
                self.logger.error(f"Error loading Optuna study: {e}. Using default parameters.")

        strategy_backtester = StrategyBacktester(self.global_config, self.data_source)
        backtest_result = strategy_backtester.backtest_strategy(
            scenario_config, monthly_data, daily_data, rets_full
        )
        
        train_end_date = pd.to_datetime(scenario_config.get("train_end_date", "2018-12-31"))
        
        self.results[scenario_config["name"]] = {
            "returns": backtest_result.returns, 
            "display_name": scenario_config["name"], 
            "train_end_date": train_end_date,
            "trade_stats": backtest_result.trade_stats,
            "trade_history": backtest_result.trade_history,
            "performance_stats": backtest_result.performance_stats,
            "charts_data": backtest_result.charts_data
        }

    def _run_optimize_mode_new_architecture(
        self, 
        scenario_config: Dict[str, Any], 
        monthly_data: pd.DataFrame, 
        daily_data: pd.DataFrame, 
        rets_full: pd.DataFrame
    ) -> None:
        """Run optimization mode using the new OptimizationOrchestrator architecture.
        
        This method uses the new architecture with OptimizationOrchestrator,
        parameter generators, and BacktestEvaluator to perform optimization.
        
        Args:
            scenario_config: Scenario configuration
            monthly_data: Monthly price data
            daily_data: Daily OHLC data
            rets_full: Full period returns data
        """
        from .optimization.factory import create_parameter_generator
        from .optimization.orchestrator import OptimizationOrchestrator
        from .optimization.evaluator import BacktestEvaluator
        from .backtesting.strategy_backtester import StrategyBacktester
        from .optimization.results import OptimizationData
        from .utils import generate_randomized_wfo_windows
        from .backtester_logic.execution import run_optimize_mode
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Running optimization using new architecture for scenario: {scenario_config['name']}")
        
        # Get optimizer type from CLI args (this is the key integration point)
        optimizer_type = getattr(self.args, 'optimizer', 'optuna')
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Using optimizer type from CLI: {optimizer_type}")
        
        try:
            # Create parameter generator using factory
            parameter_generator = create_parameter_generator(
                optimizer_type=optimizer_type,
                random_state=self.random_state
            )
        except (ImportError, ValueError) as e:
            logger.error(f"Failed to create parameter generator '{optimizer_type}': {e}")
            raise

        try:
            # Create StrategyBacktester for pure backtesting
            strategy_backtester = StrategyBacktester(
                global_config=self.global_config,
                data_source=self.data_source
            )
            
            # Generate walk-forward windows
            windows = generate_randomized_wfo_windows(
                monthly_data.index,
                scenario_config,
                self.global_config,
                self.random_state
            )
            
            if not windows:
                raise ValueError("Not enough data for the requested walk-forward windows.")
            
            # Determine optimization targets and metrics
            optimization_targets_config = scenario_config.get("optimization_targets", [])
            metrics_to_optimize = [t["name"] for t in optimization_targets_config] or \
                                  [scenario_config.get("optimization_metric", "Calmar")]
            is_multi_objective = len(metrics_to_optimize) > 1
            
            # Prepare optimization data
            optimization_data = OptimizationData(
                monthly=monthly_data,
                daily=daily_data,
                returns=rets_full,
                windows=windows
            )
            
            # Convert scenario optimization specs to parameter space format
            parameter_space = self._convert_optimization_specs_to_parameter_space(scenario_config)
            
            # Create optimization config from CLI args and scenario config
            optimization_config = {
                'parameter_space': parameter_space,
                'max_evaluations': getattr(self.args, 'optuna_trials', 200),
                'timeout_seconds': getattr(self.args, 'optuna_timeout_sec', None),
                'optimization_targets': optimization_targets_config,
                'metrics_to_optimize': metrics_to_optimize,
                'pruning_enabled': getattr(self.args, 'pruning_enabled', False),
                'pruning_n_startup_trials': getattr(self.args, 'pruning_n_startup_trials', 5),
                'pruning_n_warmup_steps': getattr(self.args, 'pruning_n_warmup_trials', 0),
                'pruning_interval_steps': getattr(self.args, 'pruning_interval_steps', 1),
                'study_name': getattr(self.args, 'study_name', None),
                'storage_url': getattr(self.args, 'storage_url', None),
                'random_seed': self.random_state
            }
            
            # Create BacktestEvaluator
            evaluator = BacktestEvaluator(
                metrics_to_optimize=metrics_to_optimize,
                is_multi_objective=is_multi_objective
            )
            
            # Create OptimizationOrchestrator or ParallelOptimizationRunner
            optimization_result = None
            if optimizer_type == 'optuna':
                from .optimization.parallel_optimization_runner import ParallelOptimizationRunner
                parallel_runner = ParallelOptimizationRunner(
                    scenario_config=scenario_config,
                    optimization_config=optimization_config,
                    data=optimization_data,
                    n_jobs=getattr(self.args, 'n_jobs', 1),
                    storage_url=optimization_config.get('storage_url', 'sqlite:///optuna_studies.db'),
                )
                optimization_result = parallel_runner.run()
            else:
                orchestrator = OptimizationOrchestrator(
                    parameter_generator=parameter_generator,
                    evaluator=evaluator,
                    timeout_seconds=getattr(self.args, 'optuna_timeout_sec', None),
                    early_stop_patience=getattr(self.args, 'early_stop_patience', 10)
                )
                optimization_result = orchestrator.optimize(
                    scenario_config=scenario_config,
                    optimization_config=optimization_config,
                    data=optimization_data,
                    backtester=strategy_backtester
                )
            
            # Process results and update self.results for compatibility
            optimal_params = optimization_result.best_parameters
            optimized_scenario = scenario_config.copy()
            optimized_scenario["strategy_params"] = optimal_params
            
            # Run full backtest with optimal parameters
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Running full backtest with optimal parameters: {optimal_params}")
            
            full_backtest_result = strategy_backtester.backtest_strategy(
                optimized_scenario, monthly_data, daily_data, rets_full
            )
            
            # Store results in the expected format for compatibility
            optimized_name = f"{scenario_config['name']}_Optimized"
            train_end_date = pd.to_datetime(scenario_config.get("train_end_date", "2018-12-31"))
            
            self.results[optimized_name] = {
                "returns": full_backtest_result.returns,
                "display_name": optimized_name,
                "optimal_params": optimal_params,
                "num_trials_for_dsr": optimization_result.n_evaluations,
                "train_end_date": train_end_date,
                "best_trial_obj": optimization_result.best_trial,
                "constraint_status": "passed",  # TODO: Implement constraint handling
                "constraint_message": "",
                "constraint_violations": [],
                "constraints_config": {},
                "trade_stats": full_backtest_result.trade_stats,
                "trade_history": full_backtest_result.trade_history,
                "performance_stats": full_backtest_result.performance_stats,
                "charts_data": full_backtest_result.charts_data
            }
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"New architecture optimization completed for {scenario_config['name']}")
                
        except (ValueError, TypeError) as e:
            logger.error(f"Error during optimization setup: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during optimization: {e}")
            raise

    def _convert_optimization_specs_to_parameter_space(self, scenario_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert scenario optimization specs to parameter space format.
        
        This method converts the scenario configuration's 'optimize' section
        to the parameter space format expected by parameter generators.
        
        Args:
            scenario_config: Scenario configuration containing 'optimize' section
            
        Returns:
            Dictionary defining the parameter space for optimization
        """
        parameter_space = {}
        optimization_specs = scenario_config.get("optimize", [])
        
        for spec in optimization_specs:
            param_name = spec["parameter"]
            
            # Get parameter type from spec or defaults
            param_type = spec.get("type")
            if not param_type:
                if "min_value" in spec and "max_value" in spec:
                    # Infer type from min/max value
                    if isinstance(spec["min_value"], int) and isinstance(spec["max_value"], int):
                        param_type = "int"
                    else:
                        param_type = "float"
                else:
                    param_type = self.global_config.get("optimizer_parameter_defaults", {}).get(param_name, {}).get("type", "float")
            
            # Convert to parameter space format
            if param_type == "int":
                parameter_space[param_name] = {
                    "type": "int",
                    "low": spec["min_value"],
                    "high": spec["max_value"],
                    "step": spec.get("step", 1)
                }
            elif param_type == "float":
                parameter_space[param_name] = {
                    "type": "float", 
                    "low": spec["min_value"],
                    "high": spec["max_value"],
                    "step": spec.get("step", None)
                }
            elif param_type == "categorical":
                choices = spec.get("choices") or spec.get("values")
                if not choices:
                    logger.error(f"Categorical parameter '{param_name}' is missing 'choices' or 'values' in spec: {spec}")
                    raise KeyError(f"Categorical parameter '{param_name}' must have 'choices' or 'values' defined.")
                parameter_space[param_name] = {
                    "type": "categorical",
                    "choices": choices
                }
            elif param_type == "multi-categorical":
                choices = spec.get("choices") or spec.get("values")
                if not choices:
                    logger.error(f"Multi-categorical parameter '{param_name}' is missing 'choices' or 'values' in spec: {spec}")
                    raise KeyError(f"Multi-categorical parameter '{param_name}' must have 'choices' or 'values' defined.")
                parameter_space[param_name] = {
                    "type": "multi-categorical",
                    "values": choices
                }
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Converted optimization specs to parameter space: {parameter_space}")
        
        return parameter_space

    def evaluate_trial_parameters(self, scenario_config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, float]:
        """Evaluates a single set of parameters and returns performance metrics."""
        temp_scenario_config = scenario_config.copy()
        temp_scenario_config['strategy_params'] = params
        
        # This is a simplified call for demonstration.
        # A real implementation would need to handle the full backtest process.
        returns = self.run_scenario(
            temp_scenario_config, 
            self.monthly_data, 
            self.daily_data_ohlc, 
            self.rets_full, 
            verbose=False
        )
        
        if returns is None or returns.empty:
            return {metric: 0.0 for metric in self.metrics_to_optimize}

        from .reporting.performance_metrics import calculate_metrics
        benchmark_rets = self.daily_data_ohlc[self.global_config["benchmark"]].pct_change().fillna(0)
        metrics = calculate_metrics(returns, benchmark_rets, self.global_config["benchmark"])
        return {metric: metrics.get(metric, 0.0) for metric in self.metrics_to_optimize}

