"""
Evaluation engine logic extracted from Backtester class.

This module implements the EvaluationEngine class that handles all performance
evaluation operations including fast evaluation, walk-forward analysis, and parameter testing.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union, cast, Protocol

import numpy as np
import pandas as pd

from ..strategies._core.base.base_strategy import BaseStrategy
from ..strategies._core.strategy_factory import StrategyFactory
from ..optimization.results import OptimizationData
from ..interfaces.attribute_accessor_interface import (
    IAttributeAccessor,
    IModuleAttributeAccessor,
    create_attribute_accessor,
    create_module_attribute_accessor,
)

logger = logging.getLogger(__name__)


class EvaluationEngine:
    """
    Handles performance evaluation for backtesting and optimization.

    This class encapsulates all evaluation-related operations that were previously
    part of the Backtester class, following the Single Responsibility Principle.
    """

    def __init__(
        self,
        global_config: Dict[str, Any],
        data_source: Any,
        strategy_manager: Any,
        attribute_accessor: Optional[IAttributeAccessor] = None,
        module_accessor: Optional[IModuleAttributeAccessor] = None,
    ):
        """
        Initialize EvaluationEngine with configuration and dependencies.

        Args:
            global_config: Global configuration dictionary
            data_source: Data source instance for fetching market data
            strategy_manager: StrategyManager instance for creating strategies
            attribute_accessor: Injected accessor for attribute access (DIP)
            module_accessor: Injected accessor for module attribute access (DIP)
        """
        self.global_config = global_config
        self.data_source = data_source
        self.strategy_manager = strategy_manager
        self.logger = logger
        # Dependency injection for attribute access (DIP)
        self._attribute_accessor = attribute_accessor or create_attribute_accessor()
        self._module_accessor = module_accessor or create_module_attribute_accessor()

        # Cache for optimization performance
        self._daily_index_cache: Optional[np.ndarray] = None
        self._daily_prices_np_cache: Optional[np.ndarray] = None

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("EvaluationEngine initialized")

    def get_monte_carlo_trial_threshold(self, optimization_mode: str) -> int:
        """
        Get threshold for Monte Carlo trials based on optimization mode.

        Args:
            optimization_mode: Optimization mode ('fast', 'balanced', 'comprehensive')

        Returns:
            Threshold value for Monte Carlo trials
        """
        thresholds = {"fast": 20, "balanced": 10, "comprehensive": 5}
        return thresholds.get(optimization_mode, 10)

    def evaluate_single_window(
        self,
        window_config: Dict[str, Any],
        scenario_config: Dict[str, Any],
        shared_data: Dict[str, Any],
        run_scenario_func,
    ) -> Tuple[List[float], pd.Series]:
        """
        Evaluate a single walk-forward window.

        Args:
            window_config: Configuration for the window (indices, dates)
            scenario_config: Scenario configuration
            shared_data: Shared data including monthly/daily data, metrics
            run_scenario_func: Function to run scenario evaluation

        Returns:
            Tuple of (metrics_list, window_returns)
        """
        from ..reporting.performance_metrics import calculate_metrics

        window_idx = window_config["window_idx"]
        tr_start = window_config["tr_start"]
        tr_end = window_config["tr_end"]
        te_start = window_config["te_start"]
        te_end = window_config["te_end"]

        monthly_data = shared_data["monthly_data"]
        daily_data = shared_data["daily_data"]
        shared_data["rets_full"]
        trial_synthetic_data = shared_data.get("trial_synthetic_data")
        replacement_info = shared_data.get("replacement_info")
        mc_adaptive_enabled = shared_data.get("mc_adaptive_enabled", False)
        metrics_to_optimize = shared_data["metrics_to_optimize"]
        global_config = shared_data["global_config"]

        try:
            m_slice = monthly_data.loc[tr_start:tr_end]
            d_slice = daily_data.loc[tr_start:te_end]

            current_daily_data_ohlc = d_slice

            if (
                mc_adaptive_enabled
                and trial_synthetic_data is not None
                and replacement_info is not None
            ):
                current_daily_data_ohlc = d_slice.copy()

                for asset in replacement_info.selected_assets:
                    if asset in trial_synthetic_data:
                        synthetic_ohlc_for_asset = trial_synthetic_data[asset]

                        window_synthetic_ohlc = synthetic_ohlc_for_asset.loc[te_start:te_end]

                        if not window_synthetic_ohlc.empty:
                            if isinstance(current_daily_data_ohlc.columns, pd.MultiIndex):
                                for field in window_synthetic_ohlc.columns:
                                    if (asset, field) in current_daily_data_ohlc.columns:
                                        current_daily_data_ohlc.loc[
                                            window_synthetic_ohlc.index, (asset, field)
                                        ] = window_synthetic_ohlc[field]
                            else:
                                for field in window_synthetic_ohlc.columns:
                                    col_name = f"{asset}_{field}"
                                    if col_name in current_daily_data_ohlc.columns:
                                        current_daily_data_ohlc.loc[
                                            window_synthetic_ohlc.index, col_name
                                        ] = window_synthetic_ohlc[field]
                                    elif (
                                        field == "Close"
                                        and asset in current_daily_data_ohlc.columns
                                    ):
                                        current_daily_data_ohlc.loc[
                                            window_synthetic_ohlc.index, asset
                                        ] = window_synthetic_ohlc[field]

            window_returns = run_scenario_func(
                scenario_config, m_slice, current_daily_data_ohlc, rets_daily=None, verbose=False
            )

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

    def evaluate_walk_forward_fast(
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
        """
        Evaluate walk-forward optimization using fast path.

        Args:
            trial: Optimization trial object
            scenario_config: Scenario configuration
            windows: List of walk-forward windows
            monthly_data_np: Monthly data as numpy array
            daily_data_np: Daily data as numpy array
            rets_full_np: Returns data as numpy array
            metrics_to_optimize: List of metrics to optimize
            is_multi_objective: Whether this is multi-objective optimization
            signals: Signal matrix
            strategy_instance: Strategy instance

        Returns:
            Objective value (float for single objective, tuple for multi-objective)
        """

        # Always use new architecture evaluation path
        from ..backtesting.strategy_backtester import StrategyBacktester
        from ..optimization.evaluator import BacktestEvaluator

        monthly_data = (
            pd.DataFrame(monthly_data_np) if monthly_data_np is not None else pd.DataFrame()
        )
        daily_data = pd.DataFrame(daily_data_np) if daily_data_np is not None else pd.DataFrame()
        rets_full = pd.DataFrame(rets_full_np) if rets_full_np is not None else pd.DataFrame()

        # Create new architecture components
        strategy_backtester = StrategyBacktester(self.global_config, self.data_source)
        evaluator = BacktestEvaluator(
            metrics_to_optimize=metrics_to_optimize, is_multi_objective=is_multi_objective
        )

        # Create optimization data
        optimization_data = OptimizationData(
            monthly=monthly_data, daily=daily_data, returns=rets_full, windows=windows
        )

        # Extract parameters from trial
        parameters = scenario_config.get("strategy_params", {}).copy()
        if hasattr(trial, "params") and trial.params is not None:
            parameters.update(trial.params)
        elif hasattr(trial, "user_attrs") and "parameters" in trial.user_attrs:
            parameters.update(trial.user_attrs["parameters"])

        # Evaluate using new architecture
        evaluation_result = evaluator.evaluate_parameters(
            parameters, scenario_config, optimization_data, strategy_backtester
        )

        obj = evaluation_result.objective_value
        # Normalize to match annotation float | tuple[float, ...]
        if isinstance(obj, list):
            return tuple(obj)
        return cast(float | tuple[float, ...], obj)

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
        """
        Fast evaluation of parameters using optimized paths.

        Args:
            trial: Optimization trial object
            scenario_config: Scenario configuration
            windows: List of walk-forward windows
            monthly_data: Monthly price data
            daily_data: Daily OHLC data
            rets_full: Full period returns
            metrics_to_optimize: List of metrics to optimize
            is_multi_objective: Whether this is multi-objective optimization

        Returns:
            Tuple of (objective_value, full_pnl_returns)
        """
        import os
        from ..utils import _df_to_float32_array

        disable_env = os.environ.get("DISABLE_NUMBA_WALKFORWARD", "0") == "1"
        enable_env = os.environ.get("ENABLE_NUMBA_WALKFORWARD", "0") == "1"
        use_fast = enable_env or not disable_env
        if not use_fast:
            # Use new architecture components
            from ..backtesting.strategy_backtester import StrategyBacktester
            from ..optimization.results import OptimizationData
            from ..optimization.evaluator import BacktestEvaluator

            # Create new architecture components
            strategy_backtester = StrategyBacktester(self.global_config, self.data_source)
            evaluator = BacktestEvaluator(
                metrics_to_optimize=metrics_to_optimize, is_multi_objective=is_multi_objective
            )

            # Create optimization data
            optimization_data = OptimizationData(
                monthly=monthly_data, daily=daily_data, returns=rets_full, windows=windows
            )

            # Extract parameters from trial
            parameters = scenario_config.get("strategy_params", {}).copy()
            if hasattr(trial, "params") and trial.params is not None:
                parameters.update(trial.params)
            elif hasattr(trial, "user_attrs") and "parameters" in trial.user_attrs:
                parameters.update(trial.user_attrs["parameters"])

            # Evaluate using new architecture
            evaluation_result = evaluator.evaluate_parameters(
                parameters, scenario_config, optimization_data, strategy_backtester
            )

            objective_value_raw = evaluation_result.objective_value
            if isinstance(objective_value_raw, list):
                norm_obj: float | tuple[float, ...] = tuple(objective_value_raw)
            else:
                # At runtime this may be float or tuple; cast covers both
                norm_obj = cast(float | tuple[float, ...], objective_value_raw)
            full_pnl_returns = pd.Series(dtype=float)
            if trial and hasattr(trial, "user_attrs") and "full_pnl_returns" in trial.user_attrs:
                pnl_dict = trial.user_attrs["full_pnl_returns"]
                if isinstance(pnl_dict, dict):
                    full_pnl_returns = pd.Series(pnl_dict)
                    full_pnl_returns.index = pd.to_datetime(full_pnl_returns.index)
            return norm_obj, full_pnl_returns

        try:
            # Update cache if DataFrame index changed (different data slice)
            if self._daily_index_cache is None or not daily_data.index.equals(
                pd.Index(self._daily_index_cache)
            ):
                # Convert to NumPy datetime64 for fast, type-stable searchsorted
                self._daily_index_cache = daily_data.index.values.astype("datetime64[ns]")

                # Handle Multi-Index vs single-level columns only when cache invalidated
                if isinstance(daily_data.columns, pd.MultiIndex):
                    self._daily_prices_np_cache, _ = _df_to_float32_array(daily_data, field="Close")
                else:
                    self._daily_prices_np_cache, _ = _df_to_float32_array(daily_data)

            strategy_instance = StrategyFactory.create_strategy(
                str(scenario_config["strategy"]),
                scenario_config["strategy_params"],
                global_config=self.global_config,
            )

            # Build a full-length signal matrix aligned with the daily price DataFrame
            if isinstance(daily_data.columns, pd.MultiIndex):
                price_df_for_cols = daily_data.xs("Close", level="Field", axis=1)
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
            monthly_data_np, _ = _df_to_float32_array(monthly_data)
            daily_data_np, _ = _df_to_float32_array(daily_data)
            rets_full_np, _ = _df_to_float32_array(rets_full)

            obj_evaluated = self.evaluate_walk_forward_fast(
                trial,
                scenario_config,
                windows,
                monthly_data_np,
                daily_data_np,
                rets_full_np,
                metrics_to_optimize,
                is_multi_objective,
                signals.to_numpy(),
                strategy_instance,
            )

            full_pnl_returns = pd.Series(dtype=float)

            # obj_evaluated is annotated as float | tuple[float, ...] by evaluate_walk_forward_fast
            norm_obj2: float | tuple[float, ...] = obj_evaluated
            return norm_obj2, full_pnl_returns

        except Exception as exc:
            logger.error("Fast evaluation failed - falling back to new architecture: %s", exc)
            # Use new architecture components as fallback
            from ..backtesting.strategy_backtester import StrategyBacktester
            from ..optimization.results import OptimizationData
            from ..optimization.evaluator import BacktestEvaluator

            # Create new architecture components
            strategy_backtester = StrategyBacktester(self.global_config, self.data_source)
            evaluator = BacktestEvaluator(
                metrics_to_optimize=metrics_to_optimize, is_multi_objective=is_multi_objective
            )

            # Create optimization data
            optimization_data = OptimizationData(
                monthly=monthly_data, daily=daily_data, returns=rets_full, windows=windows
            )

            # Extract parameters from trial
            parameters = scenario_config.get("strategy_params", {}).copy()
            if hasattr(trial, "params") and trial.params is not None:
                parameters.update(trial.params)
            elif hasattr(trial, "user_attrs") and "parameters" in trial.user_attrs:
                parameters.update(trial.user_attrs["parameters"])

            # Evaluate using new architecture
            evaluation_result = evaluator.evaluate_parameters(
                parameters, scenario_config, optimization_data, strategy_backtester
            )

            objective_value_raw2 = evaluation_result.objective_value
            if isinstance(objective_value_raw2, list):
                norm_obj3: float | tuple[float, ...] = tuple(objective_value_raw2)
            else:
                norm_obj3 = cast(float | tuple[float, ...], objective_value_raw2)
            full_pnl_returns = pd.Series(dtype=float)
            if trial and hasattr(trial, "user_attrs") and "full_pnl_returns" in trial.user_attrs:
                pnl_dict = trial.user_attrs["full_pnl_returns"]
                if isinstance(pnl_dict, dict):
                    full_pnl_returns = pd.Series(pnl_dict)
                    full_pnl_returns.index = pd.to_datetime(full_pnl_returns.index)
            return norm_obj3, full_pnl_returns

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
        """
        Numba-optimized fast evaluation of parameters.

        Args:
            trial: Optimization trial object
            scenario_config: Scenario configuration
            windows: List of walk-forward windows
            monthly_data: Monthly price data
            daily_data: Daily OHLC data
            rets_full: Full period returns
            metrics_to_optimize: List of metrics to optimize
            is_multi_objective: Whether this is multi-objective optimization

        Returns:
            Tuple of (objective_value, full_pnl_returns)
        """

        # Gate numba import via shim for mypy
        class _RunBacktestNumba(Protocol):
            def __call__(
                self,
                prices: np.ndarray,
                signals: np.ndarray,
                start_idx: np.ndarray,
                end_idx: np.ndarray,
            ) -> np.ndarray: ...

        def _get_run_backtest_numba() -> _RunBacktestNumba:
            try:
                # Import and assert callable shape for type checker without returning Any
                from .. import numba_kernels as _nk2

                if not hasattr(_nk2, "run_backtest_numba"):
                    raise AttributeError("run_backtest_numba not available")
                func = self._module_accessor.get_module_attribute(_nk2, "run_backtest_numba")
                # Cast to the precise callable signature
                return cast(_RunBacktestNumba, func)
            except Exception as import_exc:
                exc_msg = str(import_exc)

                def _missing(*_: Any, **__: Any) -> np.ndarray:
                    raise RuntimeError(f"Numba kernel unavailable: {exc_msg}")

                return cast(_RunBacktestNumba, _missing)

        run_backtest_numba = _get_run_backtest_numba()
        from ..utils import _df_to_float32_array

        try:
            # Update cache if DataFrame index changed (different data slice)
            if self._daily_index_cache is None or not daily_data.index.equals(
                pd.Index(self._daily_index_cache)
            ):
                self._daily_index_cache = daily_data.index.to_numpy()

                # Handle Multi-Index vs single-level columns only when cache invalidated
                if isinstance(daily_data.columns, pd.MultiIndex):
                    self._daily_prices_np_cache, _ = _df_to_float32_array(daily_data, field="Close")
                else:
                    self._daily_prices_np_cache, _ = _df_to_float32_array(daily_data)

            prices_np = self._daily_prices_np_cache

            strategy_instance = StrategyFactory.create_strategy(
                str(scenario_config["strategy"]),
                scenario_config["strategy_params"],
                global_config=self.global_config,
            )

            # Build a full-length signals matrix
            logger.debug("Determining tickers for signal generation.")
            if isinstance(daily_data.columns, pd.MultiIndex):
                price_df_for_cols = daily_data.xs("Close", level="Field", axis=1)
            else:
                price_df_for_cols = daily_data

            tickers = list(price_df_for_cols.columns)
            logger.debug(f"Tickers for signal generation: {tickers}")

            # Initialise signals DataFrame with NaNs
            logger.debug("Initializing signals DataFrame.")
            signals_df = pd.DataFrame(
                data=np.nan,
                index=daily_data.index,
                columns=tickers,
                dtype=np.float32,
            )
            logger.debug("Successfully initialized signals DataFrame.")

            # Generate signals only on each test window start date
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
                        # Align columns
                        for col in window_signal.columns:
                            if col in signals_df.columns:
                                signals_df.at[te_start, col] = window_signal.iloc[0][col]
                        logger.debug(
                            f"Successfully assigned signal for window starting at {te_start}"
                        )
                except Exception as sig_exc:
                    logger.error(
                        "Signal generation failed for window start %s: %s",
                        te_start,
                        sig_exc,
                    )
            logger.debug("Finished signal generation loop.")

            # Forward-fill to make positions persistent
            signals_df.ffill(inplace=True)
            signals_df.fillna(0.0, inplace=True)

            # Convert to float32 NumPy array for Numba kernel
            signals_np, _ = _df_to_float32_array(signals_df)

            # Ensure cached index is numpy datetime64[ns]
            if self._daily_index_cache is None or not np.issubdtype(
                self._daily_index_cache.dtype, np.datetime64
            ):
                self._daily_index_cache = daily_data.index.values.astype("datetime64[ns]")
            start_indices = np.asarray(
                [np.searchsorted(self._daily_index_cache, np.datetime64(w[2])) for w in windows],
                dtype=np.int64,
            )
            end_indices = np.asarray(
                [np.searchsorted(self._daily_index_cache, np.datetime64(w[3])) for w in windows],
                dtype=np.int64,
            )

            if prices_np is None or signals_np is None:
                if logger.isEnabledFor(logging.ERROR):
                    logger.error("prices_np or signals_np is None. Cannot run numba backtest.")
                return np.nan, pd.Series(dtype=float)

            # Ensure correct dtype for Numba kernel
            prices_np = prices_np.astype(np.float32)
            signals_np = signals_np.astype(np.float32)

            portfolio_returns = run_backtest_numba(
                prices_np, signals_np, start_indices, end_indices
            )

            # Return the mean of the portfolio returns as the objective value
            objective_value: float | tuple[float, ...] = float(np.nanmean(portfolio_returns))
            full_pnl_returns = pd.Series(portfolio_returns, index=[w[2] for w in windows])
            return objective_value, full_pnl_returns

        except Exception as exc:
            logger.error("Numba evaluation failed: %s", exc)
            return np.nan, pd.Series(dtype=float)

    def evaluate_trial_parameters(
        self,
        scenario_config: Dict[str, Any],
        params: Dict[str, Any],
        monthly_data: pd.DataFrame,
        daily_data_ohlc: pd.DataFrame,
        rets_full: Union[pd.DataFrame, pd.Series],
        run_scenario_func,
    ) -> Dict[str, float]:
        """
        Evaluates a single set of parameters and returns performance metrics.

        Args:
            scenario_config: Base scenario configuration
            params: Parameters to evaluate
            monthly_data: Monthly price data
            daily_data_ohlc: Daily OHLC data
            rets_full: Full period returns data
            run_scenario_func: Function to run scenario evaluation

        Returns:
            Dictionary of metric names to values
        """
        temp_scenario_config = scenario_config.copy()
        temp_scenario_config["strategy_params"] = params

        returns = run_scenario_func(
            temp_scenario_config,
            monthly_data,
            daily_data_ohlc,
            rets_full if isinstance(rets_full, pd.DataFrame) else rets_full.to_frame(),
            verbose=False,
        )

        metrics_list = self._attribute_accessor.get_attribute(self, "metrics_to_optimize", None)
        if returns is None or returns.empty:
            if metrics_list is None:
                return {}
            return {metric: 0.0 for metric in metrics_list}

        from ..reporting.performance_metrics import calculate_metrics

        benchmark_series = daily_data_ohlc[self.global_config["benchmark"]]
        benchmark_rets = benchmark_series.pct_change(fill_method=None).fillna(0)
        metrics = calculate_metrics(returns, benchmark_rets, self.global_config["benchmark"])
        if metrics_list is None:
            # Ensure precise type dict[str, float]
            return {k: float(v) for k, v in metrics.items()}
        return {metric: float(metrics.get(metric, 0.0)) for metric in metrics_list}
