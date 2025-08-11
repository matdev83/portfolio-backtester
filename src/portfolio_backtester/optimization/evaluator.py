"""
BacktestEvaluator for walk-forward analysis.

This module implements the BacktestEvaluator class that performs walk-forward analysis
for parameter sets. It provides a consistent evaluation interface for all optimization
backends and supports both single and multi-objective optimization.
"""

import logging
import numpy as np
import pandas as pd
import os
from typing import Any, Dict, List, Union, Optional
try:
    # Prefer optional import; used only when parallel path is enabled
    from joblib import Parallel, delayed  # type: ignore
except Exception:  # pragma: no cover - joblib may not be available in some envs
    Parallel = None  # type: ignore[assignment]
    delayed = None  # type: ignore[assignment]

from .results import EvaluationResult, OptimizationData
from ..backtesting.results import WindowResult
from ..backtesting.strategy_backtester import StrategyBacktester
from ..parallel_wfo import ParallelWFOProcessor, create_parallel_wfo_processor
from ..optimization.wfo_window import WFOWindow

# WFO enhancement imports
from ..backtesting.window_evaluator import WindowEvaluator

# Performance optimizer imports - hard dependency for Alpha
from .performance_optimizer import (
    optimize_dataframe_memory,
)

# Enhanced WFO always enabled – no dual path
WFO_ENHANCEMENT_AVAILABLE: bool = True

# New performance optimization imports (module doesn't exist yet)
PERFORMANCE_ABSTRACTION_AVAILABLE = False
AbstractPerformanceOptimizer = type(None)
AbstractTradeTracker = type(None)

# Legacy ParallelOptimizer removed - now using integrated ParallelWFOProcessor


logger = logging.getLogger(__name__)


def _lookup_metric(metrics: Dict[str, float], name: str, default: float = -1e9) -> float:
    """Case- and alias-insensitive metric lookup helper."""
    key = metrics.get(name)
    if key is not None and not pd.isna(key):
        return float(key)
    lower = name.lower()
    for alias in (lower, lower.replace(" ", "_"), f"{lower}_ratio"):
        val = metrics.get(alias)
        if val is not None and not pd.isna(val):
            return float(val)
    return default


class BacktestEvaluator:
    """Performs walk-forward analysis for parameter sets.

    This class evaluates parameter sets across all time windows using walk-forward
    analysis and aggregates the results into final evaluation metrics. It works
    identically for all optimization backends and supports both single and
    multi-objective optimization.

    Attributes:
        metrics_to_optimize: List of metric names to optimize
        is_multi_objective: Whether this is multi-objective optimization
        aggregate_length_weighted: Whether to weight results by window length
    """

    def __init__(
        self,
        metrics_to_optimize: List[str],
        is_multi_objective: bool,
        aggregate_length_weighted: bool = False,
        strategy_backtester: Optional[StrategyBacktester] = None,
        n_jobs: int = 1,
        enable_memory_optimization: bool = True,
        enable_parallel_optimization: bool = True,
    ):
        """Initialize the BacktestEvaluator.

        Args:
            metrics_to_optimize: List of metric names to optimize (e.g., ['sharpe_ratio'])
            is_multi_objective: Whether this is multi-objective optimization
            aggregate_length_weighted: Whether to weight aggregation by window length
            strategy_backtester: Pure backtesting engine (optional for backward compatibility)
            n_jobs: Number of parallel jobs for evaluation
            enable_memory_optimization: Whether to enable memory optimizations
            enable_parallel_optimization: Whether to enable parallel processing optimizations
        """
        # Core attributes required by existing code
        self.metrics_to_optimize = metrics_to_optimize
        self.is_multi_objective = is_multi_objective
        self.aggregate_length_weighted = aggregate_length_weighted

        # New architecture attributes
        self.strategy_backtester = strategy_backtester
        self.n_jobs = n_jobs if n_jobs and n_jobs > 0 else (os.cpu_count() or 1)
        self.enable_memory_optimization = enable_memory_optimization
        self.enable_parallel_optimization = enable_parallel_optimization

        # WFO enhancement attributes
        self.evaluation_frequency: Optional[str] = None  # Will be determined per scenario
        self.window_evaluator: Optional[Any] = None  # Will be initialized if needed

        # Legacy parallel_optimizer removed - now using integrated ParallelWFOProcessor

        # Initialize WFO parallel processor for window-level parallelization using DIP
        if self.enable_parallel_optimization:
            # Use fewer workers for window-level parallelization to avoid resource conflicts
            # when already in a trial-level multiprocessing context
            wfo_workers = min(self.n_jobs, 4)  # Cap at 4 workers for window parallelization
            wfo_config = {
                "parallel_wfo_config": {
                    "enable_parallel": True,
                    "max_workers": wfo_workers,
                    "process_timeout": 300,
                    "min_windows_for_parallel": 2,
                }
            }
            self.wfo_parallel_processor: Optional[ParallelWFOProcessor] = (
                create_parallel_wfo_processor(wfo_config)
            )
        else:
            self.wfo_parallel_processor = None

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"BacktestEvaluator initialized: metrics={self.metrics_to_optimize}, "
                f"multi_objective={self.is_multi_objective}, length_weighted={self.aggregate_length_weighted}"
            )

    def _get_universe_tickers(self, strategy: Any) -> List[str]:
        """Get universe tickers from strategy or use default.

        Args:
            strategy: Strategy instance

        Returns:
            List of ticker symbols
        """
        # Try to get universe from strategy
        if hasattr(strategy, "get_universe"):
            try:
                universe = strategy.get_universe()
                return list(universe) if universe else []
            except Exception:
                pass

        # Default fallback - this will be overridden by actual implementation
        return ["SPY", "TLT", "GLD", "VTI", "QQQ"]

    def evaluate_parameters(
        self,
        parameters: Dict[str, Any],
        scenario_config: Dict[str, Any],
        data: OptimizationData,
        backtester: "StrategyBacktester",
    ) -> EvaluationResult:
        """
        Evaluate a given set of parameters by running a backtest.
        This is the single, unified evaluation path.
        """
        # If not a WFO, create a single window for the entire period
        if not scenario_config.get("is_wfo", True):
            single_window = WFOWindow(
                train_start=pd.to_datetime(scenario_config["start_date"]),
                train_end=pd.to_datetime(scenario_config["end_date"]),
                test_start=pd.to_datetime(scenario_config["start_date"]),
                test_end=pd.to_datetime(scenario_config["end_date"]),
            )
            data.windows = [single_window]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Evaluating parameters: {parameters}")

        # Initialize window evaluator if needed, now with backtester instance
        if self.window_evaluator is None:
            self.window_evaluator = WindowEvaluator(backtester=backtester)

        enhanced_windows = data.windows
        window_results: List[WindowResult] = []

        # Derive universe tickers from daily data if available
        if data.daily is not None and not data.daily.empty:
            if isinstance(data.daily.columns, pd.MultiIndex):
                derived_universe = data.daily.columns.get_level_values(0).unique().tolist()
            else:
                derived_universe = list(data.daily.columns)
        else:
            derived_universe = []

        def _eval_single_window(win: WFOWindow) -> WindowResult:
            """Worker-safe single window evaluation."""
            try:
                strategy = backtester._get_strategy(
                    scenario_config["strategy"], parameters, scenario_config
                )
                universe_tickers_local = derived_universe or self._get_universe_tickers(strategy)
                benchmark_ticker_local = (
                    universe_tickers_local[0] if universe_tickers_local else "SPY"
                )
                return self.window_evaluator.evaluate_window(
                    window=win,
                    strategy=strategy,
                    daily_data=data.daily,
                    full_monthly_data=data.monthly,
                    full_rets_daily=data.returns,
                    benchmark_data=data.daily,
                    universe_tickers=universe_tickers_local,
                    benchmark_ticker=benchmark_ticker_local,
                )
            except Exception:
                return WindowResult(
                    window_returns=pd.Series(dtype=float),
                    metrics={metric: -1e9 for metric in self.metrics_to_optimize},
                    train_start=win.train_start,
                    train_end=win.train_end,
                    test_start=win.test_start,
                    test_end=win.test_end,
                    trades=[],
                    final_weights={},
                )

        # Parallelize across windows when enabled and beneficial
        can_parallelize = (
            self.enable_parallel_optimization
            and Parallel is not None
            and self.n_jobs > 1
            and len(enhanced_windows) >= 2
        )

        if can_parallelize:
            workers = min(self.n_jobs, 4)  # cap to avoid oversubscription
            window_results = Parallel(n_jobs=workers, prefer="processes")(
                delayed(_eval_single_window)(win) for win in enhanced_windows
            )
        else:
            for idx, enhanced_window in enumerate(enhanced_windows, start=1):
                logger.info(
                    "[Trial Progress] Window %d/%d %s–%s evaluation started",
                    idx,
                    len(enhanced_windows),
                    enhanced_window.test_start.date(),
                    enhanced_window.test_end.date(),
                )
                result = _eval_single_window(enhanced_window)
                logger.info(
                    "[Trial Progress] Window %d/%d finished: Sharpe=%s",
                    idx,
                    len(enhanced_windows),
                    _lookup_metric(result.metrics, "Sharpe", float("nan")),
                )
                window_results.append(result)

        # Aggregate results
        return self._aggregate_window_results(window_results, parameters)

    def _aggregate_window_results(
        self, window_results: List[WindowResult], parameters: Dict[str, Any]
    ) -> EvaluationResult:
        """Aggregate results from daily evaluation windows.

        Args:
            window_results: List of WindowResult objects
            parameters: Parameter dictionary for this evaluation

        Returns:
            EvaluationResult: Aggregated results
        """
        # Combine daily returns from all windows
        all_returns = []
        all_trades = []

        for result in window_results:
            if len(result.window_returns) > 0:
                all_returns.append(result.window_returns)
            if hasattr(result, "trades") and result.trades:
                all_trades.extend(result.trades)

        if all_returns:
            combined_returns = pd.concat(all_returns)
        else:
            combined_returns = pd.Series(dtype=float)

        # Calculate performance metrics
        metrics = self._calculate_metrics(combined_returns, all_trades)

        # Extract objective values for optimization
        if self.is_multi_objective:
            objective_values = []
            for metric_name in self.metrics_to_optimize:
                metric_value = _lookup_metric(metrics, metric_name, -1e9)
                if pd.isna(metric_value):
                    metric_value = -1e9
                objective_values.append(float(metric_value))
            aggregated_objective: Union[float, List[float]] = objective_values
        else:
            metric_name = self.metrics_to_optimize[0]
            metric_value = _lookup_metric(metrics, metric_name, -1e9)
            if pd.isna(metric_value):
                metric_value = -1e9
            aggregated_objective = float(metric_value)

        return EvaluationResult(
            objective_value=aggregated_objective, metrics=metrics, window_results=window_results
        )

    def _calculate_metrics(self, returns: pd.Series, trades: List) -> Dict[str, float]:
        """Calculate performance metrics from returns and trades.

        Args:
            returns: Series of portfolio returns
            trades: List of Trade objects

        Returns:
            Dictionary of calculated metrics
        """
        if len(returns) == 0:
            return {"total_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0}

        # Ensure we have numeric data only
        returns_numeric = pd.to_numeric(returns, errors="coerce").fillna(0)
        # Filter out any remaining non-numeric values
        returns_numeric = returns_numeric[pd.notna(returns_numeric)]

        if len(returns_numeric) == 0:
            return {"total_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0}

        # Basic return metrics
        prod_result = (1 + returns_numeric).prod()
        if pd.isna(prod_result) or not isinstance(prod_result, (int, float, np.number)):
            total_return = 0.0
        else:
            try:
                total_return = float(prod_result) - 1.0
            except (TypeError, ValueError):
                total_return = 0.0
        annual_return = float((1 + returns_numeric.mean()) ** 252 - 1)
        volatility = float(returns_numeric.std() * np.sqrt(252))
        sharpe_ratio = float(annual_return / volatility if volatility > 0 else 0.0)

        # Sortino ratio
        downside = returns_numeric[returns_numeric < 0]
        downside_vol = downside.std() * np.sqrt(252)
        sortino_ratio = annual_return / downside_vol if downside_vol > 0 else 0.0

        # Drawdown calculation
        cumulative: pd.Series = (1 + returns_numeric).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Trade-based metrics
        trade_metrics = {}
        if trades:
            durations = [trade.duration_days for trade in trades if hasattr(trade, "duration_days")]
            if durations:
                trade_metrics.update(
                    {
                        "num_trades": len(trades),
                        "avg_trade_duration": np.mean(durations),
                        "max_trade_duration": max(durations),
                        "min_trade_duration": min(durations),
                    }
                )

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            **trade_metrics,
        }

    def _aggregate_objective_values(
        self, objective_values: List[Union[float, List[float]]], window_lengths: List[int]
    ) -> Union[float, List[float]]:
        """Aggregate objective values across windows.

        Args:
            objective_values: List of objective values from each window
            window_lengths: List of window lengths for potential weighting

        Returns:
            Aggregated objective value (float for single-objective, list for multi-objective)
        """
        if not objective_values:
            if self.is_multi_objective:
                return [-1e9] * len(self.metrics_to_optimize)
            else:
                return -1e9

        if self.is_multi_objective:
            # Convert to numpy array for easier manipulation
            obj_array = np.array(objective_values, dtype=float)

            if self.aggregate_length_weighted and len(window_lengths) == len(objective_values):
                # Length-weighted average
                weights = np.array(window_lengths, dtype=float)
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                    aggregated = np.average(obj_array, axis=0, weights=weights)
                else:
                    aggregated = np.mean(obj_array, axis=0)
            else:
                # Simple average
                aggregated = np.mean(obj_array, axis=0)

            result_list = aggregated.tolist() if hasattr(aggregated, "tolist") else list(aggregated)
            return result_list
        else:
            # Single objective
            numeric_values = np.array(objective_values, dtype=float)

            if self.aggregate_length_weighted and len(window_lengths) == len(objective_values):
                # Length-weighted average
                weights = np.array(window_lengths, dtype=float)
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                    aggregated = np.average(numeric_values, weights=weights)
                else:
                    aggregated = np.mean(numeric_values)
            else:
                # Simple average
                aggregated = np.mean(numeric_values)

            return float(aggregated)

    def _aggregate_metrics(
        self, window_results: List[WindowResult], window_lengths: List[int]
    ) -> Dict[str, float]:
        """Aggregate metrics across windows.

        Args:
            window_results: List of WindowResult objects
            window_lengths: List of window lengths for potential weighting

        Returns:
            Dictionary of aggregated metrics
        """
        if not window_results:
            return {}

        # Collect all metric names
        all_metric_names: set[str] = set()
        for window_result in window_results:
            if isinstance(window_result, WindowResult):
                all_metric_names.update(window_result.metrics.keys())

        aggregated_metrics = {}

        for metric_name in all_metric_names:
            # Extract metric values from all windows
            metric_values = []
            for window_result in window_results:
                if isinstance(window_result, WindowResult):
                    value = _lookup_metric(window_result.metrics, metric_name, np.nan)
                    # Replace NaN with a large negative value for aggregation
                    if pd.isna(value) or value is None:
                        value = -1e9
                    metric_values.append(float(value))

            # Aggregate the metric
            if self.aggregate_length_weighted and len(window_lengths) == len(metric_values):
                # Length-weighted average
                weights = np.array(window_lengths, dtype=float)
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                    aggregated_value = np.average(metric_values, weights=weights)
                else:
                    aggregated_value = np.mean(metric_values)
            else:
                # Simple average
                aggregated_value = np.mean(metric_values)

            aggregated_metrics[metric_name] = float(aggregated_value)

        return aggregated_metrics

    def _optimize_data_memory(self, data: OptimizationData) -> OptimizationData:
        """Optimize memory usage of data containers."""
        try:
            # Optimize DataFrame memory usage
            if hasattr(data, "monthly") and data.monthly is not None:
                data.monthly = optimize_dataframe_memory(data.monthly)

            if hasattr(data, "daily") and data.daily is not None:
                data.daily = optimize_dataframe_memory(data.daily)

            if hasattr(data, "returns") and data.returns is not None:
                data.returns = optimize_dataframe_memory(data.returns)

            return data
        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")
            return data
