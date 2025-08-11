"""
BacktestEvaluator for walk-forward analysis.

This module implements the BacktestEvaluator class that performs walk-forward analysis
for parameter sets. It provides a consistent evaluation interface for all optimization
backends and supports both single and multi-objective optimization.
"""

import logging
import copy
import numpy as np
import pandas as pd
import os
from typing import Any, Dict, List, Union, Optional

from .results import EvaluationResult, OptimizationData
from ..backtesting.results import WindowResult
from ..backtesting.strategy_backtester import StrategyBacktester
from ..parallel_wfo import ParallelWFOProcessor, create_parallel_wfo_processor

# WFO enhancement imports
from ..backtesting.window_evaluator import WindowEvaluator
from .wfo_window import WFOWindow

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

    def _determine_evaluation_frequency(self, scenario_config: Dict[str, Any]) -> str:
        """Determine required evaluation frequency based on strategy configuration.

        Args:
            scenario_config: Scenario configuration dictionary

        Returns:
            Evaluation frequency ('D', 'W', or 'M')
        """
        strategy_class = scenario_config.get("strategy_class", "")
        strategy_name = scenario_config.get("strategy", "")
        timing_config = scenario_config.get("timing_config", {})

        # Intramonth or seasonal signal strategies need daily evaluation
        lowered_class = strategy_class.lower()
        lowered_name = strategy_name.lower()
        if (
            "intramonth" in lowered_class
            or "intramonth" in lowered_name
            or "seasonalsignal" in lowered_class
            or "seasonalsignal" in lowered_name
        ):
            return "D"

        # Signal-based timing with daily scanning
        if timing_config.get("mode") == "signal_based":
            scan_freq = timing_config.get("scan_frequency", "D")
            if scan_freq.upper() == "D":
                return "D"

        # Check rebalance frequency
        rebalance_freq = scenario_config.get("rebalance_frequency", "M")
        if rebalance_freq == "D":
            return "D"

        # Default to monthly for backward compatibility
        return "M"

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
        Evaluate a given set of parameters.
        """
        # Data is now prepared in the OptimizationOrchestrator, so we can use it directly.

        # Determine evaluation frequency for this scenario
        self.evaluation_frequency = self._determine_evaluation_frequency(scenario_config)

        # Use enhanced daily evaluation if needed
        if self.evaluation_frequency == "D" and hasattr(data, "daily") and data.daily is not None:
            return self._evaluate_parameters_daily(parameters, scenario_config, data, backtester)
        else:
            # Use existing monthly evaluation for backward compatibility
            return self._evaluate_parameters_monthly(parameters, scenario_config, data, backtester)

    def _evaluate_parameters_monthly(
        self,
        parameters: Dict[str, Any],
        scenario_config: Dict[str, Any],
        data: OptimizationData,
        backtester: StrategyBacktester,
    ) -> EvaluationResult:
        """Evaluate parameters with existing monthly evaluation (backward compatibility).

        Args:
            parameters: Dictionary of parameter values to evaluate
            scenario_config: Base scenario configuration
            data: OptimizationData containing price data and windows
            backtester: StrategyBacktester instance for evaluation

        Returns:
            EvaluationResult: Aggregated evaluation results across all windows
        """
        # Optimize data memory usage if enabled
        if self.enable_memory_optimization:
            data = self._optimize_data_memory(data)

        # Create scenario config with the parameters to evaluate
        trial_scenario_config = copy.deepcopy(scenario_config)
        if "strategy_params" not in trial_scenario_config:
            trial_scenario_config["strategy_params"] = {}

        # Add strategy prefix to parameters if needed
        strategy_name = trial_scenario_config.get("strategy", "")
        prefixed_parameters = {}
        for param_name, param_value in parameters.items():
            # Check if parameter already has a prefix
            if "." in param_name:
                prefixed_parameters[param_name] = param_value
            else:
                # Add strategy prefix
                prefixed_param_name = f"{strategy_name}.{param_name}"
                prefixed_parameters[prefixed_param_name] = param_value

        trial_scenario_config["strategy_params"].update(prefixed_parameters)

        # Evaluate each window
        # Use integrated parallel WFO processing (handles both parallel and sequential internally)
        window_results = self._evaluate_windows_parallel(
            data.windows, trial_scenario_config, data, backtester
        )

        # Extract objective values and window lengths
        objective_values: List[Union[float, List[float]]] = []
        window_lengths = []
        for window_result in window_results:
            # Extract objective value(s) from window metrics
            if self.is_multi_objective:
                obj_values = []
                for metric_name in self.metrics_to_optimize:
                    metric_value = _lookup_metric(window_result.metrics, metric_name)
                    if pd.isna(metric_value):
                        metric_value = -1e9
                    obj_values.append(float(metric_value))
                objective_values.append(obj_values)
            else:
                metric_name = self.metrics_to_optimize[0]
                metric_value = _lookup_metric(window_result.metrics, metric_name)
                if pd.isna(metric_value):
                    metric_value = -1e9
                objective_values.append(float(metric_value))
            window_lengths.append(len(window_result.window_returns))

        # Aggregate results across windows
        aggregated_objective = self._aggregate_objective_values(objective_values, window_lengths)

        # Aggregate metrics across windows
        aggregated_metrics = self._aggregate_metrics(window_results, window_lengths)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Aggregated objective: {aggregated_objective}")

        return EvaluationResult(
            objective_value=aggregated_objective,
            metrics=aggregated_metrics,
            window_results=window_results,
        )

    def _evaluate_parameters_daily(
        self,
        parameters: Dict[str, Any],
        scenario_config: Dict[str, Any],
        data: OptimizationData,
        backtester: StrategyBacktester,
    ) -> EvaluationResult:
        """Evaluate parameters with daily strategy evaluation.

        Args:
            parameters: Dictionary of parameter values to evaluate
            scenario_config: Base scenario configuration
            data: OptimizationData containing price data and windows
            backtester: StrategyBacktester instance for evaluation

        Returns:
            EvaluationResult: Aggregated evaluation results across all windows
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Using daily evaluation for parameters: {parameters}")

        # Initialize window evaluator if needed, now with backtester instance
        if self.window_evaluator is None:
            self.window_evaluator = WindowEvaluator(backtester=backtester)

        # Convert existing windows to enhanced WFOWindow objects
        enhanced_windows = []
        for window in data.windows:
            enhanced_windows.append(
                WFOWindow(
                    train_start=window[0],
                    train_end=window[1],
                    test_start=window[2],
                    test_end=window[3],
                    evaluation_frequency=self.evaluation_frequency or "M",
                    strategy_name=scenario_config.get("name", "unknown"),
                )
            )

        # Evaluate each window with daily evaluation
        window_results = []
        # Derive universe tickers from daily data if available, else fallback to strategy/meta detection
        if data.daily is not None and not data.daily.empty:
            if isinstance(data.daily.columns, pd.MultiIndex):
                derived_universe = data.daily.columns.get_level_values(0).unique().tolist()
            else:
                derived_universe = list(data.daily.columns)
        else:
            derived_universe = []

        for idx, enhanced_window in enumerate(enhanced_windows, start=1):
            logger.info(
                "[Trial Progress] Window %d/%d %s–%s evaluation started",
                idx,
                len(enhanced_windows),
                enhanced_window.test_start.date(),
                enhanced_window.test_end.date(),
            )
            try:
                # Create strategy instance with parameters
                strategy = backtester._get_strategy(
                    scenario_config["strategy"], parameters, scenario_config
                )

                # Get universe tickers (prefer derived list)
                universe_tickers = derived_universe or self._get_universe_tickers(strategy)
                benchmark_ticker = universe_tickers[0] if universe_tickers else "SPY"

                # Evaluate window with daily evaluation
                result = self.window_evaluator.evaluate_window(
                    window=enhanced_window,
                    strategy=strategy,
                    daily_data=data.daily,
                    full_monthly_data=data.monthly,
                    full_rets_daily=data.returns,
                    benchmark_data=data.daily,  # Assuming benchmark is in daily data
                    universe_tickers=universe_tickers,
                    benchmark_ticker=benchmark_ticker,
                )

                logger.info(
                    "[Trial Progress] Window %d/%d finished: Sharpe=%s",
                    idx,
                    len(enhanced_windows),
                    _lookup_metric(result.metrics, "Sharpe", float("nan")),
                )
                window_results.append(result)

            except Exception as e:
                logger.error(
                    f"Error evaluating window {enhanced_window.test_start.date()} to {enhanced_window.test_end.date()}: {e}"
                )
                # Create empty result for failed window
                empty_result = WindowResult(
                    window_returns=pd.Series(dtype=float),
                    metrics={metric: -1e9 for metric in self.metrics_to_optimize},
                    train_start=enhanced_window.train_start,
                    train_end=enhanced_window.train_end,
                    test_start=enhanced_window.test_start,
                    test_end=enhanced_window.test_end,
                    trades=[],
                    final_weights={},
                )
                window_results.append(empty_result)

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
            all_metric_names.update(window_result.metrics.keys())

        aggregated_metrics = {}

        for metric_name in all_metric_names:
            # Extract metric values from all windows
            metric_values = []
            for window_result in window_results:
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

    def _evaluate_windows_parallel(
        self,
        windows: List[Any],
        scenario_config: Dict[str, Any],
        data: OptimizationData,
        backtester: StrategyBacktester,
    ) -> List[WindowResult]:
        """Evaluate windows in parallel using the refactored ParallelWFOProcessor."""

        def evaluate_window_func(window_config, config, shared_data):
            """Worker function for parallel window evaluation."""
            try:
                # Extract the actual window from the config dictionary
                window = window_config["window"]
                window_result = backtester.evaluate_window(
                    config,
                    window,
                    shared_data["monthly"],
                    shared_data["daily"],
                    shared_data["returns"],
                )
                if window_result is None:
                    # Create empty window result for failed evaluation
                    if len(window) >= 4:
                        train_start, train_end, test_start, test_end = window[:4]
                    else:
                        train_start = train_end = test_start = test_end = pd.Timestamp.now()
                    window_result = WindowResult(
                        window_returns=pd.Series(dtype=float),
                        metrics={metric: -1e9 for metric in self.metrics_to_optimize},
                        train_start=train_start,
                        train_end=train_end,
                        test_start=test_start,
                        test_end=test_end,
                    )
                return window_result, pd.Series(dtype=float)  # Return expected tuple format
            except Exception as e:
                logger.warning(f"Failed to evaluate window {window}: {e}")
                # Create empty window result for failed evaluation
                if len(window) >= 4:
                    train_start, train_end, test_start, test_end = window[:4]
                else:
                    train_start = train_end = test_start = test_end = pd.Timestamp.now()
                empty_window = WindowResult(
                    window_returns=pd.Series(dtype=float),
                    metrics={metric: -1e9 for metric in self.metrics_to_optimize},
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )
                return empty_window, pd.Series(dtype=float)

        # Prepare shared data for parallel processing
        shared_data = {"monthly": data.monthly, "daily": data.daily, "returns": data.returns}

        # Convert windows to the format expected by ParallelWFOProcessor
        window_configs = [{"window": window} for window in windows]

        # Use the refactored ParallelWFOProcessor for window-level parallelization
        if self.wfo_parallel_processor is not None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Evaluating {len(windows)} windows in parallel using ParallelWFOProcessor"
                )

            parallel_results = self.wfo_parallel_processor.process_windows_parallel(
                windows=window_configs,
                evaluate_window_func=evaluate_window_func,
                scenario_config=scenario_config,
                shared_data=shared_data,
            )

            # Extract just the WindowResult objects (first element of each tuple)
            window_results = [result[0] for result in parallel_results]
        else:
            # Fallback to sequential processing
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Evaluating {len(windows)} windows sequentially (parallel processing disabled)"
                )

            window_results = []
            for idx, window_config in enumerate(window_configs, start=1):
                win = window_config["window"]
                logger.info(
                    "[Trial Progress] Window %d/%d %s–%s evaluation started",
                    idx,
                    len(window_configs),
                    win[2].date() if len(win) >= 3 else "?",
                    win[3].date() if len(win) >= 4 else "?",
                )
                result, _ = evaluate_window_func(window_config, scenario_config, shared_data)
                logger.info(
                    "[Trial Progress] Window %d/%d finished: Sharpe=%s",
                    idx,
                    len(window_configs),
                    _lookup_metric(result.metrics, "Sharpe", float("nan")),
                )
                window_results.append(result)

        return window_results
