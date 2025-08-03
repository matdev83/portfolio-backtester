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
import time
from typing import Any, Dict, List, Union, Optional

from .results import EvaluationResult, OptimizationData
from ..backtesting.results import WindowResult
from ..backtesting.strategy_backtester import StrategyBacktester

# New WFO enhancement imports
try:
    from ..backtesting.window_evaluator import WindowEvaluator
    from .wfo_window import WFOWindow
    WFO_ENHANCEMENT_AVAILABLE = True
except ImportError:
    WFO_ENHANCEMENT_AVAILABLE = False
    WindowEvaluator = None
    WFOWindow = None

# New performance optimization imports
try:
    from .performance import AbstractPerformanceOptimizer, AbstractTradeTracker
    PERFORMANCE_ABSTRACTION_AVAILABLE = True
except ImportError:
    PERFORMANCE_ABSTRACTION_AVAILABLE = False
    AbstractPerformanceOptimizer = None
    AbstractTradeTracker = None

# Legacy performance optimizer imports (for backward compatibility)
try:
    from .performance_optimizer import (
        optimize_dataframe_memory, 
        cleanup_memory_if_needed,
        record_evaluation_performance,
        ParallelOptimizer
    )
    PERFORMANCE_OPTIMIZER_AVAILABLE = True
except ImportError:
    PERFORMANCE_OPTIMIZER_AVAILABLE = False
    # Provide fallback implementations
    def optimize_dataframe_memory(df):
        return df
    
    def cleanup_memory_if_needed():
        pass
    
    def record_evaluation_performance(time_seconds):
        pass
    
    class ParallelOptimizer:
        def __init__(self, n_jobs=1):
            self.n_jobs = n_jobs
        
        def parallel_map(self, func, items, **kwargs):
            return [func(item, **kwargs) for item in items]

logger = logging.getLogger(__name__)


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
        enable_parallel_optimization: bool = True
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
        self.n_jobs = n_jobs
        self.enable_memory_optimization = enable_memory_optimization
        self.enable_parallel_optimization = enable_parallel_optimization
        
        # WFO enhancement attributes
        self.evaluation_frequency = None  # Will be determined per scenario
        self.window_evaluator = None  # Will be initialized if needed
        
        # Initialize performance optimizers
        if enable_parallel_optimization and n_jobs > 1:
            self.parallel_optimizer = ParallelOptimizer(n_jobs=n_jobs)
        else:
            self.parallel_optimizer = None
        
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
        strategy_class = scenario_config.get('strategy_class', '')
        strategy_name = scenario_config.get('strategy', '')
        timing_config = scenario_config.get('timing_config', {})
        
        # Intramonth strategies need daily evaluation
        if 'intramonth' in strategy_class.lower() or 'intramonth' in strategy_name.lower():
            return 'D'
        
        # Signal-based timing with daily scanning
        if timing_config.get('mode') == 'signal_based':
            scan_freq = timing_config.get('scan_frequency', 'D')
            if scan_freq == 'D':
                return 'D'
        
        # Check rebalance frequency
        rebalance_freq = scenario_config.get('rebalance_frequency', 'M')
        if rebalance_freq == 'D':
            return 'D'
        
        # Default to monthly for backward compatibility
        return 'M'
    
    def _get_universe_tickers(self, strategy) -> List[str]:
        """Get universe tickers from strategy or use default.
        
        Args:
            strategy: Strategy instance
            
        Returns:
            List of ticker symbols
        """
        # Try to get universe from strategy
        if hasattr(strategy, 'get_universe'):
            try:
                return strategy.get_universe()
            except:
                pass
        
        # Default fallback - this will be overridden by actual implementation
        return ['SPY', 'TLT', 'GLD', 'VTI', 'QQQ']
    
    def evaluate_parameters(
        self,
        parameters: Dict[str, Any],
        scenario_config: Dict[str, Any], 
        data: OptimizationData,
        backtester: StrategyBacktester
    ) -> EvaluationResult:
        """Evaluate parameters across all time windows using walk-forward analysis.
        
        This method evaluates a parameter set across all walk-forward windows,
        aggregates the results, and returns a structured EvaluationResult.
        
        Args:
            parameters: Dictionary of parameter values to evaluate
            scenario_config: Base scenario configuration
            data: OptimizationData containing price data and windows
            backtester: StrategyBacktester instance for evaluation
            
        Returns:
            EvaluationResult: Aggregated evaluation results across all windows
        """
        start_time = time.time()
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Evaluating parameters: {parameters}")
        
        # Determine evaluation frequency for this scenario
        self.evaluation_frequency = self._determine_evaluation_frequency(scenario_config)
        
        # Use enhanced daily evaluation if needed and available
        if (self.evaluation_frequency == 'D' and 
            WFO_ENHANCEMENT_AVAILABLE and 
            hasattr(data, 'daily') and 
            data.daily is not None):
            return self._evaluate_parameters_daily(parameters, scenario_config, data, backtester)
        else:
            # Use existing monthly evaluation for backward compatibility
            return self._evaluate_parameters_monthly(parameters, scenario_config, data, backtester)
    
    def _evaluate_parameters_monthly(
        self,
        parameters: Dict[str, Any],
        scenario_config: Dict[str, Any], 
        data: OptimizationData,
        backtester: StrategyBacktester
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
        if self.enable_parallel_optimization and self.n_jobs > 1 and self.parallel_optimizer is not None:
            window_results = self._evaluate_windows_parallel(
                data.windows,
                trial_scenario_config,
                data,
                backtester
            )
        else:
            window_results = []
            for window in data.windows:
                try:
                    window_result = backtester.evaluate_window(
                        trial_scenario_config,
                        window,
                        data.monthly,
                        data.daily,
                        data.returns
                    )
                    if window_result is None:
                        train_start, train_end, test_start, test_end = window
                        window_result = WindowResult(
                            window_returns=pd.Series(dtype=float),
                            metrics={metric: -1e9 for metric in self.metrics_to_optimize},
                            train_start=train_start,
                            train_end=train_end,
                            test_start=test_start,
                            test_end=test_end,
                        )
                    window_results.append(window_result)
                except Exception as e:
                    logger.warning(f"Failed to evaluate window {window}: {e}")
                    # Create empty window result for failed evaluation
                    train_start, train_end, test_start, test_end = window
                    empty_window = WindowResult(
                        window_returns=pd.Series(dtype=float),
                        metrics={metric: -1e9 for metric in self.metrics_to_optimize},
                        train_start=train_start,
                        train_end=train_end,
                        test_start=test_start,
                        test_end=test_end
                    )
                    window_results.append(empty_window)
        
        # Extract objective values and window lengths
        objective_values = []
        window_lengths = []
        for window_result in window_results:
            # Extract objective value(s) from window metrics
            if self.is_multi_objective:
                obj_values = []
                for metric_name in self.metrics_to_optimize:
                    metric_value = window_result.metrics.get(metric_name, 0.0)
                    if pd.isna(metric_value):
                        metric_value = -1e9
                    obj_values.append(float(metric_value))
                objective_values.append(obj_values)
            else:
                metric_name = self.metrics_to_optimize[0]
                metric_value = window_result.metrics.get(metric_name, 0.0)
                if pd.isna(metric_value):
                    metric_value = -1e9
                objective_values.append(float(metric_value))
            window_lengths.append(len(window_result.window_returns))
        
        # Aggregate results across windows
        aggregated_objective = self._aggregate_objective_values(
            objective_values, window_lengths
        )
        
        # Aggregate metrics across windows
        aggregated_metrics = self._aggregate_metrics(window_results, window_lengths)
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Aggregated objective: {aggregated_objective}")
        
        return EvaluationResult(
            objective_value=aggregated_objective,
            metrics=aggregated_metrics,
            window_results=window_results
        )
    
    def _evaluate_parameters_daily(
        self,
        parameters: Dict[str, Any],
        scenario_config: Dict[str, Any], 
        data: OptimizationData,
        backtester: StrategyBacktester
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
        
        # Initialize window evaluator if needed
        if self.window_evaluator is None:
            self.window_evaluator = WindowEvaluator()
        
        # Convert existing windows to enhanced WFOWindow objects
        enhanced_windows = []
        for window in data.windows:
            enhanced_windows.append(WFOWindow(
                train_start=window[0],
                train_end=window[1],
                test_start=window[2],
                test_end=window[3],
                evaluation_frequency=self.evaluation_frequency,
                strategy_name=scenario_config.get('name', 'unknown')
            ))
        
        # Evaluate each window with daily evaluation
        window_results = []
        for window in enhanced_windows:
            try:
                # Create strategy instance with parameters
                strategy = backtester._get_strategy(scenario_config, parameters)
                
                # Get universe tickers
                universe_tickers = self._get_universe_tickers(strategy)
                benchmark_ticker = 'SPY'  # Default benchmark
                
                # Evaluate window with daily evaluation
                result = self.window_evaluator.evaluate_window(
                    window=window,
                    strategy=strategy,
                    daily_data=data.daily,
                    benchmark_data=data.daily,  # Assuming benchmark is in daily data
                    universe_tickers=universe_tickers,
                    benchmark_ticker=benchmark_ticker
                )
                
                window_results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating window {window.test_start.date()} to {window.test_end.date()}: {e}")
                # Create empty result for failed window
                empty_result = WindowResult(
                    window_returns=pd.Series(dtype=float),
                    metrics={metric: -1e9 for metric in self.metrics_to_optimize},
                    train_start=window.train_start,
                    train_end=window.train_end,
                    test_start=window.test_start,
                    test_end=window.test_end,
                    trades=[],
                    final_weights={}
                )
                window_results.append(empty_result)
        
        # Aggregate results
        return self._aggregate_window_results(window_results, parameters)
    
    def _aggregate_window_results(self, window_results: List[WindowResult], parameters: Dict[str, Any]) -> EvaluationResult:
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
            if hasattr(result, 'trades') and result.trades:
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
                metric_value = metrics.get(metric_name, -1e9)
                if pd.isna(metric_value):
                    metric_value = -1e9
                objective_values.append(float(metric_value))
            aggregated_objective = objective_values
        else:
            metric_name = self.metrics_to_optimize[0]
            metric_value = metrics.get(metric_name, -1e9)
            if pd.isna(metric_value):
                metric_value = -1e9
            aggregated_objective = float(metric_value)
        
        return EvaluationResult(
            objective_value=aggregated_objective,
            metrics=metrics,
            window_results=window_results
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
            return {'total_return': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0}
        
        # Basic return metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + returns.mean()) ** 252 - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0.0
        
        # Drawdown calculation
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade-based metrics
        trade_metrics = {}
        if trades:
            durations = [trade.duration_days for trade in trades if hasattr(trade, 'duration_days')]
            if durations:
                trade_metrics.update({
                    'num_trades': len(trades),
                    'avg_trade_duration': np.mean(durations),
                    'max_trade_duration': max(durations),
                    'min_trade_duration': min(durations)
                })
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            **trade_metrics
        }
    
    def _aggregate_objective_values(
        self, 
        objective_values: List[Union[float, List[float]]], 
        window_lengths: List[int]
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
            
            return aggregated.tolist()
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
        self, 
        window_results: List[WindowResult], 
        window_lengths: List[int]
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
        all_metric_names = set()
        for window_result in window_results:
            all_metric_names.update(window_result.metrics.keys())
        
        aggregated_metrics = {}
        
        for metric_name in all_metric_names:
            # Extract metric values from all windows
            metric_values = []
            for window_result in window_results:
                value = window_result.metrics.get(metric_name, np.nan)
                # Replace NaN with a large negative value for aggregation
                if pd.isna(value):
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
            if hasattr(data, 'monthly') and data.monthly is not None:
                data.monthly = optimize_dataframe_memory(data.monthly)
            
            if hasattr(data, 'daily') and data.daily is not None:
                data.daily = optimize_dataframe_memory(data.daily)
            
            if hasattr(data, 'returns') and data.returns is not None:
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
        backtester: StrategyBacktester
    ) -> List[Any]:
        """Evaluate windows in parallel."""
        def evaluate_single_window(window):
            try:
                return backtester.evaluate_window(
                    scenario_config,
                    window,
                    data.monthly,
                    data.daily,
                    data.returns
                )
            except Exception as e:
                logger.error(f"Window evaluation failed: {e}")
                return None
        
        # Use parallel optimizer for window evaluation
        return self.parallel_optimizer.parallel_map(evaluate_single_window, windows)