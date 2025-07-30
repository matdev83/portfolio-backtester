"""
BacktestEvaluator for walk-forward analysis.

This module implements the BacktestEvaluator class that performs walk-forward analysis
for parameter sets. It provides a consistent evaluation interface for all optimization
backends and supports both single and multi-objective optimization.
"""

import logging
import numpy as np
import pandas as pd
import time
from typing import Any, Dict, List, Union, Optional

from .results import EvaluationResult, OptimizationData
from ..backtesting.results import WindowResult
from ..backtesting.strategy_backtester import StrategyBacktester
from .performance_optimizer import (
    optimize_dataframe_memory, 
    cleanup_memory_if_needed,
    record_evaluation_performance,
    ParallelOptimizer
)

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
        
        # Optimize data memory usage if enabled
        if self.enable_memory_optimization:
            data = self._optimize_data_memory(data)
        
        # Create scenario config with the parameters to evaluate
        trial_scenario_config = scenario_config.copy()
        trial_scenario_config["strategy_params"] = parameters
        
        # Evaluate each window
        window_results = []
        objective_values = []
        window_lengths = []
        
        for window in data.windows:
            try:
                window_result = backtester.evaluate_window(
                    trial_scenario_config,
                    window,
                    data.monthly,
                    data.daily,
                    data.returns
                )
                
                window_results.append(window_result)
                
                # Extract objective value(s) from window metrics
                if self.is_multi_objective:
                    # For multi-objective, extract all requested metrics
                    obj_values = []
                    for metric_name in self.metrics_to_optimize:
                        metric_value = window_result.metrics.get(metric_name, 0.0)
                        # Handle NaN values
                        if pd.isna(metric_value):
                            metric_value = -1e9  # Large negative value for invalid results
                        obj_values.append(float(metric_value))
                    objective_values.append(obj_values)
                else:
                    # For single objective, use the first (and only) metric
                    metric_name = self.metrics_to_optimize[0]
                    metric_value = window_result.metrics.get(metric_name, 0.0)
                    # Handle NaN values
                    if pd.isna(metric_value):
                        metric_value = -1e9  # Large negative value for invalid results
                    objective_values.append(float(metric_value))
                
                # Track window length for potential length-weighted aggregation
                window_lengths.append(len(window_result.window_returns))
                
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
                
                # Add failed objective values
                if self.is_multi_objective:
                    objective_values.append([-1e9] * len(self.metrics_to_optimize))
                else:
                    objective_values.append(-1e9)
                
                window_lengths.append(0)
        
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