"""
Unit tests for BacktestEvaluator.

Tests the BacktestEvaluator class that performs walk-forward analysis
for parameter sets in both single and multi-objective optimization scenarios.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, List

from src.portfolio_backtester.optimization.evaluator import BacktestEvaluator
from src.portfolio_backtester.optimization.results import (
    EvaluationResult, OptimizationData
)
from src.portfolio_backtester.backtesting.results import (
    WindowResult,
)

class TestBacktestEvaluator:
    """Test cases for BacktestEvaluator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample data
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        self.sample_monthly = pd.DataFrame({
            'AAPL': np.random.randn(len(dates)),
            'MSFT': np.random.randn(len(dates))
        }, index=dates)
        
        self.sample_daily = pd.DataFrame({
            'AAPL': np.random.randn(len(dates)) * 100 + 100,
            'MSFT': np.random.randn(len(dates)) * 100 + 200
        }, index=dates)
        
        self.sample_returns = pd.DataFrame({
            'AAPL': np.random.randn(len(dates)) * 0.02,
            'MSFT': np.random.randn(len(dates)) * 0.02
        }, index=dates)
        
        # Create sample windows
        self.sample_windows = [
            (pd.Timestamp('2020-01-01'), pd.Timestamp('2020-03-31'), 
             pd.Timestamp('2020-04-01'), pd.Timestamp('2020-06-30')),
            (pd.Timestamp('2020-04-01'), pd.Timestamp('2020-06-30'), 
             pd.Timestamp('2020-07-01'), pd.Timestamp('2020-09-30')),
            (pd.Timestamp('2020-07-01'), pd.Timestamp('2020-09-30'), 
             pd.Timestamp('2020-10-01'), pd.Timestamp('2020-12-31'))
        ]
        
        self.optimization_data = OptimizationData(
            monthly=self.sample_monthly,
            daily=self.sample_daily,
            returns=self.sample_returns,
            windows=self.sample_windows
        )
    
    def test_init_single_objective(self):
        """Test BacktestEvaluator initialization for single objective."""
        evaluator = BacktestEvaluator(
            metrics_to_optimize=['sharpe_ratio'],
            is_multi_objective=False
        )
        
        assert evaluator.metrics_to_optimize == ['sharpe_ratio']
        assert evaluator.is_multi_objective is False
        assert evaluator.aggregate_length_weighted is False
    
    def test_init_multi_objective(self):
        """Test BacktestEvaluator initialization for multi-objective."""
        evaluator = BacktestEvaluator(
            metrics_to_optimize=['sharpe_ratio', 'max_drawdown'],
            is_multi_objective=True,
            aggregate_length_weighted=True
        )
        
        assert evaluator.metrics_to_optimize == ['sharpe_ratio', 'max_drawdown']
        assert evaluator.is_multi_objective is True
        assert evaluator.aggregate_length_weighted is True
    
    def test_evaluate_parameters_single_objective_success(self):
        """Test successful parameter evaluation for single objective."""
        evaluator = BacktestEvaluator(
            metrics_to_optimize=['sharpe_ratio'],
            is_multi_objective=False
        )
        
        # Mock backtester
        mock_backtester = Mock()
        
        # Create mock window results
        mock_window_results = []
        for i, window in enumerate(self.sample_windows):
            window_returns = pd.Series(
                np.random.randn(90) * 0.01, 
                index=pd.date_range(window[2], window[3], freq='D')[:90]
            )
            window_result = WindowResult(
                window_returns=window_returns,
                metrics={'sharpe_ratio': 1.5 + i * 0.1},
                train_start=window[0],
                train_end=window[1],
                test_start=window[2],
                test_end=window[3]
            )
            mock_window_results.append(window_result)
        
        mock_backtester.evaluate_window.side_effect = mock_window_results
        
        # Test parameters
        parameters = {'param1': 0.5, 'param2': 10}
        scenario_config = {
            'strategy': 'test_strategy',
            'strategy_params': {'base_param': 1.0}
        }
        
        # Evaluate parameters
        result = evaluator.evaluate_parameters(
            parameters, scenario_config, self.optimization_data, mock_backtester
        )
        
        # Verify result
        assert isinstance(result, EvaluationResult)
        assert isinstance(result.objective_value, float)
        assert result.objective_value == pytest.approx(1.6, rel=1e-2)  # Average of 1.5, 1.6, 1.7
        assert len(result.window_results) == 3
        assert 'sharpe_ratio' in result.metrics
        
        # Verify backtester was called correctly
        assert mock_backtester.evaluate_window.call_count == 3
        
        # Check that strategy_params were updated correctly
        for call in mock_backtester.evaluate_window.call_args_list:
            called_config = call[0][0]  # First argument (scenario_config)
            expected_params = scenario_config['strategy_params'].copy()
            # Parameters should be prefixed with strategy name
            strategy_name = scenario_config['strategy']
            for param_name, param_value in parameters.items():
                prefixed_name = f"{strategy_name}.{param_name}"
                expected_params[prefixed_name] = param_value
            assert called_config['strategy_params'] == expected_params
    
    def test_evaluate_parameters_multi_objective_success(self):
        """Test successful parameter evaluation for multi-objective."""
        evaluator = BacktestEvaluator(
            metrics_to_optimize=['sharpe_ratio', 'max_drawdown'],
            is_multi_objective=True
        )
        
        # Mock backtester
        mock_backtester = Mock()
        
        # Create mock window results
        mock_window_results = []
        for i, window in enumerate(self.sample_windows):
            window_returns = pd.Series(
                np.random.randn(90) * 0.01, 
                index=pd.date_range(window[2], window[3], freq='D')[:90]
            )
            window_result = WindowResult(
                window_returns=window_returns,
                metrics={
                    'sharpe_ratio': 1.5 + i * 0.1,
                    'max_drawdown': -0.1 - i * 0.01
                },
                train_start=window[0],
                train_end=window[1],
                test_start=window[2],
                test_end=window[3]
            )
            mock_window_results.append(window_result)
        
        mock_backtester.evaluate_window.side_effect = mock_window_results
        
        # Test parameters
        parameters = {'param1': 0.5, 'param2': 10}
        scenario_config = {
            'strategy': 'test_strategy',
            'strategy_params': {'base_param': 1.0}
        }
        
        # Evaluate parameters
        result = evaluator.evaluate_parameters(
            parameters, scenario_config, self.optimization_data, mock_backtester
        )
        
        # Verify result
        assert isinstance(result, EvaluationResult)
        assert isinstance(result.objective_value, list)
        assert len(result.objective_value) == 2
        assert result.objective_value[0] == pytest.approx(1.6, rel=1e-2)  # Average sharpe
        assert result.objective_value[1] == pytest.approx(-0.11, rel=1e-2)  # Average drawdown
        assert len(result.window_results) == 3
        assert 'sharpe_ratio' in result.metrics
        assert 'max_drawdown' in result.metrics
    
    def test_evaluate_parameters_with_failed_window(self):
        """Test parameter evaluation when one window fails."""
        evaluator = BacktestEvaluator(
            metrics_to_optimize=['sharpe_ratio'],
            is_multi_objective=False
        )
        
        # Mock backtester
        mock_backtester = Mock()
        
        # Create mock window results with one failure
        mock_window_results = []
        for i, window in enumerate(self.sample_windows):
            if i == 1:  # Second window fails
                mock_backtester.evaluate_window.side_effect = [
                    mock_window_results[0] if mock_window_results else None,
                    Exception("Window evaluation failed"),
                    None  # Will be set below
                ]
                continue
            
            window_returns = pd.Series(
                np.random.randn(90) * 0.01, 
                index=pd.date_range(window[2], window[3], freq='D')[:90]
            )
            window_result = WindowResult(
                window_returns=window_returns,
                metrics={'sharpe_ratio': 1.5 + i * 0.1},
                train_start=window[0],
                train_end=window[1],
                test_start=window[2],
                test_end=window[3]
            )
            mock_window_results.append(window_result)
        
        # Set up side effects properly
        def side_effect(*args, **kwargs):
            call_count = side_effect.call_count
            side_effect.call_count += 1
            
            if call_count == 0:
                return mock_window_results[0]
            elif call_count == 1:
                raise Exception("Window evaluation failed")
            else:
                return mock_window_results[1]
        
        side_effect.call_count = 0
        mock_backtester.evaluate_window.side_effect = side_effect
        
        # Test parameters
        parameters = {'param1': 0.5, 'param2': 10}
        scenario_config = {
            'strategy': 'test_strategy',
            'strategy_params': {'base_param': 1.0}
        }
        
        # Evaluate parameters
        result = evaluator.evaluate_parameters(
            parameters, scenario_config, self.optimization_data, mock_backtester
        )
        
        # Verify result - should handle the failed window gracefully
        assert isinstance(result, EvaluationResult)
        assert isinstance(result.objective_value, float)
        assert len(result.window_results) == 3
        
        # The failed window should have empty metrics
        failed_window = result.window_results[1]
        assert failed_window.metrics['sharpe_ratio'] == -1e9
    
    def test_evaluate_parameters_length_weighted_aggregation(self):
        """Test length-weighted aggregation of results."""
        evaluator = BacktestEvaluator(
            metrics_to_optimize=['sharpe_ratio'],
            is_multi_objective=False,
            aggregate_length_weighted=True
        )
        
        # Mock backtester
        mock_backtester = Mock()
        
        # Create mock window results with different lengths
        mock_window_results = []
        window_lengths = [60, 90, 92]  # Use actual window lengths that match date ranges
        
        for i, (window, length) in enumerate(zip(self.sample_windows, window_lengths)):
            # Create date range that matches the actual window period
            actual_dates = pd.date_range(window[2], window[3], freq='D')
            # Truncate to desired length
            window_returns = pd.Series(
                np.random.randn(min(length, len(actual_dates))) * 0.01, 
                index=actual_dates[:min(length, len(actual_dates))]
            )
            window_result = WindowResult(
                window_returns=window_returns,
                metrics={'sharpe_ratio': 1.0 + i * 0.5},  # 1.0, 1.5, 2.0
                train_start=window[0],
                train_end=window[1],
                test_start=window[2],
                test_end=window[3]
            )
            mock_window_results.append(window_result)
        
        mock_backtester.evaluate_window.side_effect = mock_window_results
        
        # Test parameters
        parameters = {'param1': 0.5}
        scenario_config = {'strategy': 'test_strategy', 'strategy_params': {}}
        
        # Evaluate parameters
        result = evaluator.evaluate_parameters(
            parameters, scenario_config, self.optimization_data, mock_backtester
        )
        
        # Calculate expected weighted average
        # Values: [1.0, 1.5, 2.0], Weights: [60, 90, 92] -> [60/242, 90/242, 92/242]
        expected_weighted_avg = (1.0 * 60 + 1.5 * 90 + 2.0 * 92) / (60 + 90 + 92)
        
        assert result.objective_value == pytest.approx(expected_weighted_avg, rel=1e-3)
    
    def test_aggregate_objective_values_single_objective(self):
        """Test aggregation of objective values for single objective."""
        evaluator = BacktestEvaluator(['sharpe_ratio'], False)
        
        objective_values = [1.0, 1.5, 2.0]
        window_lengths = [60, 90, 120]
        
        # Test simple average
        result = evaluator._aggregate_objective_values(objective_values, window_lengths)
        assert result == pytest.approx(1.5, rel=1e-3)
        
        # Test length-weighted average
        evaluator.aggregate_length_weighted = True
        result = evaluator._aggregate_objective_values(objective_values, window_lengths)
        expected = (1.0 * 60 + 1.5 * 90 + 2.0 * 120) / (60 + 90 + 120)
        assert result == pytest.approx(expected, rel=1e-3)
    
    def test_aggregate_objective_values_multi_objective(self):
        """Test aggregation of objective values for multi-objective."""
        evaluator = BacktestEvaluator(['sharpe_ratio', 'max_drawdown'], True)
        
        objective_values = [[1.0, -0.1], [1.5, -0.15], [2.0, -0.2]]
        window_lengths = [60, 90, 120]
        
        # Test simple average
        result = evaluator._aggregate_objective_values(objective_values, window_lengths)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == pytest.approx(1.5, rel=1e-3)
        assert result[1] == pytest.approx(-0.15, rel=1e-3)
    
    def test_aggregate_metrics(self):
        """Test aggregation of metrics across windows."""
        evaluator = BacktestEvaluator(['sharpe_ratio'], False)
        
        # Create window results
        window_results = []
        for i in range(3):
            window_result = WindowResult(
                window_returns=pd.Series([0.01] * (60 + i * 30)),
                metrics={
                    'sharpe_ratio': 1.0 + i * 0.5,
                    'total_return': 0.1 + i * 0.05
                },
                train_start=pd.Timestamp('2020-01-01'),
                train_end=pd.Timestamp('2020-03-31'),
                test_start=pd.Timestamp('2020-04-01'),
                test_end=pd.Timestamp('2020-06-30')
            )
            window_results.append(window_result)
        
        window_lengths = [60, 90, 120]
        
        # Test aggregation
        result = evaluator._aggregate_metrics(window_results, window_lengths)
        
        assert 'sharpe_ratio' in result
        assert 'total_return' in result
        assert result['sharpe_ratio'] == pytest.approx(1.5, rel=1e-3)
        assert result['total_return'] == pytest.approx(0.15, rel=1e-3)  # Average of 0.1, 0.15, 0.2
    
    def test_empty_objective_values(self):
        """Test handling of empty objective values."""
        evaluator = BacktestEvaluator(['sharpe_ratio'], False)
        
        result = evaluator._aggregate_objective_values([], [])
        assert result == -1e9
        
        # Multi-objective case
        evaluator.is_multi_objective = True
        evaluator.metrics_to_optimize = ['sharpe_ratio', 'max_drawdown']
        result = evaluator._aggregate_objective_values([], [])
        assert result == [-1e9, -1e9]
    
    def test_nan_handling(self):
        """Test handling of NaN values in metrics."""
        evaluator = BacktestEvaluator(['sharpe_ratio'], False)
        
        # Mock backtester that returns NaN metrics
        mock_backtester = Mock()
        window_result = WindowResult(
            window_returns=pd.Series([0.01, 0.02]),
            metrics={'sharpe_ratio': np.nan},
            train_start=pd.Timestamp('2020-01-01'),
            train_end=pd.Timestamp('2020-03-31'),
            test_start=pd.Timestamp('2020-04-01'),
            test_end=pd.Timestamp('2020-06-30')
        )
        mock_backtester.evaluate_window.return_value = window_result
        
        # Create minimal optimization data with one window
        minimal_data = OptimizationData(
            monthly=self.sample_monthly.iloc[:100],
            daily=self.sample_daily.iloc[:100],
            returns=self.sample_returns.iloc[:100],
            windows=[self.sample_windows[0]]
        )
        
        result = evaluator.evaluate_parameters(
            {'param1': 1.0}, 
            {'strategy': 'test', 'strategy_params': {}}, 
            minimal_data, 
            mock_backtester
        )
        
        # NaN should be converted to -1e9
        assert result.objective_value == -1e9