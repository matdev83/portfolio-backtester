"""
Integration tests for OptimizationOrchestrator.

Tests the OptimizationOrchestrator class with mock parameter generators
to verify proper coordination between parameter generators and evaluators.
"""

import pytest
import pandas as pd
import numpy as np
import time
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, List

from src.portfolio_backtester.optimization.orchestrator import (
    OptimizationOrchestrator, ProgressTracker, ParameterGenerator
)
from src.portfolio_backtester.optimization.evaluator import BacktestEvaluator
from src.portfolio_backtester.optimization.results import (
    OptimizationResult, EvaluationResult, OptimizationData
)
from src.portfolio_backtester.backtesting.results import (
    WindowResult,
)

class MockParameterGenerator:
    """Mock parameter generator for testing."""
    
    def __init__(self, parameter_sets: List[Dict[str, Any]], best_result: OptimizationResult):
        self.parameter_sets = parameter_sets
        self.best_result = best_result
        self.current_index = 0
        self.results = []
        self.initialized = False
    
    def initialize(self, scenario_config: Dict[str, Any], optimization_config: Dict[str, Any]) -> None:
        self.initialized = True
    
    def suggest_parameters(self) -> Dict[str, Any]:
        if self.current_index >= len(self.parameter_sets):
            raise IndexError("No more parameters to suggest")
        
        params = self.parameter_sets[self.current_index]
        self.current_index += 1
        return params
    
    def report_result(self, parameters: Dict[str, Any], result: Any) -> None:
        self.results.append((parameters, result))
    
    def is_finished(self) -> bool:
        return self.current_index >= len(self.parameter_sets)
    
    def get_best_result(self) -> OptimizationResult:
        return self.best_result


class TestProgressTracker:
    """Test cases for ProgressTracker class."""
    
    def test_init_single_objective(self):
        """Test ProgressTracker initialization for single objective."""
        tracker = ProgressTracker(
            timeout_seconds=300,
            early_stop_patience=10,
            is_multi_objective=False
        )
        
        assert tracker.timeout_seconds == 300
        assert tracker.early_stop_patience == 10
        assert tracker.is_multi_objective is False
        assert tracker.total_evaluations == 0
        assert tracker.best_value is None
        assert tracker.evaluations_without_improvement == 0
    
    def test_init_multi_objective(self):
        """Test ProgressTracker initialization for multi-objective."""
        tracker = ProgressTracker(
            timeout_seconds=None,
            early_stop_patience=None,
            is_multi_objective=True
        )
        
        assert tracker.timeout_seconds is None
        assert tracker.early_stop_patience is None
        assert tracker.is_multi_objective is True
    
    def test_update_progress_single_objective_improvement(self):
        """Test progress update with improvement in single objective."""
        tracker = ProgressTracker(is_multi_objective=False)
        
        # First evaluation
        tracker.update_progress(1.5)
        assert tracker.total_evaluations == 1
        assert tracker.best_value == 1.5
        assert tracker.evaluations_without_improvement == 0
        
        # Improvement
        tracker.update_progress(2.0)
        assert tracker.total_evaluations == 2
        assert tracker.best_value == 2.0
        assert tracker.evaluations_without_improvement == 0
        
        # No improvement
        tracker.update_progress(1.8)
        assert tracker.total_evaluations == 3
        assert tracker.best_value == 2.0
        assert tracker.evaluations_without_improvement == 1
    
    def test_update_progress_multi_objective_improvement(self):
        """Test progress update with improvement in multi-objective."""
        tracker = ProgressTracker(is_multi_objective=True)
        
        # First evaluation
        tracker.update_progress([1.5, -0.1])
        assert tracker.total_evaluations == 1
        assert tracker.best_value == [1.5, -0.1]
        assert tracker.evaluations_without_improvement == 0
        
        # Improvement in first objective
        tracker.update_progress([2.0, -0.15])
        assert tracker.total_evaluations == 2
        assert tracker.best_value == [2.0, -0.15]
        assert tracker.evaluations_without_improvement == 0
        
        # No improvement
        tracker.update_progress([1.8, -0.2])
        assert tracker.total_evaluations == 3
        assert tracker.best_value == [2.0, -0.15]
        assert tracker.evaluations_without_improvement == 1
    
    def test_should_stop_timeout(self):
        """Test timeout-based stopping."""
        tracker = ProgressTracker(timeout_seconds=0.1)  # Very short timeout
        
        assert not tracker.should_stop()
        
        # Wait for timeout
        time.sleep(0.15)
        assert tracker.should_stop()
    
    def test_should_stop_early_stopping(self):
        """Test early stopping based on patience."""
        tracker = ProgressTracker(early_stop_patience=2, is_multi_objective=False)
        
        # Initial evaluation
        tracker.update_progress(1.0)
        assert not tracker.should_stop()
        
        # No improvement
        tracker.update_progress(0.9)
        assert not tracker.should_stop()
        
        # Still no improvement - should trigger early stopping
        tracker.update_progress(0.8)
        assert tracker.should_stop()
    
    def test_get_status(self):
        """Test getting optimization status."""
        tracker = ProgressTracker(
            timeout_seconds=300,
            early_stop_patience=10,
            is_multi_objective=False
        )
        
        tracker.update_progress(1.5)
        status = tracker.get_status()
        
        assert status['total_evaluations'] == 1
        assert status['best_value'] == 1.5
        assert status['evaluations_without_improvement'] == 0
        assert status['timeout_seconds'] == 300
        assert status['early_stop_patience'] == 10
        assert 'elapsed_seconds' in status


class TestOptimizationOrchestrator:
    """Test cases for OptimizationOrchestrator class."""
    
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
             pd.Timestamp('2020-07-01'), pd.Timestamp('2020-09-30'))
        ]
        
        self.optimization_data = OptimizationData(
            monthly=self.sample_monthly,
            daily=self.sample_daily,
            returns=self.sample_returns,
            windows=self.sample_windows
        )
    
    def test_init(self):
        """Test OptimizationOrchestrator initialization."""
        # Create mock components
        mock_generator = Mock(spec=ParameterGenerator)
        mock_evaluator = Mock(spec=BacktestEvaluator)
        mock_evaluator.is_multi_objective = False
        
        orchestrator = OptimizationOrchestrator(
            parameter_generator=mock_generator,
            evaluator=mock_evaluator,
            timeout_seconds=300,
            early_stop_patience=10
        )
        
        assert orchestrator.parameter_generator is mock_generator
        assert orchestrator.evaluator is mock_evaluator
        assert orchestrator.progress_tracker.timeout_seconds == 300
        assert orchestrator.progress_tracker.early_stop_patience == 10
    
    def test_optimize_single_objective_success(self):
        """Test successful single-objective optimization."""
        # Create mock evaluator
        evaluator = BacktestEvaluator(
            metrics_to_optimize=['sharpe_ratio'],
            is_multi_objective=False
        )
        
        # Create parameter sets to test
        parameter_sets = [
            {'param1': 0.1, 'param2': 5},
            {'param1': 0.2, 'param2': 10},
            {'param1': 0.3, 'param2': 15}
        ]
        
        # Create expected best result
        best_result = OptimizationResult(
            best_parameters={'param1': 0.2, 'param2': 10},
            best_value=2.5,
            n_evaluations=3,
            optimization_history=[]
        )
        
        # Create mock parameter generator
        mock_generator = MockParameterGenerator(parameter_sets, best_result)
        
        # Create orchestrator
        orchestrator = OptimizationOrchestrator(
            parameter_generator=mock_generator,
            evaluator=evaluator
        )
        
        # Mock backtester
        mock_backtester = Mock()
        
        # Create mock evaluation results
        evaluation_results = []
        for i, params in enumerate(parameter_sets):
            window_results = []
            for window in self.sample_windows:
                window_result = WindowResult(
                    window_returns=pd.Series([0.01] * 90),
                    metrics={'sharpe_ratio': 1.0 + i * 0.5},
                    train_start=window[0],
                    train_end=window[1],
                    test_start=window[2],
                    test_end=window[3]
                )
                window_results.append(window_result)
            
            eval_result = EvaluationResult(
                objective_value=1.0 + i * 0.5,
                metrics={'sharpe_ratio': 1.0 + i * 0.5},
                window_results=window_results
            )
            evaluation_results.append(eval_result)
        
        # Mock the evaluator's evaluate_parameters method
        evaluator.evaluate_parameters = Mock(side_effect=evaluation_results)
        
        # Run optimization
        scenario_config = {'strategy': 'test_strategy', 'strategy_params': {}}
        optimization_config = {'max_trials': 3}
        
        result = orchestrator.optimize(
            scenario_config, optimization_config, self.optimization_data, mock_backtester
        )
        
        # Verify results
        assert isinstance(result, OptimizationResult)
        assert result.best_parameters == {'param1': 0.2, 'param2': 10}
        assert result.best_value == 2.5
        assert result.n_evaluations == 3
        
        # Verify generator was initialized
        assert mock_generator.initialized
        
        # Verify all parameter sets were evaluated
        assert len(mock_generator.results) == 3
        
        # Verify evaluator was called correctly
        assert evaluator.evaluate_parameters.call_count == 3
    
    def test_optimize_multi_objective_success(self):
        """Test successful multi-objective optimization."""
        # Create mock evaluator
        evaluator = BacktestEvaluator(
            metrics_to_optimize=['sharpe_ratio', 'max_drawdown'],
            is_multi_objective=True
        )
        
        # Create parameter sets to test
        parameter_sets = [
            {'param1': 0.1, 'param2': 5},
            {'param1': 0.2, 'param2': 10}
        ]
        
        # Create expected best result
        best_result = OptimizationResult(
            best_parameters={'param1': 0.2, 'param2': 10},
            best_value=[2.0, -0.1],
            n_evaluations=2,
            optimization_history=[]
        )
        
        # Create mock parameter generator
        mock_generator = MockParameterGenerator(parameter_sets, best_result)
        
        # Create orchestrator
        orchestrator = OptimizationOrchestrator(
            parameter_generator=mock_generator,
            evaluator=evaluator
        )
        
        # Mock backtester
        mock_backtester = Mock()
        
        # Create mock evaluation results
        evaluation_results = []
        for i, params in enumerate(parameter_sets):
            window_results = []
            for window in self.sample_windows:
                window_result = WindowResult(
                    window_returns=pd.Series([0.01] * 90),
                    metrics={
                        'sharpe_ratio': 1.5 + i * 0.5,
                        'max_drawdown': -0.1 - i * 0.05
                    },
                    train_start=window[0],
                    train_end=window[1],
                    test_start=window[2],
                    test_end=window[3]
                )
                window_results.append(window_result)
            
            eval_result = EvaluationResult(
                objective_value=[1.5 + i * 0.5, -0.1 - i * 0.05],
                metrics={
                    'sharpe_ratio': 1.5 + i * 0.5,
                    'max_drawdown': -0.1 - i * 0.05
                },
                window_results=window_results
            )
            evaluation_results.append(eval_result)
        
        # Mock the evaluator's evaluate_parameters method
        evaluator.evaluate_parameters = Mock(side_effect=evaluation_results)
        
        # Run optimization
        scenario_config = {'strategy': 'test_strategy', 'strategy_params': {}}
        optimization_config = {'max_trials': 2}
        
        result = orchestrator.optimize(
            scenario_config, optimization_config, self.optimization_data, mock_backtester
        )
        
        # Verify results
        assert isinstance(result, OptimizationResult)
        assert result.best_parameters == {'param1': 0.2, 'param2': 10}
        assert result.best_value == [2.0, -0.1]
        assert result.n_evaluations == 2
    
    def test_optimize_with_timeout(self):
        """Test optimization with timeout."""
        # Create mock evaluator
        evaluator = BacktestEvaluator(
            metrics_to_optimize=['sharpe_ratio'],
            is_multi_objective=False
        )
        
        # Create many parameter sets (more than we can evaluate in timeout)
        parameter_sets = [{'param1': i * 0.1} for i in range(100)]
        
        best_result = OptimizationResult(
            best_parameters={'param1': 0.1},
            best_value=1.0,
            n_evaluations=1,
            optimization_history=[]
        )
        
        mock_generator = MockParameterGenerator(parameter_sets, best_result)
        
        # Create orchestrator with very short timeout
        orchestrator = OptimizationOrchestrator(
            parameter_generator=mock_generator,
            evaluator=evaluator,
            timeout_seconds=0.1  # Very short timeout
        )
        
        # Mock backtester and evaluator
        mock_backtester = Mock()
        
        def slow_evaluation(*args, **kwargs):
            time.sleep(0.05)  # Slow evaluation
            return EvaluationResult(
                objective_value=1.0,
                metrics={'sharpe_ratio': 1.0},
                window_results=[]
            )
        
        evaluator.evaluate_parameters = Mock(side_effect=slow_evaluation)
        
        # Run optimization
        scenario_config = {'strategy': 'test_strategy', 'strategy_params': {}}
        optimization_config = {}
        
        start_time = time.time()
        result = orchestrator.optimize(
            scenario_config, optimization_config, self.optimization_data, mock_backtester
        )
        elapsed = time.time() - start_time
        
        # Should have stopped due to timeout
        assert elapsed < 1.0  # Should be much less than if all 100 evaluations ran
        assert isinstance(result, OptimizationResult)
    
    def test_optimize_with_early_stopping(self):
        """Test optimization with early stopping."""
        # Create mock evaluator
        evaluator = BacktestEvaluator(
            metrics_to_optimize=['sharpe_ratio'],
            is_multi_objective=False
        )
        
        # Create parameter sets with decreasing performance
        parameter_sets = [
            {'param1': 0.1},  # Best
            {'param1': 0.2},  # Worse
            {'param1': 0.3},  # Even worse
            {'param1': 0.4},  # Should trigger early stopping
        ]
        
        best_result = OptimizationResult(
            best_parameters={'param1': 0.1},
            best_value=2.0,
            n_evaluations=3,
            optimization_history=[]
        )
        
        mock_generator = MockParameterGenerator(parameter_sets, best_result)
        
        # Create orchestrator with early stopping
        orchestrator = OptimizationOrchestrator(
            parameter_generator=mock_generator,
            evaluator=evaluator,
            early_stop_patience=2  # Stop after 2 evaluations without improvement
        )
        
        # Mock backtester
        mock_backtester = Mock()
        
        # Create evaluation results with decreasing performance
        evaluation_results = [
            EvaluationResult(objective_value=2.0, metrics={'sharpe_ratio': 2.0}, window_results=[]),
            EvaluationResult(objective_value=1.5, metrics={'sharpe_ratio': 1.5}, window_results=[]),
            EvaluationResult(objective_value=1.0, metrics={'sharpe_ratio': 1.0}, window_results=[]),
            EvaluationResult(objective_value=0.5, metrics={'sharpe_ratio': 0.5}, window_results=[])
        ]
        
        evaluator.evaluate_parameters = Mock(side_effect=evaluation_results)
        
        # Run optimization
        scenario_config = {'strategy': 'test_strategy', 'strategy_params': {}}
        optimization_config = {}
        
        result = orchestrator.optimize(
            scenario_config, optimization_config, self.optimization_data, mock_backtester
        )
        
        # Should have stopped early (after 3 evaluations: 1 best + 2 without improvement)
        assert evaluator.evaluate_parameters.call_count == 3
        assert len(mock_generator.results) == 3
        assert isinstance(result, OptimizationResult)
    
    def test_optimize_with_evaluation_error(self):
        """Test optimization handling evaluation errors gracefully."""
        # Create mock evaluator
        evaluator = BacktestEvaluator(
            metrics_to_optimize=['sharpe_ratio'],
            is_multi_objective=False
        )
        
        parameter_sets = [{'param1': 0.1}]
        best_result = OptimizationResult(
            best_parameters={},
            best_value=-1e9,
            n_evaluations=0,
            optimization_history=[]
        )
        
        mock_generator = MockParameterGenerator(parameter_sets, best_result)
        
        orchestrator = OptimizationOrchestrator(
            parameter_generator=mock_generator,
            evaluator=evaluator
        )
        
        # Mock backtester
        mock_backtester = Mock()
        
        # Mock evaluator to raise an exception
        evaluator.evaluate_parameters = Mock(side_effect=Exception("Evaluation failed"))
        
        # Run optimization
        scenario_config = {'strategy': 'test_strategy', 'strategy_params': {}}
        optimization_config = {}
        
        result = orchestrator.optimize(
            scenario_config, optimization_config, self.optimization_data, mock_backtester
        )
        
        # Should return the best result from generator (empty in this case)
        assert isinstance(result, OptimizationResult)
        assert result.best_parameters == {}
        assert result.best_value == -1e9
    
    def test_get_progress_status(self):
        """Test getting progress status during optimization."""
        evaluator = BacktestEvaluator(['sharpe_ratio'], False)
        mock_generator = Mock(spec=ParameterGenerator)
        mock_generator.is_finished.return_value = True
        
        orchestrator = OptimizationOrchestrator(
            parameter_generator=mock_generator,
            evaluator=evaluator
        )
        
        # Update progress manually
        orchestrator.progress_tracker.update_progress(1.5)
        
        status = orchestrator.get_progress_status()
        
        assert status['total_evaluations'] == 1
        assert status['best_value'] == 1.5
        assert 'elapsed_seconds' in status