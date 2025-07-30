"""
End-to-end integration tests for complete optimization workflows.

This module tests complete optimization workflows using both Optuna and PyGAD
generators, verifying proper integration between all components in realistic
scenarios. It tests error handling, recovery mechanisms, and multi-objective
optimization functionality.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch

from tests.base.integration_test_base import BaseIntegrationTest
from tests.fixtures.market_data import MarketDataFixture
from tests.fixtures.strategy_data import StrategyDataFixture

from src.portfolio_backtester.optimization.orchestrator import OptimizationOrchestrator
from src.portfolio_backtester.optimization.evaluator import BacktestEvaluator
from src.portfolio_backtester.optimization.factory import create_parameter_generator
from src.portfolio_backtester.backtesting.strategy_backtester import StrategyBacktester
from src.portfolio_backtester.optimization.results import OptimizationData
from src.portfolio_backtester.feature_flags import FeatureFlags


@pytest.mark.integration
@pytest.mark.optimization
class TestEndToEndOptimization(BaseIntegrationTest):
    """Test complete optimization workflows end-to-end."""
    
    def setUp(self):
        """Set up test fixtures and data."""
        super().setUp()
        
        # Create test data using available fixture methods
        daily_ohlcv = MarketDataFixture.create_basic_data(
            tickers=("AAPL", "MSFT", "GOOGL"),
            start_date="2020-01-01",
            end_date="2023-12-31",
            freq="B"  # Business days
        )
        
        # Create monthly data by resampling daily data
        self.monthly_data = daily_ohlcv.resample('ME').agg({
            ('AAPL', 'Open'): 'first',
            ('AAPL', 'High'): 'max',
            ('AAPL', 'Low'): 'min',
            ('AAPL', 'Close'): 'last',
            ('AAPL', 'Volume'): 'sum',
            ('MSFT', 'Open'): 'first',
            ('MSFT', 'High'): 'max',
            ('MSFT', 'Low'): 'min',
            ('MSFT', 'Close'): 'last',
            ('MSFT', 'Volume'): 'sum',
            ('GOOGL', 'Open'): 'first',
            ('GOOGL', 'High'): 'max',
            ('GOOGL', 'Low'): 'min',
            ('GOOGL', 'Close'): 'last',
            ('GOOGL', 'Volume'): 'sum'
        })
        
        # Use daily data as is
        self.daily_data = daily_ohlcv
        
        # Create returns data from close prices
        close_prices = daily_ohlcv.xs('Close', level='Field', axis=1)
        self.returns_data = close_prices.pct_change().dropna()
        
        # Create walk-forward windows
        self.windows = self._create_test_windows()
        
        # Create optimization data
        self.optimization_data = OptimizationData(
            monthly=self.monthly_data,
            daily=self.daily_data,
            returns=self.returns_data,
            windows=self.windows
        )
        
        # Create test scenario configuration
        self.scenario_config = {
            "name": "test_optimization",
            "strategy_name": "momentum_strategy",
            "strategy_params": {
                "lookback_period": 12,
                "rebalance_frequency": "monthly",
                "position_size": 0.1
            },
            "universe": ["AAPL", "MSFT", "GOOGL"],
            "benchmark": "SPY"
        }
        
        # Create parameter space for optimization
        self.parameter_space = {
            "lookback_period": {
                "type": "int",
                "low": 6,
                "high": 24,
                "step": 1
            },
            "position_size": {
                "type": "float",
                "low": 0.05,
                "high": 0.3,
                "step": 0.05
            },
            "momentum_threshold": {
                "type": "categorical",
                "choices": ["low", "medium", "high"]
            }
        }
        
        # Create global config mock
        self.global_config = {
            "data_source": "mock",
            "cache_enabled": False,
            "parallel_processing": False
        }
        
        # Create temporary directory for test outputs
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test resources."""
        super().tearDown()
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_windows(self):
        """Create test walk-forward windows."""
        windows = []
        start_date = pd.Timestamp("2020-01-01")
        end_date = pd.Timestamp("2023-12-31")
        
        # Create 4 windows for testing
        for i in range(4):
            train_start = start_date + pd.DateOffset(months=i*6)
            train_end = train_start + pd.DateOffset(months=12)
            test_start = train_end + pd.DateOffset(days=1)
            test_end = test_start + pd.DateOffset(months=3)
            
            if test_end <= end_date:
                windows.append((train_start, train_end, test_start, test_end))
        
        return windows
    
    def _create_mock_backtester(self):
        """Create a mock strategy backtester for testing."""
        mock_backtester = Mock(spec=StrategyBacktester)
        
        # Mock evaluate_window to return realistic results
        def mock_evaluate_window(scenario_config, window, monthly_data, daily_data, returns_data):
            from src.portfolio_backtester.backtesting.results import WindowResult
            
            # Generate mock returns for the test window
            test_start, test_end = window[2], window[3]
            test_dates = pd.date_range(test_start, test_end, freq='D')
            mock_returns = pd.Series(
                np.random.normal(0.001, 0.02, len(test_dates)),
                index=test_dates
            )
            
            # Generate mock metrics
            total_return = mock_returns.sum()
            volatility = mock_returns.std() * np.sqrt(252)
            sharpe_ratio = (mock_returns.mean() * 252) / volatility if volatility > 0 else 0
            
            metrics = {
                "total_return": total_return,
                "annualized_return": mock_returns.mean() * 252,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": -0.05,
                "calmar_ratio": sharpe_ratio * 0.8
            }
            
            return WindowResult(
                window_returns=mock_returns,
                metrics=metrics,
                train_start=window[0],
                train_end=window[1],
                test_start=window[2],
                test_end=window[3]
            )
        
        mock_backtester.evaluate_window.side_effect = mock_evaluate_window
        return mock_backtester
    
    def test_optuna_end_to_end_single_objective(self):
        """Test complete Optuna optimization workflow with single objective."""
        # Create optimization configuration
        optimization_config = {
            "parameter_space": self.parameter_space,
            "metrics_to_optimize": ["sharpe_ratio"],
            "max_evaluations": 10,
            "optimization_targets": [
                {"name": "sharpe_ratio", "direction": "maximize"}
            ]
        }
        
        # Create components
        parameter_generator = create_parameter_generator("optuna", random_state=42)
        evaluator = BacktestEvaluator(
            metrics_to_optimize=["sharpe_ratio"],
            is_multi_objective=False
        )
        orchestrator = OptimizationOrchestrator(
            parameter_generator=parameter_generator,
            evaluator=evaluator,
            timeout_seconds=60,
            early_stop_patience=None
        )
        
        # Create mock backtester
        mock_backtester = self._create_mock_backtester()
        
        # Run optimization
        result = orchestrator.optimize(
            scenario_config=self.scenario_config,
            optimization_config=optimization_config,
            data=self.optimization_data,
            backtester=mock_backtester
        )
        
        # Verify results
        self.assertIsNotNone(result)
        self.assertIsInstance(result.best_parameters, dict)
        self.assertIsInstance(result.best_value, (int, float))
        self.assertEqual(result.n_evaluations, 10)
        self.assertIsInstance(result.optimization_history, list)
        
        # Verify parameter values are within bounds
        for param_name, param_value in result.best_parameters.items():
            param_config = self.parameter_space[param_name]
            if param_config["type"] == "int":
                self.assertGreaterEqual(param_value, param_config["low"])
                self.assertLessEqual(param_value, param_config["high"])
            elif param_config["type"] == "float":
                self.assertGreaterEqual(param_value, param_config["low"])
                self.assertLessEqual(param_value, param_config["high"])
            elif param_config["type"] == "categorical":
                self.assertIn(param_value, param_config["choices"])
        
        # Verify backtester was called correctly
        self.assertEqual(mock_backtester.evaluate_window.call_count, 10 * len(self.windows))
    
    def test_optuna_end_to_end_multi_objective(self):
        """Test complete Optuna optimization workflow with multi-objective."""
        # Create multi-objective optimization configuration
        optimization_config = {
            "parameter_space": self.parameter_space,
            "metrics_to_optimize": ["sharpe_ratio", "calmar_ratio"],
            "max_evaluations": 8,
            "optimization_targets": [
                {"name": "sharpe_ratio", "direction": "maximize"},
                {"name": "calmar_ratio", "direction": "maximize"}
            ]
        }
        
        # Create components
        parameter_generator = create_parameter_generator("optuna", random_state=42)
        evaluator = BacktestEvaluator(
            metrics_to_optimize=["sharpe_ratio", "calmar_ratio"],
            is_multi_objective=True
        )
        orchestrator = OptimizationOrchestrator(
            parameter_generator=parameter_generator,
            evaluator=evaluator,
            timeout_seconds=60
        )
        
        # Create mock backtester
        mock_backtester = self._create_mock_backtester()
        
        # Run optimization
        result = orchestrator.optimize(
            scenario_config=self.scenario_config,
            optimization_config=optimization_config,
            data=self.optimization_data,
            backtester=mock_backtester
        )
        
        # Verify results
        self.assertIsNotNone(result)
        self.assertIsInstance(result.best_parameters, dict)
        self.assertIsInstance(result.best_value, list)
        self.assertEqual(len(result.best_value), 2)  # Two objectives
        self.assertEqual(result.n_evaluations, 8)
        
        # Verify backtester was called correctly
        self.assertEqual(mock_backtester.evaluate_window.call_count, 8 * len(self.windows))
    
    def test_genetic_end_to_end_single_objective(self):
        """Test complete genetic algorithm optimization workflow."""
        # Create optimization configuration
        optimization_config = {
            "parameter_space": self.parameter_space,
            "metrics_to_optimize": ["sharpe_ratio"],
            "max_evaluations": 12,
            "genetic_algorithm_params": {
                "num_generations": 3,
                "sol_per_pop": 4,
                "num_parents_mating": 2
            }
        }
        
        # Create components
        parameter_generator = create_parameter_generator("genetic", random_state=42)
        evaluator = BacktestEvaluator(
            metrics_to_optimize=["sharpe_ratio"],
            is_multi_objective=False
        )
        orchestrator = OptimizationOrchestrator(
            parameter_generator=parameter_generator,
            evaluator=evaluator,
            timeout_seconds=60
        )
        
        # Create mock backtester
        mock_backtester = self._create_mock_backtester()
        
        # Run optimization
        result = orchestrator.optimize(
            scenario_config=self.scenario_config,
            optimization_config=optimization_config,
            data=self.optimization_data,
            backtester=mock_backtester
        )
        
        # Verify results
        self.assertIsNotNone(result)
        self.assertIsInstance(result.best_parameters, dict)
        self.assertIsInstance(result.best_value, (int, float))
        self.assertGreaterEqual(result.n_evaluations, 1)  # At least one evaluation
        self.assertIsInstance(result.optimization_history, list)
        
        # Verify parameter values are within bounds
        for param_name, param_value in result.best_parameters.items():
            param_config = self.parameter_space[param_name]
            if param_config["type"] == "int":
                self.assertGreaterEqual(param_value, param_config["low"])
                self.assertLessEqual(param_value, param_config["high"])
            elif param_config["type"] == "float":
                self.assertGreaterEqual(param_value, param_config["low"])
                self.assertLessEqual(param_value, param_config["high"])
            elif param_config["type"] == "categorical":
                self.assertIn(param_value, param_config["choices"])
    
    def test_optimization_with_timeout(self):
        """Test optimization with timeout handling."""
        # Create optimization configuration with very short timeout
        optimization_config = {
            "parameter_space": self.parameter_space,
            "metrics_to_optimize": ["sharpe_ratio"],
            "max_evaluations": 100  # More than we can complete in timeout
        }
        
        # Create components with short timeout
        parameter_generator = create_parameter_generator("optuna", random_state=42)
        evaluator = BacktestEvaluator(
            metrics_to_optimize=["sharpe_ratio"],
            is_multi_objective=False
        )
        orchestrator = OptimizationOrchestrator(
            parameter_generator=parameter_generator,
            evaluator=evaluator,
            timeout_seconds=1  # Very short timeout
        )
        
        # Create mock backtester with delay
        mock_backtester = self._create_mock_backtester()
        
        def slow_evaluate_window(*args, **kwargs):
            import time
            time.sleep(0.1)  # Add delay to trigger timeout
            return mock_backtester.evaluate_window.side_effect(*args, **kwargs)
        
        mock_backtester.evaluate_window.side_effect = slow_evaluate_window
        
        # Run optimization (should timeout)
        result = orchestrator.optimize(
            scenario_config=self.scenario_config,
            optimization_config=optimization_config,
            data=self.optimization_data,
            backtester=mock_backtester
        )
        
        # Verify that optimization stopped due to timeout
        self.assertIsNotNone(result)
        self.assertLess(result.n_evaluations, 100)  # Should not complete all evaluations
    
    def test_optimization_with_early_stopping(self):
        """Test optimization with early stopping."""
        # Create optimization configuration with early stopping
        optimization_config = {
            "parameter_space": self.parameter_space,
            "metrics_to_optimize": ["sharpe_ratio"],
            "max_evaluations": 50
        }
        
        # Create components with early stopping
        parameter_generator = create_parameter_generator("optuna", random_state=42)
        evaluator = BacktestEvaluator(
            metrics_to_optimize=["sharpe_ratio"],
            is_multi_objective=False
        )
        orchestrator = OptimizationOrchestrator(
            parameter_generator=parameter_generator,
            evaluator=evaluator,
            early_stop_patience=3  # Stop after 3 evaluations without improvement
        )
        
        # Create mock backtester that returns constant results (no improvement)
        mock_backtester = Mock(spec=StrategyBacktester)
        
        def constant_evaluate_window(scenario_config, window, monthly_data, daily_data, returns_data):
            from src.portfolio_backtester.backtesting.results import WindowResult
            
            # Return constant results to trigger early stopping
            test_start, test_end = window[2], window[3]
            test_dates = pd.date_range(test_start, test_end, freq='D')
            mock_returns = pd.Series(
                [0.001] * len(test_dates),  # Constant returns
                index=test_dates
            )
            
            metrics = {
                "sharpe_ratio": 0.5,  # Constant Sharpe ratio
                "total_return": 0.1,
                "volatility": 0.15
            }
            
            return WindowResult(
                window_returns=mock_returns,
                metrics=metrics,
                train_start=window[0],
                train_end=window[1],
                test_start=window[2],
                test_end=window[3]
            )
        
        mock_backtester.evaluate_window.side_effect = constant_evaluate_window
        
        # Run optimization (should stop early)
        result = orchestrator.optimize(
            scenario_config=self.scenario_config,
            optimization_config=optimization_config,
            data=self.optimization_data,
            backtester=mock_backtester
        )
        
        # Verify that optimization stopped early
        self.assertIsNotNone(result)
        self.assertLess(result.n_evaluations, 50)  # Should stop before max evaluations
    
    def test_optimization_error_handling(self):
        """Test optimization error handling and recovery."""
        # Create optimization configuration
        optimization_config = {
            "parameter_space": self.parameter_space,
            "metrics_to_optimize": ["sharpe_ratio"],
            "max_evaluations": 10
        }
        
        # Create components
        parameter_generator = create_parameter_generator("optuna", random_state=42)
        evaluator = BacktestEvaluator(
            metrics_to_optimize=["sharpe_ratio"],
            is_multi_objective=False
        )
        orchestrator = OptimizationOrchestrator(
            parameter_generator=parameter_generator,
            evaluator=evaluator
        )
        
        # Create mock backtester that fails sometimes
        mock_backtester = Mock(spec=StrategyBacktester)
        call_count = 0
        
        def failing_evaluate_window(scenario_config, window, monthly_data, daily_data, returns_data):
            nonlocal call_count
            call_count += 1
            
            # Fail every 3rd call
            if call_count % 3 == 0:
                raise ValueError("Simulated evaluation failure")
            
            from src.portfolio_backtester.backtesting.results import WindowResult
            
            test_start, test_end = window[2], window[3]
            test_dates = pd.date_range(test_start, test_end, freq='D')
            mock_returns = pd.Series(
                np.random.normal(0.001, 0.02, len(test_dates)),
                index=test_dates
            )
            
            metrics = {
                "sharpe_ratio": np.random.normal(0.5, 0.2),
                "total_return": mock_returns.sum(),
                "volatility": mock_returns.std() * np.sqrt(252)
            }
            
            return WindowResult(
                window_returns=mock_returns,
                metrics=metrics,
                train_start=window[0],
                train_end=window[1],
                test_start=window[2],
                test_end=window[3]
            )
        
        mock_backtester.evaluate_window.side_effect = failing_evaluate_window
        
        # Run optimization (should handle errors gracefully)
        result = orchestrator.optimize(
            scenario_config=self.scenario_config,
            optimization_config=optimization_config,
            data=self.optimization_data,
            backtester=mock_backtester
        )
        
        # Verify that optimization completed despite errors
        self.assertIsNotNone(result)
        self.assertIsInstance(result.best_parameters, dict)
        # Should have some evaluations, but may be less than max due to failures
        self.assertGreaterEqual(result.n_evaluations, 1)
    
    def test_progress_tracking(self):
        """Test optimization progress tracking."""
        # Create optimization configuration
        optimization_config = {
            "parameter_space": self.parameter_space,
            "metrics_to_optimize": ["sharpe_ratio"],
            "max_evaluations": 5
        }
        
        # Create components
        parameter_generator = create_parameter_generator("optuna", random_state=42)
        evaluator = BacktestEvaluator(
            metrics_to_optimize=["sharpe_ratio"],
            is_multi_objective=False
        )
        orchestrator = OptimizationOrchestrator(
            parameter_generator=parameter_generator,
            evaluator=evaluator
        )
        
        # Create mock backtester
        mock_backtester = self._create_mock_backtester()
        
        # Track progress during optimization
        progress_updates = []
        
        def track_progress():
            status = orchestrator.get_progress_status()
            progress_updates.append(status.copy())
        
        # Mock the orchestrator to track progress
        original_optimize = orchestrator.optimize
        
        def optimize_with_tracking(*args, **kwargs):
            # Track initial status
            track_progress()
            
            # Run optimization
            result = original_optimize(*args, **kwargs)
            
            # Track final status
            track_progress()
            
            return result
        
        orchestrator.optimize = optimize_with_tracking
        
        # Run optimization
        result = orchestrator.optimize(
            scenario_config=self.scenario_config,
            optimization_config=optimization_config,
            data=self.optimization_data,
            backtester=mock_backtester
        )
        
        # Verify progress tracking
        self.assertGreaterEqual(len(progress_updates), 2)  # At least initial and final
        
        # Verify progress data structure
        for status in progress_updates:
            self.assertIn('total_evaluations', status)
            self.assertIn('best_value', status)
            self.assertIn('elapsed_seconds', status)
            self.assertIsInstance(status['total_evaluations'], int)
            self.assertIsInstance(status['elapsed_seconds'], (int, float))
    
    def test_component_integration_validation(self):
        """Test that all components integrate properly."""
        # Test parameter generator creation
        optuna_generator = create_parameter_generator("optuna", random_state=42)
        genetic_generator = create_parameter_generator("genetic", random_state=42)
        
        self.assertIsNotNone(optuna_generator)
        self.assertIsNotNone(genetic_generator)
        
        # Test evaluator creation
        evaluator = BacktestEvaluator(
            metrics_to_optimize=["sharpe_ratio"],
            is_multi_objective=False
        )
        self.assertIsNotNone(evaluator)
        
        # Test orchestrator creation
        orchestrator = OptimizationOrchestrator(
            parameter_generator=optuna_generator,
            evaluator=evaluator
        )
        self.assertIsNotNone(orchestrator)
        
        # Test that components have required interfaces
        self.assertTrue(hasattr(optuna_generator, 'initialize'))
        self.assertTrue(hasattr(optuna_generator, 'suggest_parameters'))
        self.assertTrue(hasattr(optuna_generator, 'report_result'))
        self.assertTrue(hasattr(optuna_generator, 'is_finished'))
        self.assertTrue(hasattr(optuna_generator, 'get_best_result'))
        
        self.assertTrue(hasattr(evaluator, 'evaluate_parameters'))
        
        self.assertTrue(hasattr(orchestrator, 'optimize'))
        self.assertTrue(hasattr(orchestrator, 'get_progress_status'))


if __name__ == '__main__':
    pytest.main([__file__])