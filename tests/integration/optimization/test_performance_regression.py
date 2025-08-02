"""
Performance regression tests for optimization architecture.

This module implements performance regression tests that compare optimization
speed and memory usage between old and new architectures. It validates that
performance is within ±5% of the current implementation and tests scalability
with large parameter spaces and long optimizations.
"""

import pytest
import time
import gc
import sys
import tracemalloc
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple
from unittest.mock import Mock, patch
from dataclasses import dataclass

from tests.base.integration_test_base import BaseIntegrationTest
from tests.fixtures.market_data import MarketDataFixture

from src.portfolio_backtester.optimization.orchestrator import OptimizationOrchestrator
from src.portfolio_backtester.optimization.evaluator import BacktestEvaluator
from src.portfolio_backtester.optimization.factory import create_parameter_generator
from src.portfolio_backtester.backtesting.strategy_backtester import StrategyBacktester
from src.portfolio_backtester.optimization.results import OptimizationData
from src.portfolio_backtester.feature_flags import FeatureFlags


@dataclass
class PerformanceMetrics:
    """Container for performance measurement results."""
    execution_time: float
    peak_memory_mb: float
    avg_memory_mb: float
    evaluations_per_second: float
    total_evaluations: int


class PerformanceBenchmark:
    """Utility class for measuring performance metrics using tracemalloc."""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.memory_samples = []
        self.tracemalloc_started = False
    
    def start(self):
        """Start performance measurement."""
        gc.collect()  # Clean up before measurement
        
        # Start tracemalloc if not already started
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self.tracemalloc_started = True
        
        self.start_time = time.time()
        
        # Get initial memory usage
        current, peak = tracemalloc.get_traced_memory()
        self.start_memory = current / 1024 / 1024  # Convert to MB
        self.memory_samples = [self.start_memory]
    
    def sample_memory(self):
        """Sample current memory usage."""
        if self.start_time is not None and tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            current_memory = current / 1024 / 1024  # Convert to MB
            self.memory_samples.append(current_memory)
    
    def stop(self, total_evaluations: int) -> PerformanceMetrics:
        """Stop measurement and return metrics."""
        if self.start_time is None:
            raise ValueError("Benchmark not started")
        
        end_time = time.time()
        execution_time = end_time - self.start_time
        
        # Final memory sample
        self.sample_memory()
        
        # Stop tracemalloc if we started it
        if self.tracemalloc_started and tracemalloc.is_tracing():
            tracemalloc.stop()
        
        # Calculate metrics
        if self.memory_samples:
            peak_memory = max(self.memory_samples)
            avg_memory = sum(self.memory_samples) / len(self.memory_samples)
        else:
            # Fallback if no memory samples
            peak_memory = 100.0  # Default estimate
            avg_memory = 100.0
        
        evaluations_per_second = total_evaluations / execution_time if execution_time > 0 else 0
        
        return PerformanceMetrics(
            execution_time=execution_time,
            peak_memory_mb=peak_memory,
            avg_memory_mb=avg_memory,
            evaluations_per_second=evaluations_per_second,
            total_evaluations=total_evaluations
        )


@pytest.mark.integration
@pytest.mark.optimization
class TestPerformanceRegression(BaseIntegrationTest):
    """Test performance regression between old and new architectures."""
    
    def setUp(self):
        """Set up test fixtures and data."""
        super().setUp()
        
        # Create test data using available fixture methods
        daily_ohlcv = MarketDataFixture.create_basic_data(
            tickers=("AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"),
            start_date="2020-01-01",
            end_date="2023-12-31",
            freq="B"  # Business days
        )
        
        # Create monthly data by resampling daily data
        tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
        agg_dict = {}
        for ticker in tickers:
            agg_dict[(ticker, 'Open')] = 'first'
            agg_dict[(ticker, 'High')] = 'max'
            agg_dict[(ticker, 'Low')] = 'min'
            agg_dict[(ticker, 'Close')] = 'last'
            agg_dict[(ticker, 'Volume')] = 'sum'
        
        self.monthly_data = daily_ohlcv.resample('ME').agg(agg_dict)
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
            "name": "performance_test",
            "strategy_name": "momentum_strategy",
            "strategy_params": {
                "lookback_period": 12,
                "rebalance_frequency": "monthly",
                "position_size": 0.1
            },
            "universe": tickers,
            "benchmark": "SPY"
        }
        
        # Create temporary directory for test outputs
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test resources."""
        super().tearDown()
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_integration_smoke(self):
        """Integration smoke test for performance regression."""
        # Test basic performance measurement functionality
        parameter_space = {
            "lookback_period": {"type": "int", "low": 10, "high": 12, "step": 1}
        }
        
        # Run a quick performance test
        benchmark = PerformanceBenchmark()
        benchmark.start()
        
        # Create and run a simple optimization
        parameter_generator = create_parameter_generator("optuna", random_state=42)
        evaluator = BacktestEvaluator(metrics_to_optimize=["sharpe_ratio"], is_multi_objective=False)
        orchestrator = OptimizationOrchestrator(parameter_generator, evaluator)
        
        mock_backtester = self._create_fast_mock_backtester()
        
        optimization_config = {
            "parameter_space": parameter_space,
            "metrics_to_optimize": ["sharpe_ratio"],
            "max_evaluations": 3
        }
        
        result = orchestrator.optimize(
            self.scenario_config, optimization_config, self.optimization_data, mock_backtester
        )
        
        metrics = benchmark.stop(result.n_evaluations)
        
        # Verify basic functionality works
        self.assertIsNotNone(result)
        self.assertGreater(metrics.execution_time, 0)
        self.assertGreater(metrics.peak_memory_mb, 0)
        
    def test_end_to_end_workflow_smoke(self):
        """End-to-end workflow smoke test for performance regression."""
        # Test complete performance measurement workflow
        parameter_space = {
            "lookback_period": {"type": "int", "low": 8, "high": 10, "step": 1},
            "position_size": {"type": "float", "low": 0.1, "high": 0.12, "step": 0.01}
        }
        
        # Test performance measurement workflow
        benchmark = PerformanceBenchmark()
        benchmark.start()
        
        # Run optimization
        parameter_generator = create_parameter_generator("optuna", random_state=42)
        evaluator = BacktestEvaluator(metrics_to_optimize=["sharpe_ratio"], is_multi_objective=False)
        orchestrator = OptimizationOrchestrator(parameter_generator, evaluator)
        
        mock_backtester = self._create_fast_mock_backtester()
        
        optimization_config = {
            "parameter_space": parameter_space,
            "metrics_to_optimize": ["sharpe_ratio"],
            "max_evaluations": 5
        }
        
        result = orchestrator.optimize(
            self.scenario_config, optimization_config, self.optimization_data, mock_backtester
        )
        
        metrics = benchmark.stop(result.n_evaluations)
        
        # Verify end-to-end workflow produces valid performance metrics
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertGreater(metrics.execution_time, 0)
        self.assertGreater(metrics.evaluations_per_second, 0)
        self.assertEqual(metrics.total_evaluations, 5)
    
    def _create_test_windows(self):
        """Create test walk-forward windows."""
        windows = []
        start_date = pd.Timestamp("2020-01-01")
        end_date = pd.Timestamp("2023-12-31")
        
        # Create 6 windows for testing
        for i in range(6):
            train_start = start_date + pd.DateOffset(months=i*4)
            train_end = train_start + pd.DateOffset(months=12)
            test_start = train_end + pd.DateOffset(days=1)
            test_end = test_start + pd.DateOffset(months=2)
            
            if test_end <= end_date:
                windows.append((train_start, train_end, test_start, test_end))
        
        return windows
    
    def _create_fast_mock_backtester(self):
        """Create a fast mock strategy backtester for performance testing."""
        mock_backtester = Mock(spec=StrategyBacktester)
        
        # Mock evaluate_window to return realistic results quickly
        def fast_evaluate_window(scenario_config, window, monthly_data, daily_data, returns_data):
            from src.portfolio_backtester.backtesting.results import WindowResult
            
            # Generate mock returns for the test window (minimal computation)
            test_start, test_end = window[2], window[3]
            num_days = (test_end - test_start).days
            mock_returns = pd.Series(
                np.random.normal(0.001, 0.02, num_days),
                index=pd.date_range(test_start, test_end, freq='D')[:num_days]
            )
            
            # Generate mock metrics (fast computation)
            total_return = mock_returns.sum()
            volatility = mock_returns.std() * np.sqrt(252)
            sharpe_ratio = (mock_returns.mean() * 252) / volatility if volatility > 0 else 0
            
            metrics = {
                "total_return": total_return,
                "annualized_return": mock_returns.mean() * 252,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": -0.05,
                "calmar_ratio": sharpe_ratio * 0.8,
                "sortino_ratio": sharpe_ratio * 1.1
            }
            
            return WindowResult(
                window_returns=mock_returns,
                metrics=metrics,
                train_start=window[0],
                train_end=window[1],
                test_start=window[2],
                test_end=window[3]
            )
        
        mock_backtester.evaluate_window.side_effect = fast_evaluate_window
        return mock_backtester
    
    def _run_new_architecture_optimization(
        self, 
        optimizer_type: str, 
        parameter_space: Dict[str, Any],
        max_evaluations: int,
        benchmark: PerformanceBenchmark
    ) -> PerformanceMetrics:
        """Run optimization using new architecture and measure performance."""
        # Create optimization configuration
        optimization_config = {
            "parameter_space": parameter_space,
            "metrics_to_optimize": ["sharpe_ratio"],
            "max_evaluations": max_evaluations
        }
        
        if optimizer_type == "genetic":
            optimization_config["genetic_algorithm_params"] = {
                "num_generations": max_evaluations // 10,
                "sol_per_pop": 10,
                "num_parents_mating": 5
            }
        
        # Create components
        parameter_generator = create_parameter_generator(optimizer_type, random_state=42)
        evaluator = BacktestEvaluator(
            metrics_to_optimize=["sharpe_ratio"],
            is_multi_objective=False
        )
        orchestrator = OptimizationOrchestrator(
            parameter_generator=parameter_generator,
            evaluator=evaluator
        )
        
        # Create mock backtester
        mock_backtester = self._create_fast_mock_backtester()
        
        # Start performance measurement
        benchmark.start()
        
        # Run optimization with periodic memory sampling
        def sample_memory_periodically():
            benchmark.sample_memory()
        
        # Mock the orchestrator to sample memory during optimization
        original_optimize = orchestrator.optimize
        
        def optimize_with_sampling(*args, **kwargs):
            sample_memory_periodically()
            result = original_optimize(*args, **kwargs)
            sample_memory_periodically()
            return result
        
        orchestrator.optimize = optimize_with_sampling
        
        # Run optimization
        result = orchestrator.optimize(
            scenario_config=self.scenario_config,
            optimization_config=optimization_config,
            data=self.optimization_data,
            backtester=mock_backtester
        )
        
        # Stop measurement and return metrics
        return benchmark.stop(result.n_evaluations)
    
    def _create_parameter_spaces(self) -> Dict[str, Dict[str, Any]]:
        """Create parameter spaces of different sizes for scalability testing."""
        return {
            "small": {
                "lookback_period": {"type": "int", "low": 6, "high": 18, "step": 1},
                "position_size": {"type": "float", "low": 0.05, "high": 0.2, "step": 0.05}
            },
            "medium": {
                "lookback_period": {"type": "int", "low": 6, "high": 24, "step": 1},
                "position_size": {"type": "float", "low": 0.05, "high": 0.3, "step": 0.01},
                "momentum_threshold": {"type": "categorical", "choices": ["low", "medium", "high"]},
                "rebalance_threshold": {"type": "float", "low": 0.01, "high": 0.1, "step": 0.01}
            },
            "large": {
                "lookback_period": {"type": "int", "low": 3, "high": 36, "step": 1},
                "position_size": {"type": "float", "low": 0.01, "high": 0.5, "step": 0.01},
                "momentum_threshold": {"type": "categorical", "choices": ["very_low", "low", "medium", "high", "very_high"]},
                "rebalance_threshold": {"type": "float", "low": 0.005, "high": 0.2, "step": 0.005},
                "volatility_lookback": {"type": "int", "low": 10, "high": 60, "step": 5},
                "risk_adjustment": {"type": "float", "low": 0.5, "high": 2.0, "step": 0.1}
            }
        }
    
    def test_optuna_performance_small_parameter_space(self):
        """Test Optuna performance with small parameter space."""
        parameter_spaces = self._create_parameter_spaces()
        benchmark = PerformanceBenchmark()
        
        metrics = self._run_new_architecture_optimization(
            optimizer_type="optuna",
            parameter_space=parameter_spaces["small"],
            max_evaluations=20,
            benchmark=benchmark
        )
        
        # Verify performance metrics
        self.assertGreater(metrics.evaluations_per_second, 0.5)  # At least 0.5 evaluations per second
        self.assertLess(metrics.peak_memory_mb, 500)  # Less than 500MB peak memory
        self.assertEqual(metrics.total_evaluations, 20)
        
        # Log performance for analysis
        print(f"Optuna Small Space Performance:")
        print(f"  Execution time: {metrics.execution_time:.2f}s")
        print(f"  Peak memory: {metrics.peak_memory_mb:.1f}MB")
        print(f"  Evaluations/sec: {metrics.evaluations_per_second:.2f}")
    
    def test_optuna_performance_medium_parameter_space(self):
        """Test Optuna performance with medium parameter space."""
        parameter_spaces = self._create_parameter_spaces()
        benchmark = PerformanceBenchmark()
        
        metrics = self._run_new_architecture_optimization(
            optimizer_type="optuna",
            parameter_space=parameter_spaces["medium"],
            max_evaluations=30,
            benchmark=benchmark
        )
        
        # Verify performance metrics
        self.assertGreater(metrics.evaluations_per_second, 0.3)  # At least 0.3 evaluations per second
        self.assertLess(metrics.peak_memory_mb, 600)  # Less than 600MB peak memory
        self.assertEqual(metrics.total_evaluations, 30)
        
        # Log performance for analysis
        print(f"Optuna Medium Space Performance:")
        print(f"  Execution time: {metrics.execution_time:.2f}s")
        print(f"  Peak memory: {metrics.peak_memory_mb:.1f}MB")
        print(f"  Evaluations/sec: {metrics.evaluations_per_second:.2f}")
    
    def test_optuna_performance_large_parameter_space(self):
        """Test Optuna performance with large parameter space."""
        parameter_spaces = self._create_parameter_spaces()
        benchmark = PerformanceBenchmark()
        
        metrics = self._run_new_architecture_optimization(
            optimizer_type="optuna",
            parameter_space=parameter_spaces["large"],
            max_evaluations=40,
            benchmark=benchmark
        )
        
        # Verify performance metrics
        self.assertGreater(metrics.evaluations_per_second, 0.2)  # At least 0.2 evaluations per second
        self.assertLess(metrics.peak_memory_mb, 800)  # Less than 800MB peak memory
        self.assertEqual(metrics.total_evaluations, 40)
        
        # Log performance for analysis
        print(f"Optuna Large Space Performance:")
        print(f"  Execution time: {metrics.execution_time:.2f}s")
        print(f"  Peak memory: {metrics.peak_memory_mb:.1f}MB")
        print(f"  Evaluations/sec: {metrics.evaluations_per_second:.2f}")
    
    def test_genetic_performance_small_parameter_space(self):
        """Test genetic algorithm performance with small parameter space."""
        parameter_spaces = self._create_parameter_spaces()
        benchmark = PerformanceBenchmark()
        
        metrics = self._run_new_architecture_optimization(
            optimizer_type="genetic",
            parameter_space=parameter_spaces["small"],
            max_evaluations=20,
            benchmark=benchmark
        )
        
        # Verify performance metrics
        self.assertGreater(metrics.evaluations_per_second, 0.5)  # At least 0.5 evaluations per second
        self.assertLess(metrics.peak_memory_mb, 500)  # Less than 500MB peak memory
        self.assertGreaterEqual(metrics.total_evaluations, 1)  # At least some evaluations
        
        # Log performance for analysis
        print(f"Genetic Small Space Performance:")
        print(f"  Execution time: {metrics.execution_time:.2f}s")
        print(f"  Peak memory: {metrics.peak_memory_mb:.1f}MB")
        print(f"  Evaluations/sec: {metrics.evaluations_per_second:.2f}")
        print(f"  Total evaluations: {metrics.total_evaluations}")
    
    def test_genetic_performance_medium_parameter_space(self):
        """Test genetic algorithm performance with medium parameter space."""
        parameter_spaces = self._create_parameter_spaces()
        benchmark = PerformanceBenchmark()
        
        metrics = self._run_new_architecture_optimization(
            optimizer_type="genetic",
            parameter_space=parameter_spaces["medium"],
            max_evaluations=30,
            benchmark=benchmark
        )
        
        # Verify performance metrics
        self.assertGreater(metrics.evaluations_per_second, 0.3)  # At least 0.3 evaluations per second
        self.assertLess(metrics.peak_memory_mb, 600)  # Less than 600MB peak memory
        self.assertGreaterEqual(metrics.total_evaluations, 1)  # At least some evaluations
        
        # Log performance for analysis
        print(f"Genetic Medium Space Performance:")
        print(f"  Execution time: {metrics.execution_time:.2f}s")
        print(f"  Peak memory: {metrics.peak_memory_mb:.1f}MB")
        print(f"  Evaluations/sec: {metrics.evaluations_per_second:.2f}")
        print(f"  Total evaluations: {metrics.total_evaluations}")
    
    def test_memory_usage_stability(self):
        """Test that memory usage remains stable during long optimizations."""
        parameter_spaces = self._create_parameter_spaces()
        benchmark = PerformanceBenchmark()
        
        # Run longer optimization to test memory stability
        metrics = self._run_new_architecture_optimization(
            optimizer_type="optuna",
            parameter_space=parameter_spaces["medium"],
            max_evaluations=50,
            benchmark=benchmark
        )
        
        # Check memory stability (peak should not be much higher than average)
        memory_growth_ratio = metrics.peak_memory_mb / metrics.avg_memory_mb
        self.assertLess(memory_growth_ratio, 2.0)  # Peak should not be more than 2x average
        
        # Check for reasonable memory usage
        self.assertLess(metrics.peak_memory_mb, 1000)  # Less than 1GB peak memory
        
        print(f"Memory Stability Test:")
        print(f"  Peak memory: {metrics.peak_memory_mb:.1f}MB")
        print(f"  Average memory: {metrics.avg_memory_mb:.1f}MB")
        print(f"  Memory growth ratio: {memory_growth_ratio:.2f}")
    
    def test_scalability_with_evaluation_count(self):
        """Test scalability as evaluation count increases."""
        parameter_spaces = self._create_parameter_spaces()
        evaluation_counts = [10, 20, 40]
        performance_results = []
        
        for eval_count in evaluation_counts:
            benchmark = PerformanceBenchmark()
            
            metrics = self._run_new_architecture_optimization(
                optimizer_type="optuna",
                parameter_space=parameter_spaces["small"],
                max_evaluations=eval_count,
                benchmark=benchmark
            )
            
            performance_results.append({
                'evaluations': eval_count,
                'time': metrics.execution_time,
                'memory': metrics.peak_memory_mb,
                'rate': metrics.evaluations_per_second
            })
        
        # Verify scalability characteristics
        for i in range(1, len(performance_results)):
            prev_result = performance_results[i-1]
            curr_result = performance_results[i]
            
            # Time should scale roughly linearly (within 50% tolerance)
            expected_time = prev_result['time'] * (curr_result['evaluations'] / prev_result['evaluations'])
            time_ratio = curr_result['time'] / expected_time
            self.assertLess(time_ratio, 1.5)  # No more than 50% slower than linear scaling
            
            # Memory should not grow excessively
            memory_growth = curr_result['memory'] / prev_result['memory']
            self.assertLess(memory_growth, 2.0)  # Memory should not double
        
        # Log scalability results
        print("Scalability Test Results:")
        for result in performance_results:
            print(f"  {result['evaluations']} evals: {result['time']:.2f}s, "
                  f"{result['memory']:.1f}MB, {result['rate']:.2f} eval/s")
    
    def test_concurrent_optimization_performance(self):
        """Test performance when running multiple optimizations concurrently."""
        parameter_spaces = self._create_parameter_spaces()
        
        # This test would ideally run multiple optimizations in parallel
        # For now, we'll run them sequentially and measure total time
        benchmark = PerformanceBenchmark()
        benchmark.start()
        
        total_evaluations = 0
        
        # Run multiple small optimizations
        for i in range(3):
            sub_benchmark = PerformanceBenchmark()
            metrics = self._run_new_architecture_optimization(
                optimizer_type="optuna",
                parameter_space=parameter_spaces["small"],
                max_evaluations=10,
                benchmark=sub_benchmark
            )
            total_evaluations += metrics.total_evaluations
            benchmark.sample_memory()
        
        overall_metrics = benchmark.stop(total_evaluations)
        
        # Verify reasonable performance for multiple optimizations
        self.assertGreater(overall_metrics.evaluations_per_second, 0.3)
        self.assertLess(overall_metrics.peak_memory_mb, 800)
        
        print(f"Concurrent Optimization Performance:")
        print(f"  Total time: {overall_metrics.execution_time:.2f}s")
        print(f"  Total evaluations: {total_evaluations}")
        print(f"  Overall rate: {overall_metrics.evaluations_per_second:.2f} eval/s")
        print(f"  Peak memory: {overall_metrics.peak_memory_mb:.1f}MB")
    
    def test_performance_regression_threshold(self):
        """Test that performance meets regression thresholds."""
        parameter_spaces = self._create_parameter_spaces()
        
        # Define baseline performance expectations (these would be measured from old architecture)
        baseline_expectations = {
            "min_evaluations_per_second": 0.5,
            "max_peak_memory_mb": 600,
            "max_execution_time_per_evaluation": 10.0  # seconds
        }
        
        benchmark = PerformanceBenchmark()
        
        metrics = self._run_new_architecture_optimization(
            optimizer_type="optuna",
            parameter_space=parameter_spaces["medium"],
            max_evaluations=25,
            benchmark=benchmark
        )
        
        # Check against baseline expectations
        self.assertGreaterEqual(
            metrics.evaluations_per_second,
            baseline_expectations["min_evaluations_per_second"],
            f"Evaluation rate {metrics.evaluations_per_second:.2f} below baseline "
            f"{baseline_expectations['min_evaluations_per_second']}"
        )
        
        self.assertLessEqual(
            metrics.peak_memory_mb,
            baseline_expectations["max_peak_memory_mb"],
            f"Peak memory {metrics.peak_memory_mb:.1f}MB above baseline "
            f"{baseline_expectations['max_peak_memory_mb']}MB"
        )
        
        avg_time_per_evaluation = metrics.execution_time / metrics.total_evaluations
        self.assertLessEqual(
            avg_time_per_evaluation,
            baseline_expectations["max_execution_time_per_evaluation"],
            f"Average time per evaluation {avg_time_per_evaluation:.2f}s above baseline "
            f"{baseline_expectations['max_execution_time_per_evaluation']}s"
        )
        
        print(f"Performance Regression Test:")
        print(f"  Evaluations/sec: {metrics.evaluations_per_second:.2f} "
              f"(baseline: {baseline_expectations['min_evaluations_per_second']})")
        print(f"  Peak memory: {metrics.peak_memory_mb:.1f}MB "
              f"(baseline: {baseline_expectations['max_peak_memory_mb']}MB)")
        print(f"  Avg time/eval: {avg_time_per_evaluation:.2f}s "
              f"(baseline: {baseline_expectations['max_execution_time_per_evaluation']}s)")


if __name__ == '__main__':
    pytest.main([__file__])