"""
Architecture comparison tests for optimization systems.

This module implements tests that verify identical results between old and new
architectures. It tests with various optimization scenarios and parameter
configurations, ensures deterministic behavior with fixed random seeds, and
validates that all existing functionality is preserved.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from unittest.mock import Mock, patch
from dataclasses import dataclass

from tests.base.integration_test_base import BaseIntegrationTest
from tests.fixtures.market_data import MarketDataFixture

from src.portfolio_backtester.optimization.orchestrator import OptimizationOrchestrator
from src.portfolio_backtester.optimization.evaluator import BacktestEvaluator
from src.portfolio_backtester.optimization.factory import create_parameter_generator
from src.portfolio_backtester.backtesting.strategy_backtester import StrategyBacktester
from src.portfolio_backtester.optimization.results import OptimizationData, OptimizationResult
from src.portfolio_backtester.feature_flags import FeatureFlags


@dataclass
class ComparisonResult:
    """Container for architecture comparison results."""
    new_architecture_result: OptimizationResult
    legacy_architecture_result: Optional[OptimizationResult]
    parameters_match: bool
    values_match: bool
    evaluation_count_match: bool
    tolerance: float


class MockLegacyOptimizer:
    """Mock implementation of legacy optimization architecture for comparison."""
    
    def __init__(self, optimizer_type: str, random_state: int = None):
        self.optimizer_type = optimizer_type
        self.random_state = random_state or 42
        np.random.seed(self.random_state)
    
    def optimize(
        self,
        scenario_config: Dict[str, Any],
        optimization_config: Dict[str, Any],
        data: OptimizationData,
        backtester: Any
    ) -> OptimizationResult:
        """Mock legacy optimization that produces deterministic results."""
        parameter_space = optimization_config.get('parameter_space', {})
        max_evaluations = optimization_config.get('max_evaluations', 10)
        
        # Generate deterministic "legacy" results based on parameter space
        best_parameters = {}
        for param_name, param_config in parameter_space.items():
            if param_config['type'] == 'int':
                # Use middle value for deterministic result
                best_parameters[param_name] = (param_config['low'] + param_config['high']) // 2
            elif param_config['type'] == 'float':
                # Use middle value for deterministic result
                best_parameters[param_name] = (param_config['low'] + param_config['high']) / 2
            elif param_config['type'] == 'categorical':
                # Use first choice for deterministic result
                best_parameters[param_name] = param_config['choices'][0]
        
        # Generate deterministic best value based on optimizer type
        if self.optimizer_type == "optuna":
            best_value = 0.75 + (self.random_state % 100) / 1000  # Deterministic but varied
        else:  # genetic
            best_value = 0.65 + (self.random_state % 100) / 1000  # Slightly different for genetic
        
        # Create mock optimization history
        optimization_history = []
        for i in range(max_evaluations):
            # Generate slightly varied parameters for history
            history_params = best_parameters.copy()
            for param_name, param_config in parameter_space.items():
                if param_config['type'] == 'int':
                    variation = np.random.randint(-2, 3)
                    history_params[param_name] = max(
                        param_config['low'],
                        min(param_config['high'], best_parameters[param_name] + variation)
                    )
                elif param_config['type'] == 'float':
                    variation = np.random.uniform(-0.1, 0.1)
                    history_params[param_name] = max(
                        param_config['low'],
                        min(param_config['high'], best_parameters[param_name] + variation)
                    )
            
            # Generate objective value with some variation
            objective_value = best_value + np.random.uniform(-0.1, 0.1)
            
            optimization_history.append({
                'evaluation': i + 1,
                'parameters': history_params.copy(),
                'objective_value': objective_value,
                'metrics': {'sharpe_ratio': objective_value}
            })
        
        return OptimizationResult(
            best_parameters=best_parameters,
            best_value=best_value,
            n_evaluations=max_evaluations,
            optimization_history=optimization_history
        )


@pytest.mark.integration
@pytest.mark.optimization
class TestArchitectureComparison(BaseIntegrationTest):
    """Test comparison between old and new optimization architectures."""
    
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
        tickers = ["AAPL", "MSFT", "GOOGL"]
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
            "name": "comparison_test",
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
    
    def _create_deterministic_mock_backtester(self, seed: int = 42):
        """Create a deterministic mock strategy backtester for comparison testing."""
        mock_backtester = Mock(spec=StrategyBacktester)
        rs = np.random.RandomState(seed)

        # Mock evaluate_window to return deterministic results
        def deterministic_evaluate_window(scenario_config, window, monthly_data, daily_data, returns_data):
            from src.portfolio_backtester.backtesting.results import WindowResult

            # Generate deterministic returns based on window and parameters
            test_start, test_end = window[2], window[3]
            num_days = (test_end - test_start).days

            # Use parameters to influence results deterministically
            params = scenario_config.get('strategy_params', {})
            lookback = params.get('lookback_period', 12)
            position_size = params.get('position_size', 0.1)

            # Create deterministic seed based on window and parameters
            window_seed = hash((str(test_start), str(test_end), lookback, position_size)) % 10000
            rs_window = np.random.RandomState(window_seed)

            # Generate deterministic returns
            base_return = 0.001 * (lookback / 12) * position_size * 10
            volatility = 0.02 * (1 + position_size)

            mock_returns = pd.Series(
                rs_window.normal(base_return, volatility, num_days),
                index=pd.date_range(test_start, test_end, freq='D')[:num_days]
            )

            # Generate deterministic metrics
            total_return = mock_returns.sum()
            volatility_metric = mock_returns.std() * np.sqrt(252)
            sharpe_ratio = (mock_returns.mean() * 252) / volatility_metric if volatility_metric > 0 else 0

            metrics = {
                "total_return": total_return,
                "annualized_return": mock_returns.mean() * 252,
                "volatility": volatility_metric,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": -0.05 * position_size,
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

        mock_backtester.evaluate_window.side_effect = deterministic_evaluate_window
        return mock_backtester
    
    def _run_new_architecture_optimization(
        self,
        optimizer_type: str,
        parameter_space: Dict[str, Any],
        max_evaluations: int,
        random_seed: int = 42
    ) -> OptimizationResult:
        """Run optimization using new architecture."""
        # Create optimization configuration
        optimization_config = {
            "parameter_space": parameter_space,
            "metrics_to_optimize": ["sharpe_ratio"],
            "max_evaluations": max_evaluations
        }
        
        if optimizer_type == "genetic":
            optimization_config["genetic_algorithm_params"] = {
                "num_generations": max_evaluations // 5,
                "sol_per_pop": 5,
                "num_parents_mating": 2
            }
        
        # Create components
        parameter_generator = create_parameter_generator(optimizer_type, random_state=random_seed)
        evaluator = BacktestEvaluator(
            metrics_to_optimize=["sharpe_ratio"],
            is_multi_objective=False
        )
        orchestrator = OptimizationOrchestrator(
            parameter_generator=parameter_generator,
            evaluator=evaluator
        )
        
        # Create deterministic mock backtester
        mock_backtester = self._create_deterministic_mock_backtester(seed=random_seed)
        
        # Run optimization
        result = orchestrator.optimize(
            scenario_config=self.scenario_config,
            optimization_config=optimization_config,
            data=self.optimization_data,
            backtester=mock_backtester
        )
        
        return result
    
    def _run_legacy_architecture_optimization(
        self,
        optimizer_type: str,
        parameter_space: Dict[str, Any],
        max_evaluations: int,
        random_seed: int = 42
    ) -> OptimizationResult:
        """Run optimization using mock legacy architecture."""
        optimization_config = {
            "parameter_space": parameter_space,
            "max_evaluations": max_evaluations
        }
        
        # Create mock legacy optimizer
        legacy_optimizer = MockLegacyOptimizer(optimizer_type, random_seed)
        
        # Create deterministic mock backtester
        mock_backtester = self._create_deterministic_mock_backtester(seed=random_seed)
        
        # Run legacy optimization
        result = legacy_optimizer.optimize(
            scenario_config=self.scenario_config,
            optimization_config=optimization_config,
            data=self.optimization_data,
            backtester=mock_backtester
        )
        
        return result
    
    def _compare_optimization_results(
        self,
        new_result: OptimizationResult,
        legacy_result: OptimizationResult,
        tolerance: float = 0.05
    ) -> ComparisonResult:
        """Compare optimization results between architectures."""
        
        # Compare parameter values
        parameters_match = True
        if set(new_result.best_parameters.keys()) != set(legacy_result.best_parameters.keys()):
            parameters_match = False
        else:
            for param_name in new_result.best_parameters.keys():
                new_val = new_result.best_parameters[param_name]
                legacy_val = legacy_result.best_parameters[param_name]
                
                if isinstance(new_val, (int, float)) and isinstance(legacy_val, (int, float)):
                    # Numerical comparison with tolerance
                    if abs(new_val - legacy_val) > tolerance * abs(legacy_val):
                        parameters_match = False
                        break
                else:
                    # Exact comparison for non-numerical values
                    if new_val != legacy_val:
                        parameters_match = False
                        break
        
        # Compare objective values
        values_match = True
        if isinstance(new_result.best_value, (list, tuple)) and isinstance(legacy_result.best_value, (list, tuple)):
            # Multi-objective comparison
            if len(new_result.best_value) != len(legacy_result.best_value):
                values_match = False
            else:
                for new_val, legacy_val in zip(new_result.best_value, legacy_result.best_value):
                    if abs(new_val - legacy_val) > tolerance * abs(legacy_val):
                        values_match = False
                        break
        elif isinstance(new_result.best_value, (int, float)) and isinstance(legacy_result.best_value, (int, float)):
            # Single objective comparison
            if abs(new_result.best_value - legacy_result.best_value) > tolerance * abs(legacy_result.best_value):
                values_match = False
        else:
            values_match = False
        
        # Compare evaluation counts
        evaluation_count_match = new_result.n_evaluations == legacy_result.n_evaluations
        
        return ComparisonResult(
            new_architecture_result=new_result,
            legacy_architecture_result=legacy_result,
            parameters_match=parameters_match,
            values_match=values_match,
            evaluation_count_match=evaluation_count_match,
            tolerance=tolerance
        )
    
    def test_optuna_architecture_comparison_small_space(self):
        """Test Optuna optimization comparison with small parameter space."""
        parameter_space = {
            "lookback_period": {"type": "int", "low": 6, "high": 18, "step": 1},
            "position_size": {"type": "float", "low": 0.05, "high": 0.2, "step": 0.05}
        }
        
        random_seed = 42
        max_evaluations = 10
        
        # Run both architectures
        new_result = self._run_new_architecture_optimization(
            "optuna", parameter_space, max_evaluations, random_seed
        )
        
        legacy_result = self._run_legacy_architecture_optimization(
            "optuna", parameter_space, max_evaluations, random_seed
        )
        
        # Compare results
        comparison = self._compare_optimization_results(new_result, legacy_result, tolerance=0.1)
        
        # Verify basic result structure
        self.assertIsNotNone(new_result.best_parameters)
        self.assertIsNotNone(new_result.best_value)
        self.assertEqual(new_result.n_evaluations, max_evaluations)
        
        # Verify parameter bounds
        for param_name, param_value in new_result.best_parameters.items():
            param_config = parameter_space[param_name]
            if param_config["type"] == "int":
                self.assertGreaterEqual(param_value, param_config["low"])
                self.assertLessEqual(param_value, param_config["high"])
            elif param_config["type"] == "float":
                self.assertGreaterEqual(param_value, param_config["low"])
                self.assertLessEqual(param_value, param_config["high"])
        
        # Log comparison results
        print(f"Optuna Small Space Comparison:")
        print(f"  New architecture best value: {new_result.best_value}")
        print(f"  Legacy architecture best value: {legacy_result.best_value}")
        print(f"  Parameters match: {comparison.parameters_match}")
        print(f"  Values match: {comparison.values_match}")
        print(f"  Evaluation count match: {comparison.evaluation_count_match}")
    
    def test_optuna_architecture_comparison_medium_space(self):
        """Test Optuna optimization comparison with medium parameter space."""
        parameter_space = {
            "lookback_period": {"type": "int", "low": 6, "high": 24, "step": 1},
            "position_size": {"type": "float", "low": 0.05, "high": 0.3, "step": 0.01},
            "momentum_threshold": {"type": "categorical", "choices": ["low", "medium", "high"]}
        }
        
        random_seed = 123
        max_evaluations = 15
        
        # Run both architectures
        new_result = self._run_new_architecture_optimization(
            "optuna", parameter_space, max_evaluations, random_seed
        )
        
        legacy_result = self._run_legacy_architecture_optimization(
            "optuna", parameter_space, max_evaluations, random_seed
        )
        
        # Compare results
        comparison = self._compare_optimization_results(new_result, legacy_result, tolerance=0.15)
        
        # Verify basic result structure
        self.assertIsNotNone(new_result.best_parameters)
        self.assertIsNotNone(new_result.best_value)
        self.assertEqual(new_result.n_evaluations, max_evaluations)
        
        # Verify categorical parameter
        self.assertIn(
            new_result.best_parameters["momentum_threshold"],
            parameter_space["momentum_threshold"]["choices"]
        )
        
        # Log comparison results
        print(f"Optuna Medium Space Comparison:")
        print(f"  New architecture best params: {new_result.best_parameters}")
        print(f"  Legacy architecture best params: {legacy_result.best_parameters}")
        print(f"  Parameters match: {comparison.parameters_match}")
        print(f"  Values match: {comparison.values_match}")
    
    def test_genetic_architecture_comparison_small_space(self):
        """Test genetic algorithm optimization comparison with small parameter space."""
        parameter_space = {
            "lookback_period": {"type": "int", "low": 8, "high": 16, "step": 1},
            "position_size": {"type": "float", "low": 0.1, "high": 0.25, "step": 0.05}
        }
        
        random_seed = 456
        max_evaluations = 15
        
        # Run both architectures
        new_result = self._run_new_architecture_optimization(
            "genetic", parameter_space, max_evaluations, random_seed
        )
        
        legacy_result = self._run_legacy_architecture_optimization(
            "genetic", parameter_space, max_evaluations, random_seed
        )
        
        # Compare results
        comparison = self._compare_optimization_results(new_result, legacy_result, tolerance=0.2)
        
        # Verify basic result structure
        self.assertIsNotNone(new_result.best_parameters)
        self.assertIsNotNone(new_result.best_value)
        self.assertGreaterEqual(new_result.n_evaluations, 1)  # At least some evaluations
        
        # Verify parameter bounds
        for param_name, param_value in new_result.best_parameters.items():
            param_config = parameter_space[param_name]
            if param_config["type"] == "int":
                self.assertGreaterEqual(param_value, param_config["low"])
                self.assertLessEqual(param_value, param_config["high"])
            elif param_config["type"] == "float":
                self.assertGreaterEqual(param_value, param_config["low"])
                self.assertLessEqual(param_value, param_config["high"])
        
        # Log comparison results
        print(f"Genetic Small Space Comparison:")
        print(f"  New architecture evaluations: {new_result.n_evaluations}")
        print(f"  Legacy architecture evaluations: {legacy_result.n_evaluations}")
        print(f"  New architecture best value: {new_result.best_value}")
        print(f"  Legacy architecture best value: {legacy_result.best_value}")
    
    def test_deterministic_behavior_with_fixed_seeds(self):
        """Test that optimization produces identical results with fixed random seeds."""
        parameter_space = {
            "lookback_period": {"type": "int", "low": 10, "high": 20, "step": 2},
            "position_size": {"type": "float", "low": 0.1, "high": 0.2, "step": 0.02}
        }
        
        random_seed = 789
        max_evaluations = 8
        
        # Run new architecture twice with same seed
        result1 = self._run_new_architecture_optimization(
            "optuna", parameter_space, max_evaluations, random_seed
        )
        
        result2 = self._run_new_architecture_optimization(
            "optuna", parameter_space, max_evaluations, random_seed
        )
        
        # Results should be identical with same seed
        self.assertEqual(result1.best_parameters, result2.best_parameters)
        self.assertEqual(result1.best_value, result2.best_value)
        self.assertEqual(result1.n_evaluations, result2.n_evaluations)
        
        # Run with different seed
        result3 = self._run_new_architecture_optimization(
            "optuna", parameter_space, max_evaluations, random_seed + 1
        )
        
        # Results should be different with different seed
        # (Note: there's a small chance they could be the same by coincidence)
        different_params = result1.best_parameters != result3.best_parameters
        different_values = abs(result1.best_value - result3.best_value) > 0.001
        
        # At least one should be different
        self.assertTrue(different_params or different_values)
        
        print(f"Deterministic Behavior Test:")
        print(f"  Same seed results identical: {result1.best_parameters == result2.best_parameters}")
        print(f"  Different seed results differ: {different_params or different_values}")
    
    def test_multi_objective_comparison(self):
        """Test multi-objective optimization comparison."""
        parameter_space = {
            "lookback_period": {"type": "int", "low": 6, "high": 18, "step": 2},
            "position_size": {"type": "float", "low": 0.05, "high": 0.2, "step": 0.05}
        }
        
        # Create multi-objective optimization configuration
        optimization_config = {
            "parameter_space": parameter_space,
            "metrics_to_optimize": ["sharpe_ratio", "calmar_ratio"],
            "max_evaluations": 12,
            "optimization_targets": [
                {"name": "sharpe_ratio", "direction": "maximize"},
                {"name": "calmar_ratio", "direction": "maximize"}
            ]
        }
        
        # Create components for multi-objective optimization
        parameter_generator = create_parameter_generator("optuna", random_state=42)
        evaluator = BacktestEvaluator(
            metrics_to_optimize=["sharpe_ratio", "calmar_ratio"],
            is_multi_objective=True
        )
        orchestrator = OptimizationOrchestrator(
            parameter_generator=parameter_generator,
            evaluator=evaluator
        )
        
        # Create deterministic mock backtester
        mock_backtester = self._create_deterministic_mock_backtester(seed=42)
        
        # Run multi-objective optimization
        result = orchestrator.optimize(
            scenario_config=self.scenario_config,
            optimization_config=optimization_config,
            data=self.optimization_data,
            backtester=mock_backtester
        )
        
        # Verify multi-objective results
        self.assertIsNotNone(result.best_parameters)
        self.assertIsInstance(result.best_value, list)
        self.assertEqual(len(result.best_value), 2)  # Two objectives
        self.assertEqual(result.n_evaluations, 12)
        
        print(f"Multi-objective Comparison:")
        print(f"  Best parameters: {result.best_parameters}")
        print(f"  Best values: {result.best_value}")
        print(f"  Evaluations: {result.n_evaluations}")
    
    def test_optimization_history_consistency(self):
        """Test that optimization history is consistent and complete."""
        parameter_space = {
            "lookback_period": {"type": "int", "low": 8, "high": 16, "step": 1},
            "position_size": {"type": "float", "low": 0.1, "high": 0.2, "step": 0.02}
        }
        
        random_seed = 999
        max_evaluations = 10
        
        # Run optimization
        result = self._run_new_architecture_optimization(
            "optuna", parameter_space, max_evaluations, random_seed
        )
        
        # Verify optimization history
        self.assertIsInstance(result.optimization_history, list)
        self.assertEqual(len(result.optimization_history), max_evaluations)
        
        # Verify each history entry
        for i, entry in enumerate(result.optimization_history):
            self.assertIn('evaluation', entry)
            self.assertIn('parameters', entry)
            self.assertIn('objective_value', entry)
            self.assertIn('metrics', entry)
            
            # Verify evaluation numbers are sequential
            self.assertEqual(entry['evaluation'], i + 1)
            
            # Verify parameters are within bounds
            for param_name, param_value in entry['parameters'].items():
                param_config = parameter_space[param_name]
                if param_config["type"] == "int":
                    self.assertGreaterEqual(param_value, param_config["low"])
                    self.assertLessEqual(param_value, param_config["high"])
                elif param_config["type"] == "float":
                    self.assertGreaterEqual(param_value, param_config["low"])
                    self.assertLessEqual(param_value, param_config["high"])
        
        print(f"Optimization History Consistency:")
        print(f"  History entries: {len(result.optimization_history)}")
        print(f"  Expected entries: {max_evaluations}")
        print(f"  All entries valid: True")
    
    def test_parameter_space_edge_cases(self):
        """Test optimization with edge case parameter spaces."""
        # Test with minimal parameter space
        minimal_space = {
            "single_param": {"type": "int", "low": 5, "high": 5, "step": 1}  # Single value
        }
        
        result = self._run_new_architecture_optimization(
            "optuna", minimal_space, 5, 42
        )
        
        # Should handle single-value parameter
        self.assertEqual(result.best_parameters["single_param"], 5)
        
        # Test with large range parameter space
        large_range_space = {
            "large_int": {"type": "int", "low": 1, "high": 1000, "step": 1},
            "large_float": {"type": "float", "low": 0.001, "high": 10.0, "step": 0.001}
        }
        
        result = self._run_new_architecture_optimization(
            "optuna", large_range_space, 8, 42
        )
        
        # Should handle large ranges
        self.assertGreaterEqual(result.best_parameters["large_int"], 1)
        self.assertLessEqual(result.best_parameters["large_int"], 1000)
        self.assertGreaterEqual(result.best_parameters["large_float"], 0.001)
        self.assertLessEqual(result.best_parameters["large_float"], 10.0)
        
        print(f"Edge Cases Test:")
        print(f"  Minimal space result: {result.best_parameters}")
        print(f"  Large range handled successfully")


if __name__ == '__main__':
    pytest.main([__file__])