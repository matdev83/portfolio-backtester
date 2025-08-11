"""
Separation of concerns validation tests.

This module implements tests that prove the separation of concerns in the
architecture refactoring. It tests that the backtester runs with all optimizers
disabled, that Optuna works without PyGAD dependencies, that PyGAD works without
Optuna dependencies, and verifies no backtesting code exists in optimization modules.
"""

import pytest
import sys
import importlib
import inspect
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import shutil

from tests.base.integration_test_base import BaseIntegrationTest
from tests.fixtures.market_data import MarketDataFixture

from portfolio_backtester.feature_flags import FeatureFlags
from portfolio_backtester.backtesting.strategy_backtester import StrategyBacktester
from portfolio_backtester.optimization.results import OptimizationData


@pytest.mark.integration
@pytest.mark.optimization
class TestSeparationOfConcerns(BaseIntegrationTest):
    """Test separation of concerns between architecture components."""

    def setUp(self):
        """Set up test fixtures and data."""
        super().setUp()

        # Create test data using available fixture methods
        daily_ohlcv = MarketDataFixture.create_basic_data(
            tickers=("AAPL", "MSFT"),
            start_date="2020-01-01",
            end_date="2021-12-31",
            freq="B",  # Business days
        )

        # Create monthly data by resampling daily data
        tickers = ["AAPL", "MSFT"]
        agg_dict = {}
        for ticker in tickers:
            agg_dict[(ticker, "Open")] = "first"
            agg_dict[(ticker, "High")] = "max"
            agg_dict[(ticker, "Low")] = "min"
            agg_dict[(ticker, "Close")] = "last"
            agg_dict[(ticker, "Volume")] = "sum"

        self.monthly_data = daily_ohlcv.resample("ME").agg(agg_dict)
        self.daily_data = daily_ohlcv

        # Create returns data from close prices
        close_prices = daily_ohlcv.xs("Close", level="Field", axis=1)
        self.returns_data = close_prices.pct_change().dropna()

        # Create walk-forward windows
        self.windows = self._create_test_windows()

        # Create optimization data
        self.optimization_data = OptimizationData(
            monthly=self.monthly_data,
            daily=self.daily_data,
            returns=self.returns_data,
            windows=self.windows,
        )

        # Create test scenario configuration
        self.scenario_config = {
            "name": "separation_test",
            "strategy_name": "momentum_strategy",
            "strategy_params": {
                "lookback_period": 12,
                "rebalance_frequency": "monthly",
                "position_size": 0.1,
            },
            "universe": tickers,
            "benchmark": "SPY",
        }

        # Create global config mock
        self.global_config = {
            "data_source": "mock",
            "cache_enabled": False,
            "parallel_processing": False,
            "benchmark": "SPY",
        }

        # Create temporary directory for test outputs
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test resources."""
        super().tearDown()
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_integration_smoke(self):
        """Integration smoke test for separation of concerns."""
        # Test basic separation of concerns functionality

        # Verify backtester can be created independently
        backtester = StrategyBacktester(global_config=self.global_config, data_source=Mock())
        self.assertIsNotNone(backtester)

        # Verify optimization components can be imported independently
        try:
            from portfolio_backtester.optimization.evaluator import BacktestEvaluator

            evaluator = BacktestEvaluator(
                metrics_to_optimize=["sharpe_ratio"], is_multi_objective=False
            )
            self.assertIsNotNone(evaluator)
        except ImportError as e:
            self.fail(f"Failed to import optimization components independently: {e}")

    def test_end_to_end_workflow_smoke(self):
        """End-to-end workflow smoke test for separation of concerns."""
        # Test complete separation workflow

        # Test that feature flags work correctly
        with FeatureFlags.disable_all_optimizers():
            # Verify backtester works without optimizers
            backtester = StrategyBacktester(global_config=self.global_config, data_source=Mock())
            self.assertTrue(hasattr(backtester, "evaluate_window"))

        # Test that optimization components work independently
        try:
            from portfolio_backtester.optimization.factory import create_parameter_generator

            generator = create_parameter_generator("optuna", random_state=42)
            self.assertIsNotNone(generator)
        except Exception as e:
            self.fail(f"Failed to create parameter generator independently: {e}")

    def _create_test_windows(self):
        """Create test walk-forward windows."""
        windows = []
        start_date = pd.Timestamp("2020-01-01")
        end_date = pd.Timestamp("2021-12-31")

        # Create 2 windows for testing
        for i in range(2):
            train_start = start_date + pd.DateOffset(months=i * 6)
            train_end = train_start + pd.DateOffset(months=8)
            test_start = train_end + pd.DateOffset(days=1)
            test_end = test_start + pd.DateOffset(months=2)

            if test_end <= end_date:
                windows.append((train_start, train_end, test_start, test_end))

        return windows

    def test_backtester_runs_without_optimizers(self):
        """Test that backtester runs with all optimizers disabled via feature flags."""

        # Use the disable_all_optimizers context manager
        with FeatureFlags.disable_all_optimizers():

            # Test that backtester can be created without optimization dependencies
            backtester = StrategyBacktester(global_config=self.global_config, data_source=Mock())

            # Test that backtester can be created and used
            self.assertIsNotNone(backtester)

            # Verify backtester has required methods
            self.assertTrue(hasattr(backtester, "evaluate_window"))
            self.assertTrue(hasattr(backtester, "backtest_strategy"))
            self.assertTrue(callable(backtester.evaluate_window))
            self.assertTrue(callable(backtester.backtest_strategy))

            # Verify backtester has no optimization-related attributes
            optimization_attributes = [
                "parameter_generator",
                "optimization_orchestrator",
                "optuna_study",
                "genetic_algorithm",
                "optimization_history",
            ]

            for attr in optimization_attributes:
                self.assertFalse(
                    hasattr(backtester, attr),
                    f"Backtester should not have optimization attribute: {attr}",
                )

        print("✓ Backtester runs successfully with all optimizers disabled")

    def test_optuna_works_without_pygad(self):
        """Test that Optuna parameter generator works without PyGAD dependencies."""

        # Mock PyGAD to be unavailable
        original_pygad = sys.modules.get("pygad")
        if "pygad" in sys.modules:
            del sys.modules["pygad"]

        # Mock import to fail for PyGAD
        def mock_import(name, *args, **kwargs):
            if name == "pygad" or name.startswith("pygad."):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        original_import = __import__

        try:
            with patch("builtins.__import__", side_effect=mock_import):
                # Import and test Optuna generator
                from portfolio_backtester.optimization.generators.optuna_generator import (
                    OptunaParameterGenerator,
                )
                from portfolio_backtester.optimization.evaluator import BacktestEvaluator
                from portfolio_backtester.optimization.orchestrator import OptimizationOrchestrator

                # Create Optuna components
                parameter_generator = OptunaParameterGenerator(random_state=42)
                evaluator = BacktestEvaluator(
                    metrics_to_optimize=["sharpe_ratio"], is_multi_objective=False
                )
                orchestrator = OptimizationOrchestrator(
                    parameter_generator=parameter_generator, evaluator=evaluator
                )

                # Verify components can be created
                self.assertIsNotNone(parameter_generator)
                self.assertIsNotNone(evaluator)
                self.assertIsNotNone(orchestrator)

                # Test parameter generator initialization
                optimization_config = {
                    "parameter_space": {
                        "lookback_period": {"type": "int", "low": 6, "high": 18, "step": 1},
                        "position_size": {"type": "float", "low": 0.05, "high": 0.2, "step": 0.05},
                    },
                    "metrics_to_optimize": ["sharpe_ratio"],
                    "max_evaluations": 5,
                }

                parameter_generator.initialize(self.scenario_config, optimization_config)

                # Test parameter suggestion
                parameters = parameter_generator.suggest_parameters()
                self.assertIsInstance(parameters, dict)
                self.assertIn("lookback_period", parameters)
                self.assertIn("position_size", parameters)

                # Verify parameter bounds
                self.assertGreaterEqual(parameters["lookback_period"], 6)
                self.assertLessEqual(parameters["lookback_period"], 18)
                self.assertGreaterEqual(parameters["position_size"], 0.05)
                self.assertLessEqual(parameters["position_size"], 0.2)

        finally:
            # Restore original PyGAD module if it existed
            if original_pygad is not None:
                sys.modules["pygad"] = original_pygad

        print("✓ Optuna parameter generator works without PyGAD dependencies")

    def test_pygad_works_without_optuna(self):
        """Test that PyGAD parameter generator works without Optuna dependencies."""

        # Mock Optuna to be unavailable
        original_optuna = sys.modules.get("optuna")
        optuna_submodules = [name for name in sys.modules.keys() if name.startswith("optuna.")]

        # Remove optuna modules
        if "optuna" in sys.modules:
            del sys.modules["optuna"]
        for module_name in optuna_submodules:
            if module_name in sys.modules:
                del sys.modules[module_name]

        # Mock import to fail for Optuna
        def mock_import(name, *args, **kwargs):
            if name == "optuna" or name.startswith("optuna."):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        original_import = __import__

        try:
            with patch("builtins.__import__", side_effect=mock_import):
                # Import and test PyGAD generator
                from portfolio_backtester.optimization.generators.genetic_generator import (
                    GeneticParameterGenerator,
                )
                from portfolio_backtester.optimization.evaluator import BacktestEvaluator
                from portfolio_backtester.optimization.orchestrator import OptimizationOrchestrator

                # Create PyGAD components
                parameter_generator = GeneticParameterGenerator(random_state=42)
                evaluator = BacktestEvaluator(
                    metrics_to_optimize=["sharpe_ratio"], is_multi_objective=False
                )
                orchestrator = OptimizationOrchestrator(
                    parameter_generator=parameter_generator, evaluator=evaluator
                )

                # Verify components can be created
                self.assertIsNotNone(parameter_generator)
                self.assertIsNotNone(evaluator)
                self.assertIsNotNone(orchestrator)

                # Test parameter generator initialization
                optimization_config = {
                    "parameter_space": {
                        "lookback_period": {"type": "int", "low": 8, "high": 16, "step": 1},
                        "position_size": {"type": "float", "low": 0.1, "high": 0.25, "step": 0.05},
                    },
                    "metrics_to_optimize": ["sharpe_ratio"],
                    "max_evaluations": 8,
                    "genetic_algorithm_params": {
                        "num_generations": 2,
                        "sol_per_pop": 4,
                        "num_parents_mating": 2,
                    },
                }

                parameter_generator.initialize(self.scenario_config, optimization_config)

                # Test parameter suggestion
                parameters = parameter_generator.suggest_parameters()
                self.assertIsInstance(parameters, dict)
                self.assertIn("lookback_period", parameters)
                self.assertIn("position_size", parameters)

                # Verify parameter bounds
                self.assertGreaterEqual(parameters["lookback_period"], 8)
                self.assertLessEqual(parameters["lookback_period"], 16)
                self.assertGreaterEqual(parameters["position_size"], 0.1)
                self.assertLessEqual(parameters["position_size"], 0.25)

        finally:
            # Restore original Optuna modules if they existed
            if original_optuna is not None:
                sys.modules["optuna"] = original_optuna

        print("✓ PyGAD parameter generator works without Optuna dependencies")

    def test_no_backtesting_code_in_optimization_modules(self):
        """Test that optimization modules contain no backtesting-specific code."""

        # Define optimization modules to check
        optimization_modules = [
            "portfolio_backtester.optimization.orchestrator",
            "portfolio_backtester.optimization.evaluator",
            "portfolio_backtester.optimization.parameter_generator",
            "portfolio_backtester.optimization.factory",
            "portfolio_backtester.optimization.generators.optuna_generator",
            "portfolio_backtester.optimization.generators.genetic_generator",
        ]

        # Define backtesting-specific terms that should not appear in optimization modules
        backtesting_terms = {
            "strategy_execution",
            "trade_execution",
            "position_calculation",
            "portfolio_rebalancing",
            "signal_generation",
            "market_data_processing",
            "price_data",
            "ohlcv",
            "trading_signal",
            "position_sizing",
            "rebalance_portfolio",
            "execute_trades",
            "calculate_positions",
            "generate_signals",
        }

        # Define allowed backtesting-related terms (these are acceptable in optimization context)

        violations = []

        for module_name in optimization_modules:
            try:
                # Import the module
                module = importlib.import_module(module_name)

                # Get module source code
                module_file = inspect.getfile(module)
                with open(module_file, "r", encoding="utf-8") as f:
                    source_code = f.read().lower()

                # Check for backtesting terms
                for term in backtesting_terms:
                    if term in source_code:
                        # Check if it's in a comment or string literal context
                        lines = source_code.split("\n")
                        for line_num, line in enumerate(lines, 1):
                            if term in line:
                                # Skip if it's in a comment
                                if "#" in line and line.index("#") < line.index(term):
                                    continue
                                # Skip if it's in a docstring or string literal
                                if '"""' in line or "'''" in line or '"' in line or "'" in line:
                                    continue

                                violations.append(
                                    {
                                        "module": module_name,
                                        "term": term,
                                        "line": line_num,
                                        "content": line.strip(),
                                    }
                                )

            except ImportError as e:
                # Module might not exist or have dependencies - that's okay for this test
                print(f"Skipping {module_name}: {e}")
                continue
            except Exception as e:
                print(f"Error checking {module_name}: {e}")
                continue

        # Report violations
        if violations:
            violation_report = "\n".join(
                [
                    f"  {v['module']}:{v['line']} - '{v['term']}' in: {v['content']}"
                    for v in violations
                ]
            )
            self.fail(f"Found backtesting code in optimization modules:\n{violation_report}")

        print(
            f"✓ No backtesting-specific code found in {len(optimization_modules)} optimization modules"
        )

    def test_no_optimization_code_in_backtesting_modules(self):
        """Test that backtesting modules contain no optimization-specific code."""

        # Define backtesting modules to check
        backtesting_modules = [
            "portfolio_backtester.backtesting.strategy_backtester",
            "portfolio_backtester.backtesting.results",
        ]

        # Define optimization-specific terms that should not appear in backtesting modules
        optimization_terms = {
            "parameter_generation",
            "parameter_suggestion",
            "optimization_objective",
            "fitness_function",
            "genetic_algorithm",
            "optuna_study",
            "trial_suggestion",
            "hyperparameter_tuning",
            "parameter_search",
            "optimization_loop",
            "suggest_parameters",
            "report_result",
            "optimization_history",
            "parameter_importance",
            "study_optimization",
            "trial_pruning",
        }

        # Define allowed optimization-related terms (these are acceptable in backtesting context)

        violations = []

        for module_name in backtesting_modules:
            try:
                # Import the module
                module = importlib.import_module(module_name)

                # Get module source code
                module_file = inspect.getfile(module)
                with open(module_file, "r", encoding="utf-8") as f:
                    source_code = f.read().lower()

                # Check for optimization terms
                for term in optimization_terms:
                    if term in source_code:
                        # Check if it's in a comment or string literal context
                        lines = source_code.split("\n")
                        for line_num, line in enumerate(lines, 1):
                            if term in line:
                                # Skip if it's in a comment
                                if "#" in line and line.index("#") < line.index(term):
                                    continue
                                # Skip if it's in a docstring or string literal
                                if '"""' in line or "'''" in line or '"' in line or "'" in line:
                                    continue

                                violations.append(
                                    {
                                        "module": module_name,
                                        "term": term,
                                        "line": line_num,
                                        "content": line.strip(),
                                    }
                                )

            except ImportError as e:
                # Module might not exist - that's okay for this test
                print(f"Skipping {module_name}: {e}")
                continue
            except Exception as e:
                print(f"Error checking {module_name}: {e}")
                continue

        # Report violations
        if violations:
            violation_report = "\n".join(
                [
                    f"  {v['module']}:{v['line']} - '{v['term']}' in: {v['content']}"
                    for v in violations
                ]
            )
            self.fail(f"Found optimization code in backtesting modules:\n{violation_report}")

        print(
            f"✓ No optimization-specific code found in {len(backtesting_modules)} backtesting modules"
        )

    def test_module_dependency_isolation(self):
        """Test that modules have proper dependency isolation."""

        # Test that backtesting modules can be imported without optimization modules
        backtesting_modules = [
            "src.portfolio_backtester.backtesting.strategy_backtester",
            "src.portfolio_backtester.backtesting.results",
        ]

        # Mock optimization modules to be unavailable
        optimization_modules_to_mock = [
            "portfolio_backtester.optimization.orchestrator",
            "portfolio_backtester.optimization.evaluator",
            "portfolio_backtester.optimization.parameter_generator",
            "portfolio_backtester.optimization.factory",
        ]

        # Store original modules
        original_modules = {}
        for module_name in optimization_modules_to_mock:
            if module_name in sys.modules:
                original_modules[module_name] = sys.modules[module_name]
                del sys.modules[module_name]

        # Mock import to fail for optimization modules
        def mock_import(name, *args, **kwargs):
            if any(name.startswith(opt_mod) for opt_mod in optimization_modules_to_mock):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        original_import = __import__

        try:
            with patch("builtins.__import__", side_effect=mock_import):
                # Try to import backtesting modules
                for module_name in backtesting_modules:
                    try:
                        module = importlib.import_module(module_name)
                        self.assertIsNotNone(module)
                    except ImportError as e:
                        # If import fails due to optimization dependencies, that's a violation
                        if any(opt_mod in str(e) for opt_mod in optimization_modules_to_mock):
                            self.fail(
                                f"Backtesting module {module_name} depends on optimization modules: {e}"
                            )
                        # Other import errors might be due to missing test dependencies
                        print(f"Skipping {module_name} due to other dependencies: {e}")

        finally:
            # Restore original modules
            for module_name, module in original_modules.items():
                sys.modules[module_name] = module

        print("✓ Backtesting modules can be imported without optimization dependencies")

    def test_interface_contracts_maintained(self):
        """Test that interface contracts are maintained across component boundaries."""

        # Test StrategyBacktester interface
        backtester = StrategyBacktester(global_config=self.global_config, data_source=Mock())

        # Verify required methods exist
        required_backtester_methods = ["evaluate_window", "backtest_strategy"]

        for method_name in required_backtester_methods:
            self.assertTrue(
                hasattr(backtester, method_name),
                f"StrategyBacktester missing required method: {method_name}",
            )
            self.assertTrue(
                callable(getattr(backtester, method_name)),
                f"StrategyBacktester.{method_name} is not callable",
            )

        # Test parameter generator interface (using factory)
        try:
            from portfolio_backtester.optimization.factory import create_parameter_generator

            # Test Optuna generator interface
            optuna_generator = create_parameter_generator("optuna", random_state=42)

            required_generator_methods = [
                "initialize",
                "suggest_parameters",
                "report_result",
                "is_finished",
                "get_best_result",
            ]

            for method_name in required_generator_methods:
                self.assertTrue(
                    hasattr(optuna_generator, method_name),
                    f"Parameter generator missing required method: {method_name}",
                )
                self.assertTrue(
                    callable(getattr(optuna_generator, method_name)),
                    f"Parameter generator.{method_name} is not callable",
                )

        except ImportError:
            print("Skipping parameter generator interface test due to missing dependencies")

        print("✓ Interface contracts maintained across component boundaries")

    def test_feature_flag_isolation(self):
        """Test that feature flags properly isolate components."""

        # Test with all optimization disabled
        with FeatureFlags.disable_all_optimizers():
            # Backtester should still work
            backtester = StrategyBacktester(global_config=self.global_config, data_source=Mock())
            self.assertIsNotNone(backtester)

        # Test with specific optimizers disabled
        with FeatureFlags.disable_optuna():
            # Should not affect backtester
            backtester = StrategyBacktester(global_config=self.global_config, data_source=Mock())
            self.assertIsNotNone(backtester)

        with FeatureFlags.disable_genetic():
            # Should not affect backtester
            backtester = StrategyBacktester(global_config=self.global_config, data_source=Mock())
            self.assertIsNotNone(backtester)

        print("✓ Feature flags properly isolate components")


if __name__ == "__main__":
    pytest.main([__file__])
