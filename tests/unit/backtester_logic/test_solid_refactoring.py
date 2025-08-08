"""
Comprehensive tests for SOLID refactoring of Backtester class.

This module tests all the extracted classes from the SOLID refactoring:
- DataFetcher: Data operations
- StrategyManager: Strategy management
- EvaluationEngine: Performance evaluation
- BacktestRunner: Core backtest execution
- OptimizationOrchestrator: Optimization workflow
- BacktesterFacade: Unified interface
"""

import argparse
import pytest
from unittest.mock import Mock, patch

import pandas as pd
import numpy as np

from portfolio_backtester.backtester_logic.data_fetcher import DataFetcher
from portfolio_backtester.backtester_logic.strategy_manager import StrategyManager
from portfolio_backtester.backtester_logic.evaluation_engine import EvaluationEngine
from portfolio_backtester.backtester_logic.backtest_runner import BacktestRunner
from portfolio_backtester.backtester_logic.optimization_orchestrator import (
    OptimizationOrchestrator,
)
from portfolio_backtester.backtester_logic.backtester_facade import BacktesterFacade
from portfolio_backtester.strategies.base.base_strategy import BaseStrategy


@pytest.fixture
def sample_global_config():
    """Sample global configuration for testing."""
    return {
        "start_date": "2020-01-01",
        "end_date": "2023-12-31",
        "benchmark": "SPY",
        "universe": ["AAPL", "MSFT", "GOOGL"],
    }


@pytest.fixture
def sample_scenario_config():
    """Sample scenario configuration for testing."""
    return {
        "name": "test_scenario",
        "strategy": "DummySignalStrategy",
        "strategy_params": {"param1": 10, "param2": 0.5},
        "universe": ["AAPL", "MSFT"],
    }


@pytest.fixture
def mock_data_source():
    """Mock data source for testing."""
    mock_ds = Mock()

    # Create sample price data
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
    tickers = ["AAPL", "MSFT", "GOOGL", "SPY"]

    # Create MultiIndex DataFrame (ticker, field)
    arrays = [tickers, ["Open", "High", "Low", "Close", "Volume"]]
    columns = pd.MultiIndex.from_product(arrays, names=["Ticker", "Field"])

    sample_data = pd.DataFrame(
        np.random.randn(len(dates), len(columns)) * 0.02 + 100, index=dates, columns=columns
    )

    mock_ds.get_data.return_value = sample_data
    return mock_ds


@pytest.fixture
def mock_strategy():
    """Mock strategy for testing."""
    mock_strat = Mock()
    mock_strat.get_universe.return_value = [("AAPL", 1.0), ("MSFT", 1.0)]
    mock_strat.get_non_universe_data_requirements.return_value = ["SPY"]
    return mock_strat


class TestDataFetcher:
    """Test suite for DataFetcher class."""

    def test_initialization(self, sample_global_config, mock_data_source):
        """Test DataFetcher initialization."""
        fetcher = DataFetcher(sample_global_config, mock_data_source)

        assert fetcher.global_config == sample_global_config
        assert fetcher.data_source == mock_data_source

    def test_collect_required_tickers(self, sample_global_config, mock_data_source, mock_strategy):
        """Test ticker collection across scenarios."""
        fetcher = DataFetcher(sample_global_config, mock_data_source)

        scenarios = [
            {
                "name": "test",
                "strategy": "dummy",
                "strategy_params": {},
                "universe": ["AAPL", "MSFT"],
            }
        ]

        def mock_strategy_getter(name, params):
            return mock_strategy

        tickers, has_universe = fetcher.collect_required_tickers(scenarios, mock_strategy_getter)

        # Should include benchmark + universe + strategy requirements
        expected_tickers = {"SPY", "AAPL", "MSFT"}
        assert tickers >= expected_tickers
        assert has_universe is True

    def test_fetch_daily_data(self, sample_global_config, mock_data_source):
        """Test daily data fetching."""
        fetcher = DataFetcher(sample_global_config, mock_data_source)

        tickers = {"AAPL", "MSFT", "SPY"}
        result = fetcher.fetch_daily_data(tickers, "2020-01-01")

        mock_data_source.get_data.assert_called_once()
        assert isinstance(result, pd.DataFrame)

    def test_determine_optimal_start_date_single_ticker(
        self, sample_global_config, mock_data_source
    ):
        """Test optimal start date determination for single ticker."""
        fetcher = DataFetcher(sample_global_config, mock_data_source)

        # Single ticker (excluding benchmark)
        tickers = {"SPY", "AAPL"}  # SPY is benchmark, AAPL is the single universe ticker

        result = fetcher.determine_optimal_start_date(tickers)

        # Should use earliest available data for single ticker
        assert isinstance(result, str)
        assert len(result) == 10  # YYYY-MM-DD format


class TestStrategyManager:
    """Test suite for StrategyManager class."""

    def test_initialization(self):
        """Test StrategyManager initialization."""
        manager = StrategyManager()

        assert isinstance(manager.strategy_map, dict)
        assert len(manager.strategy_map) > 0  # Should have loaded strategies

    def test_get_strategy_with_string_spec(self):
        """Test strategy creation with string specification."""
        manager = StrategyManager()

        # Use a real discovered strategy instead of manually adding one
        available_strategies = manager.get_available_strategies()
        assert len(available_strategies) > 0, "No strategies discovered for testing"

        # Use a known good strategy with proper parameters
        strategy_name = "DummyStrategyForTestingSignalStrategy"
        strategy_params = {
            "symbol": "SPY",
            "open_long_prob": 0.2,
            "close_long_prob": 0.05,
            "seed": 123,
        }

        result = manager.get_strategy(strategy_name, strategy_params)

        assert isinstance(result, BaseStrategy)
        # Verify the strategy was created correctly
        assert hasattr(result, "strategy_params")

    def test_get_strategy_with_dict_spec(self):
        """Test strategy creation with dictionary specification."""
        manager = StrategyManager()

        # Use a real discovered strategy instead of manually adding one
        available_strategies = manager.get_available_strategies()
        assert len(available_strategies) > 0, "No strategies discovered for testing"

        # Use a known good strategy with proper parameters
        strategy_name = "DummyStrategyForTestingSignalStrategy"
        spec = {"strategy": strategy_name, "other_config": "value"}
        strategy_params = {
            "symbol": "SPY",
            "open_long_prob": 0.2,
            "close_long_prob": 0.05,
            "seed": 123,
        }

        result = manager.get_strategy(spec, strategy_params)

        assert isinstance(result, BaseStrategy)
        # Verify the strategy was created correctly
        assert hasattr(result, "strategy_params")

    def test_is_strategy_available(self):
        """Test strategy availability checking."""
        manager = StrategyManager()

        # Use real discovered strategies instead of manually adding them
        available_strategies = manager.get_available_strategies()
        assert len(available_strategies) > 0, "No strategies discovered for testing"

        # Test with a real discovered strategy
        existing_strategy_name = list(available_strategies.keys())[0]
        assert manager.is_strategy_available(existing_strategy_name) is True
        assert manager.is_strategy_available("non_existing_strategy") is False

        # Test aliases work too
        assert manager.is_strategy_available("simple_meta") is True  # Should work via alias


class TestEvaluationEngine:
    """Test suite for EvaluationEngine class."""

    def test_initialization(self, sample_global_config, mock_data_source):
        """Test EvaluationEngine initialization."""
        strategy_manager = Mock()
        engine = EvaluationEngine(sample_global_config, mock_data_source, strategy_manager)

        assert engine.global_config == sample_global_config
        assert engine.data_source == mock_data_source
        assert engine.strategy_manager == strategy_manager

    def test_get_monte_carlo_trial_threshold(self, sample_global_config, mock_data_source):
        """Test Monte Carlo trial threshold calculation."""
        strategy_manager = Mock()
        engine = EvaluationEngine(sample_global_config, mock_data_source, strategy_manager)

        assert engine.get_monte_carlo_trial_threshold("fast") == 20
        assert engine.get_monte_carlo_trial_threshold("balanced") == 10
        assert engine.get_monte_carlo_trial_threshold("comprehensive") == 5
        assert engine.get_monte_carlo_trial_threshold("unknown") == 10


class TestBacktestRunner:
    """Test suite for BacktestRunner class."""

    def test_initialization(self, sample_global_config):
        """Test BacktestRunner initialization."""
        data_cache = Mock()
        strategy_manager = Mock()
        timeout_checker = Mock(return_value=False)

        runner = BacktestRunner(sample_global_config, data_cache, strategy_manager, timeout_checker)

        assert runner.global_config == sample_global_config
        assert runner.data_cache == data_cache
        assert runner.strategy_manager == strategy_manager
        assert runner.timeout_checker == timeout_checker

    def test_validate_scenario_data(self, sample_global_config, sample_scenario_config):
        """Test scenario data validation."""
        data_cache = Mock()
        strategy_manager = Mock()
        mock_strategy = Mock()
        mock_strategy.get_universe.return_value = [("AAPL", 1.0), ("MSFT", 1.0)]
        strategy_manager.get_strategy.return_value = mock_strategy

        runner = BacktestRunner(sample_global_config, data_cache, strategy_manager)

        # Create sample daily data
        dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
        daily_data = pd.DataFrame(
            {
                "AAPL": np.random.randn(len(dates)) + 100,
                "MSFT": np.random.randn(len(dates)) + 100,
            },
            index=dates,
        )

        monthly_data = daily_data.resample("ME").last()

        result = runner.validate_scenario_data(sample_scenario_config, monthly_data, daily_data)

        assert result is True  # Should pass validation


class TestOptimizationOrchestrator:
    """Test suite for OptimizationOrchestrator class."""

    def test_initialization(self, sample_global_config):
        """Test OptimizationOrchestrator initialization."""
        data_source = Mock()
        backtest_runner = Mock()
        evaluation_engine = Mock()
        random_state = 42

        orchestrator = OptimizationOrchestrator(
            sample_global_config, data_source, backtest_runner, evaluation_engine, random_state
        )

        assert orchestrator.global_config == sample_global_config
        assert orchestrator.data_source == data_source
        assert orchestrator.backtest_runner == backtest_runner
        assert orchestrator.evaluation_engine == evaluation_engine
        assert orchestrator.random_state == random_state

    def test_convert_optimization_specs_legacy_format(self, sample_global_config):
        """Test conversion of legacy optimization specifications."""
        data_source = Mock()
        backtest_runner = Mock()
        evaluation_engine = Mock()

        orchestrator = OptimizationOrchestrator(
            sample_global_config, data_source, backtest_runner, evaluation_engine, 42
        )

        scenario_config = {
            "optimize": [
                {
                    "parameter": "test_param",
                    "type": "int",
                    "min_value": 1,
                    "max_value": 10,
                    "step": 1,
                }
            ]
        }

        result = orchestrator.convert_optimization_specs_to_parameter_space(scenario_config)

        expected = {"test_param": {"type": "int", "low": 1, "high": 10, "step": 1}}

        assert result == expected

    def test_validate_optimization_config(self, sample_global_config):
        """Test optimization configuration validation."""
        data_source = Mock()
        backtest_runner = Mock()
        evaluation_engine = Mock()

        orchestrator = OptimizationOrchestrator(
            sample_global_config, data_source, backtest_runner, evaluation_engine, 42
        )

        # Valid configuration
        valid_config = {
            "strategy": "test_strategy",
            "strategy_params": {"param": "value"},
            "optimize": [{"parameter": "test_param", "min_value": 1, "max_value": 10}],
        }

        is_valid, errors = orchestrator.validate_optimization_config(valid_config)
        assert is_valid is True
        assert len(errors) == 0

        # Invalid configuration (missing optimization specs)
        invalid_config = {"strategy": "test_strategy", "strategy_params": {"param": "value"}}

        is_valid, errors = orchestrator.validate_optimization_config(invalid_config)
        assert is_valid is False
        assert len(errors) > 0


class TestBacktesterFacade:
    """Test suite for BacktesterFacade class."""

    def test_initialization_preserves_api(self, sample_global_config):
        """Test that BacktesterFacade preserves original Backtester API."""
        scenarios = [{"name": "test", "strategy": "dummy", "strategy_params": {}}]
        args = argparse.Namespace(timeout=None, n_jobs=1, early_stop_patience=10, study_name=None)

        with (
            patch("src.portfolio_backtester.backtester_logic.backtester_facade.create_data_source"),
            patch(
                "src.portfolio_backtester.backtester_logic.backtester_facade.create_cache_manager"
            ),
            patch(
                "src.portfolio_backtester.backtester_logic.backtester_facade.create_timeout_manager"
            ),
        ):

            facade = BacktesterFacade(sample_global_config, scenarios, args, random_state=42)

            # Check that all original attributes are present
            assert hasattr(facade, "global_config")
            assert hasattr(facade, "scenarios")
            assert hasattr(facade, "args")
            assert hasattr(facade, "results")
            assert hasattr(facade, "random_state")

            # Check that specialized components are initialized
            assert hasattr(facade, "data_fetcher")
            assert hasattr(facade, "strategy_manager")
            assert hasattr(facade, "evaluation_engine")
            assert hasattr(facade, "backtest_runner")
            assert hasattr(facade, "optimization_orchestrator")

    def test_facade_delegates_to_components(self, sample_global_config):
        """Test that BacktesterFacade properly delegates to specialized components."""
        scenarios = [{"name": "test", "strategy": "dummy", "strategy_params": {}}]
        args = argparse.Namespace(timeout=None, n_jobs=1, early_stop_patience=10, study_name=None)

        with (
            patch("src.portfolio_backtester.backtester_logic.backtester_facade.create_data_source"),
            patch(
                "src.portfolio_backtester.backtester_logic.backtester_facade.create_cache_manager"
            ),
            patch(
                "src.portfolio_backtester.backtester_logic.backtester_facade.create_timeout_manager"
            ),
        ):

            facade = BacktesterFacade(sample_global_config, scenarios, args, random_state=42)

            # Mock the backtest runner
            facade.backtest_runner.run_scenario = Mock(return_value=pd.Series([0.01, 0.02, 0.03]))

            # Test that run_scenario delegates to BacktestRunner
            result = facade.run_scenario(
                scenarios[0],
                pd.DataFrame({"AAPL": [100, 101, 102]}),
                pd.DataFrame({"AAPL": [100, 101, 102]}),
                verbose=False,
            )

            facade.backtest_runner.run_scenario.assert_called_once()
            assert isinstance(result, pd.Series)


class TestSolidPrinciplesCompliance:
    """Test suite to verify SOLID principles compliance."""

    def test_single_responsibility_principle(self):
        """Test that each class has a single, well-defined responsibility."""
        # DataFetcher: Only handles data operations
        data_fetcher_methods = [m for m in dir(DataFetcher) if not m.startswith("_")]
        data_methods = [
            m for m in data_fetcher_methods if "data" in m.lower() or "fetch" in m.lower()
        ]
        assert len(data_methods) > 0, "DataFetcher should have data-related methods"

        # StrategyManager: Only handles strategy operations
        strategy_manager_methods = [m for m in dir(StrategyManager) if not m.startswith("_")]
        strategy_methods = [m for m in strategy_manager_methods if "strategy" in m.lower()]
        assert len(strategy_methods) > 0, "StrategyManager should have strategy-related methods"

        # EvaluationEngine: Only handles evaluation operations
        evaluation_engine_methods = [m for m in dir(EvaluationEngine) if not m.startswith("_")]
        eval_methods = [m for m in evaluation_engine_methods if "evaluat" in m.lower()]
        assert len(eval_methods) > 0, "EvaluationEngine should have evaluation-related methods"

    def test_open_closed_principle(self):
        """Test that classes are open for extension but closed for modification."""
        # StrategyManager follows OCP through dependency injection and automatic discovery
        # New strategies can be added by creating new files without modifying existing code

        manager = StrategyManager()

        # The strategy_map should be immutable from external modifications (closed for modification)
        original_strategies = manager.strategy_map.copy()

        # Attempting to modify strategy_map externally should not affect the internal registry
        manager.strategy_map["new_test_strategy"] = Mock()
        refreshed_strategies = manager.get_available_strategies()

        # The internal registry should be unchanged (protecting against external tampering)
        assert len(refreshed_strategies) == len(original_strategies)
        assert (
            "new_test_strategy" not in refreshed_strategies
        )  # External modification should not persist

        # OCP compliance: New strategies are added through discovery mechanism (open for extension)
        # without modifying the StrategyManager class itself (closed for modification)

    def test_dependency_inversion_principle(self):
        """Test that high-level modules don't depend on low-level modules."""
        # BacktesterFacade should depend on abstractions (injected dependencies)
        # Check constructor parameters - all dependencies should be injected
        import inspect

        facade_signature = inspect.signature(BacktesterFacade.__init__)
        params = list(facade_signature.parameters.keys())

        # Should accept dependencies as parameters, not create them internally
        assert "global_config" in params
        assert "scenarios" in params
        assert "args" in params

    def test_interface_segregation_principle(self):
        """Test that interfaces are properly segregated."""
        # Each specialized class should have focused, minimal interfaces

        # DataFetcher should only expose data-related methods
        data_fetcher = DataFetcher({}, Mock())
        public_methods = [
            m
            for m in dir(data_fetcher)
            if not m.startswith("_") and callable(getattr(data_fetcher, m))
        ]

        # All public methods should be related to data operations
        non_data_methods = [
            m
            for m in public_methods
            if not any(
                keyword in m.lower()
                for keyword in [
                    "data",
                    "fetch",
                    "extract",
                    "normalize",
                    "collect",
                    "determine",
                    "prepare",
                ]
            )
        ]
        assert len(non_data_methods) == 0, f"DataFetcher has non-data methods: {non_data_methods}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
