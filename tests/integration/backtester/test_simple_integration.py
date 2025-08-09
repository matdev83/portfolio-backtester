"""
Simplified integration tests that avoid complex mocking issues.
Focus on testing actual integration without problematic mock dependencies.
"""

import unittest
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from portfolio_backtester.strategies._core.base.base_strategy import BaseStrategy
from portfolio_backtester.timing.custom_timing_registry import TimingControllerFactory
from portfolio_backtester.timing.time_based_timing import TimeBasedTiming
from typing import Dict, Any, List


class DummyStrategy(BaseStrategy):
    """A simple dummy strategy for integration testing."""

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        tickers = all_historical_data.columns.get_level_values("Ticker").unique()
        weights = {ticker: 1.0 / len(tickers) for ticker in tickers}
        signals = pd.DataFrame(weights, index=[current_date])
        return signals


@pytest.mark.integration
class TestSimpleIntegration(unittest.TestCase):
    """Simple integration tests without complex mocking."""

    def setUp(self):
        """Set up simple test data."""
        # Create simple test data
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        tickers = ["AAPL", "MSFT"]

        # Create MultiIndex DataFrame
        np.random.seed(42)
        data_dict: Dict[str, Dict[str, Any]] = {}
        for ticker in tickers:
            prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates)))
            data_dict[ticker] = {
                "Close": prices,
                "Volume": np.random.randint(1000, 10000, len(dates)),
            }

        # Convert to MultiIndex format
        columns = pd.MultiIndex.from_product(
            [tickers, ["Close", "Volume"]], names=["Ticker", "Field"]
        )
        data_array = np.column_stack(
            [data_dict[ticker][field] for ticker in tickers for field in ["Close", "Volume"]]
        )

        self.test_data = pd.DataFrame(data_array, index=dates, columns=columns)
        self.benchmark_data = pd.DataFrame(
            {"Close": 100 * np.cumprod(1 + np.random.normal(0.0008, 0.015, len(dates)))},
            index=dates,
        )

    def test_integration_smoke(self):
        """Smoke test for basic component integration."""
        # Test strategy initialization
        strategy_config: Dict[str, Any] = {"strategy_params": {}}
        strategy = DummyStrategy(strategy_config)
        self.assertIsInstance(strategy, DummyStrategy)

        # Test timing controller initialization
        timing_config = {"mode": "time_based", "rebalance_frequency": "M"}
        timing_controller = TimingControllerFactory.create_controller(timing_config)
        self.assertIsInstance(timing_controller, TimeBasedTiming)

    def test_end_to_end_workflow_smoke(self):
        """Smoke test for the end-to-end backtesting workflow."""

        # Create a complete backtest configuration
        self.test_data[("SPY", "Close")] = self.benchmark_data["Close"]
        config: Dict[str, Any] = {
            "GLOBAL_CONFIG": {
                "benchmark": "SPY",
                "data_source": "memory",
                "start_date": "2020-01-01",
                "end_date": "2020-12-31",
                "output_dir": "test_output",
                "universe": ["AAPL", "MSFT"],
                "data_source_config": {
                    "data_frames": {
                        "daily_data": self.test_data,
                        "benchmark_data": self.benchmark_data,
                    }
                },
            },
            "BACKTEST_SCENARIOS": [
                {
                    "name": "test_scenario",
                    "strategy": "DummyStrategy",
                    "strategy_params": {},
                    "timing_config": {"mode": "time_based", "rebalance_frequency": "M"},
                    "universe_config": {"type": "fixed", "tickers": ["AAPL", "MSFT"]},
                }
            ],
        }

        from portfolio_backtester.core import Backtester

        # Run the backtester
        backtester = Backtester(
            global_config=config["GLOBAL_CONFIG"],
            scenarios=config["BACKTEST_SCENARIOS"],
            args=Mock(scenario_name="test_scenario"),
        )
        backtester.run()

        # Validate the results
        self.assertIn("test_scenario", backtester.results)
        results = backtester.results["test_scenario"]
        self.assertIn("returns", results)
        self.assertIsInstance(results["returns"], pd.Series)
        self.assertFalse(results["returns"].empty)

    def test_strategy_timing_integration(self):
        """Test strategy and timing integration."""
        # Test strategy initialization
        config: Dict[str, Any] = {"strategy_params": {}}

        strategy = DummyStrategy(config)
        self.assertIsInstance(strategy, DummyStrategy)

        # Test timing controller
        timing_controller = strategy.get_timing_controller()
        self.assertIsInstance(timing_controller, TimeBasedTiming)

    def test_data_structure_compatibility(self):
        """Test that data structures are compatible across components."""
        # Test MultiIndex structure
        self.assertIsInstance(self.test_data.columns, pd.MultiIndex)
        self.assertEqual(self.test_data.columns.names, ["Ticker", "Field"])

        # Test data access patterns
        aapl_close = self.test_data[("AAPL", "Close")]
        self.assertIsInstance(aapl_close, pd.Series)
        self.assertEqual(len(aapl_close), len(self.test_data))

    def test_component_initialization(self):
        """Test that components can be initialized without errors."""
        # Test strategy initialization with various configs
        configs: List[Dict[str, Any]] = [
            {"strategy_params": {}},
            {"strategy_params": {}},
            {"strategy_params": {}},
        ]

        for config in configs:
            try:
                strategy = DummyStrategy(config)
                self.assertIsNotNone(strategy)
            except Exception as e:
                self.fail(f"Strategy initialization failed with config {config}: {e}")


@pytest.mark.integration
@pytest.mark.fast
class TestQuickIntegration:
    """Quick integration tests for fast feedback."""

    def test_imports_work(self):
        """Test that all imports work correctly."""

        # If we get here, imports work
        assert True

    def test_basic_strategy_creation(self):
        """Test basic strategy creation."""
        config: Dict[str, Any] = {"strategy_params": {}}
        strategy = DummyStrategy(config)
        assert strategy is not None
        assert hasattr(strategy, "generate_signals")


if __name__ == "__main__":
    unittest.main()
