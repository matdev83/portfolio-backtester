"""
Unit tests for StrategyBacktester class.

These tests verify that the pure backtesting engine works correctly
in complete isolation from optimization components.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from portfolio_backtester.backtesting.strategy_backtester import StrategyBacktester
from portfolio_backtester.backtesting.results import BacktestResult, WindowResult
from portfolio_backtester.optimization.wfo_window import WFOWindow
from portfolio_backtester.strategies._core.base.base.base_strategy import BaseStrategy
from portfolio_backtester.strategies._core.registry import get_strategy_registry
from portfolio_backtester.strategies.user.signal.momentum_signal_strategy import (
    MomentumSignalStrategy,
)


class TestStrategyBacktester:
    """Test suite for StrategyBacktester."""

    def setup_method(self):
        """Set up the test environment."""
        self.global_config = {
            "data_source": {"type": "memory", "data": {}},
            "benchmark": "SPY",
        }
        self.data_source = Mock()
        self.backtester = StrategyBacktester(self.global_config, self.data_source)

    def test_initialization_with_strategy_registry(self):
        """Test backtester initializes with a strategy registry."""
        registry = get_strategy_registry()
        assert self.backtester._registry is registry

    def test_get_strategy_valid(self):
        """Test getting a valid strategy instance."""
        strategy_spec = {"strategy": "DummyStrategyForTestingSignalStrategy"}
        params = {"param1": 1}
        strategy_config = {
            "strategy": strategy_spec,
            "strategy_params": params,
        }
        # Make mock_strategy inherit from BaseStrategy
        mock_strategy = Mock(spec=BaseStrategy)

        with patch.object(self.backtester._registry, "get_strategy_class", return_value=lambda params: mock_strategy):
            result = self.backtester._get_strategy(
                "DummyStrategyForTestingSignalStrategy",
                params,
                strategy_config,
            )

        assert result == mock_strategy

    def test_get_strategy_unsupported(self):
        """Test error handling for unsupported strategy."""
        with patch.object(self.backtester._registry, "get_strategy_class", return_value=None):
            with pytest.raises(ValueError, match="Unsupported strategy: nonexistent_strategy"):
                self.backtester._get_strategy("nonexistent_strategy", {}, {})

    def test_get_strategy_wrong_type(self):
        """Test error handling when strategy class returns wrong type."""

        class MockNonStrategy:
            """A class that does not inherit from BaseStrategy."""

            def __init__(self, params):
                pass

        with patch.object(self.backtester._registry, "get_strategy_class", return_value=MockNonStrategy):
            with pytest.raises(TypeError, match="did not return a BaseStrategy instance"):
                self.backtester._get_strategy("bad_strategy", {}, {})

    @patch("portfolio_backtester.backtesting.strategy_backtester.generate_signals")
    @patch("portfolio_backtester.backtesting.strategy_backtester.size_positions")
    @patch("portfolio_backtester.backtesting.strategy_backtester.calculate_portfolio_returns")
    @patch("portfolio_backtester.backtesting.strategy_backtester.prepare_scenario_data")
    @patch("portfolio_backtester.backtesting.strategy_backtester.calculate_metrics")
    def test_backtest_strategy_success(
        self,
        mock_calculate_metrics,
        mock_prepare_scenario_data,
        mock_calculate_portfolio_returns,
        mock_size_positions,
        mock_generate_signals,
    ):
        """Test a successful run of backtest_strategy."""
        daily_df, monthly_df, returns_df = self.sample_price_data()
        strategy_config = self.strategy_config()

        mock_generate_signals.return_value = pd.DataFrame(
            {"AAPL": [1], "GOOGL": [0]}, index=[daily_df.index[10]]
        )
        mock_size_positions.return_value = pd.DataFrame(
            {"AAPL": [0.5], "GOOGL": [0]}, index=[daily_df.index[10]]
        )
        mock_trade_tracker = Mock()
        mock_trade_tracker.trade_lifecycle_manager.get_completed_trades.return_value = [Mock()]
        mock_calculate_portfolio_returns.return_value = (
            pd.Series([0.01, 0.02]),
            mock_trade_tracker,
        )
        mock_prepare_scenario_data.return_value = (monthly_df, returns_df)
        mock_calculate_metrics.return_value = {"sharpe": 2.0}

        result = self.backtester.backtest_strategy(
            strategy_config, monthly_df, daily_df, returns_df
        )

        assert result is not None
        assert result.metrics["sharpe"] == 2.0
        mock_generate_signals.assert_called()
        mock_size_positions.assert_called()
        mock_calculate_portfolio_returns.assert_called()

    def test_backtest_strategy_no_universe_tickers(
        self,
    ):
        """Test backtest_strategy with no tickers in the universe."""
        daily_df, monthly_df, returns_df = self.sample_price_data()
        strategy_config = self.strategy_config()
        strategy_config["universe"] = []

        result = self.backtester.backtest_strategy(
            strategy_config, monthly_df, daily_df, returns_df
        )

        assert result is not None
        assert result.returns.empty
        assert not result.metrics

    @patch("portfolio_backtester.backtesting.strategy_backtester.calculate_metrics")
    def test_evaluate_window_success(
        self,
        mock_calculate_metrics,
    ):
        """Test a successful run of evaluate_window."""
        daily_df, monthly_df, returns_df = self.sample_price_data()
        strategy_config = self.strategy_config()

        # Mock what _run_scenario_for_window would do
        window_returns = pd.Series([0.01, -0.01, 0.02], index=pd.to_datetime(['2020-07-01', '2020-07-02', '2020-07-03']))
        
        from portfolio_backtester.optimization.wfo_window import WFOWindow
        window = WFOWindow(
            train_start=pd.Timestamp("2020-01-01"),
            train_end=pd.Timestamp("2020-06-30"),
            test_start=pd.Timestamp("2020-07-01"),
            test_end=pd.Timestamp("2020-12-31"),
        )
        
        with patch.object(self.backtester, '_run_scenario_for_window', return_value=window_returns):
            result = self.backtester.evaluate_window(
                strategy_config, window, monthly_df, daily_df, returns_df
            )

        assert result is not None
        assert not result.window_returns.empty

    def test_evaluate_window_empty_returns(self):
        """Test evaluate_window with a scenario that returns no data."""
        daily_df, monthly_df, returns_df = self.sample_price_data()
        strategy_config = self.strategy_config()

        from portfolio_backtester.optimization.wfo_window import WFOWindow
        window = WFOWindow(
            train_start=pd.Timestamp("2020-01-01"),
            train_end=pd.Timestamp("2020-06-30"),
            test_start=pd.Timestamp("2020-07-01"),
            test_end=pd.Timestamp("2020-12-31"),
        )
        
        with patch.object(self.backtester, '_run_scenario_for_window', return_value=pd.Series(dtype=float)):
            result = self.backtester.evaluate_window(
                strategy_config, window, monthly_df, daily_df, returns_df
            )

        assert result is not None
        assert result.window_returns.empty
        assert not result.metrics

    def test_create_charts_data(self):
        """Test _create_charts_data with some sample returns."""
        returns = pd.Series([0.01, -0.02, 0.03, 0.01])
        benchmark_returns = pd.Series([0.005, -0.01, 0.02, 0.005])
        charts_data = self.backtester._create_charts_data(
            returns, benchmark_returns
        )
        assert "portfolio_cumulative" in charts_data
        assert "benchmark_cumulative" in charts_data
        assert "drawdown" in charts_data
        assert "rolling_sharpe" in charts_data

    def test_calculate_max_drawdown(self):
        """Test _calculate_max_drawdown with a known drawdown."""
        returns = pd.Series([0.1, 0.1, -0.1, -0.1, 0.1, 0.1])
        # Cumulative returns: 1.1, 1.21, 1.089, 0.9801, 1.07811, 1.185921
        # Drawdown from peak at 1.21 to trough at 0.9801
        expected_drawdown = (0.9801 - 1.21) / 1.21
        assert np.isclose(
            self.backtester._calculate_max_drawdown(returns), expected_drawdown
        )

    def test_calculate_max_drawdown_empty(self):
        """Test _calculate_max_drawdown with empty returns."""
        assert self.backtester._calculate_max_drawdown(pd.Series(dtype=float)) == 0.0

    def test_create_trade_history(self):
        """Test _create_trade_history with some sample signals."""
        daily_df, _, _ = self.sample_price_data()
        sized_signals = pd.DataFrame(
            {"AAPL": [0.5, 0], "GOOGL": [0, 0.5]},
            index=daily_df.index[:2],
        )
        trade_history = self.backtester._create_trade_history(
            sized_signals, daily_df
        )
        assert not trade_history.empty
        assert "date" in trade_history.columns
        assert "ticker" in trade_history.columns
        assert "position" in trade_history.columns
        assert "price" in trade_history.columns

    def sample_price_data(self):
        """Create sample price data for testing."""
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="B")
        tickers = ["AAPL", "MSFT", "GOOGL", "SPY"]
        daily_data = {}
        for ticker in tickers:
            np.random.seed(hash(ticker) & 0xFFFFFFFF)
            prices = 100 * (1 + np.random.randn(len(dates)) * 0.02).cumprod()
            daily_data[ticker] = prices
        daily_df = pd.DataFrame(daily_data, index=dates)
        monthly_df = daily_df.resample("ME").last()
        returns_df = daily_df.pct_change().fillna(0)
        return daily_df, monthly_df, returns_df

    def strategy_config(self):
        """Sample strategy configuration."""
        return {
            "name": "test_strategy",
            "strategy": "MomentumSignalStrategy",
            "strategy_params": {"lookback_period": 20, "num_holdings": 10},
            "universe": ["AAPL", "MSFT", "GOOGL"],
            "rebalance_frequency": "monthly",
        }


class TestStrategyBacktesterSeparationOfConcerns:
    """Test separation of concerns - verify backtester works without optimization components."""

    @pytest.fixture
    def mock_global_config(self):
        """Mock global configuration."""
        return {
            "benchmark": "SPY",
            "universe": ["AAPL", "MSFT", "GOOGL"],
            "start_date": "2020-01-01",
            "end_date": "2023-12-31",
            "portfolio_value": 100000.0,
        }

    @pytest.fixture
    def mock_data_source(self):
        """Mock data source."""
        mock_source = Mock()
        mock_source.get_data.return_value = pd.DataFrame()
        return mock_source

    def test_backtester_runs_without_optimizers_disabled(
        self, mock_global_config, mock_data_source
    ):
        """Test that backtester works with all optimizers disabled via feature flags."""
        # This test simulates having optimization components disabled
        with patch(
            "portfolio_backtester.backtesting.strategy_backtester.get_strategy_registry"
        ) as mock_get_registry:
            # Mock the strategy enumerator
            mock_registry = Mock()
            mock_registry.get_all_strategies.return_value = {"dummy": Mock}
            mock_get_registry.return_value = mock_registry

            # Simulate feature flags disabling optimizers
            with patch.dict("os.environ", {"DISABLE_OPTUNA": "1"}):
                backtester = StrategyBacktester(mock_global_config, mock_data_source)

                # Verify backtester can be created and initialized
                assert backtester is not None
                assert backtester.global_config == mock_global_config
                assert backtester.data_source == mock_data_source

    def test_no_optimization_imports_in_backtester(self):
        """Test that StrategyBacktester has no optimization-related imports."""
        import inspect
        import portfolio_backtester.backtesting.strategy_backtester as backtester_module

        # Get the source code of the entire module
        source = inspect.getsource(backtester_module)

        # Check that no optimization-related imports are present in import statements
        import_lines = [
            line.strip()
            for line in source.split("\n")
            if line.strip().startswith(("import ", "from "))
        ]

        optimization_imports = ["optuna", "parameter_generator"]

        for import_line in import_lines:
            for opt_import in optimization_imports:
                assert (
                    opt_import not in import_line.lower()
                ), f"Found optimization import in: {import_line}"

    def test_backtester_module_independence(self):
        """Test that backtesting module can be imported without optimization dependencies."""
        # This test verifies that the backtesting module doesn't have hard dependencies
        # on optimization components at import time

        try:
            from portfolio_backtester.backtesting.strategy_backtester import StrategyBacktester
            from portfolio_backtester.backtesting.results import BacktestResult, WindowResult

            # If we can import these without errors, the separation is working
            assert StrategyBacktester is not None
            assert BacktestResult is not None
            assert WindowResult is not None

        except ImportError as e:
            pytest.fail(f"Backtesting module has unwanted dependencies: {e}")

    @patch("portfolio_backtester.backtesting.strategy_backtester.get_strategy_registry")
    def test_backtester_works_in_isolation(self, mock_get_registry):
        """Test that backtester can work completely in isolation."""
        # Setup mocks to simulate minimal dependencies
        mock_registry = Mock()
        mock_registry.get_all_strategies.return_value = {"test_strategy": Mock}
        mock_get_registry.return_value = mock_registry

        global_config = {
            "benchmark": "SPY",
            "universe": ["AAPL", "MSFT"],
            "start_date": "2020-01-01",
            "end_date": "2023-12-31",
        }

        data_source = Mock()

        # Create backtester in isolation
        backtester = StrategyBacktester(global_config, data_source)

        # Verify it can perform basic operations
        assert backtester._get_strategy.__name__ == "_get_strategy"
        assert backtester._create_empty_backtest_result().__class__.__name__ == "BacktestResult"

        # Test error handling works
        # Make sure get_strategy_class returns None for nonexistent strategy
        mock_registry.get_strategy_class.return_value = None
        with pytest.raises(ValueError):
            backtester._get_strategy("nonexistent_strategy", {}, {})


if __name__ == "__main__":
    pytest.main([__file__])
