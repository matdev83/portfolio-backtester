"""
Unit tests for WindowEvaluator class.

Tests the window evaluation engine for daily strategy evaluation.
"""

import pandas as pd
import numpy as np
from unittest.mock import Mock
from portfolio_backtester.backtesting.window_evaluator import WindowEvaluator
from portfolio_backtester.optimization.wfo_window import WFOWindow
from portfolio_backtester.backtesting.results import WindowResult
from portfolio_backtester.backtesting.strategy_backtester import StrategyBacktester


class TestWindowEvaluator:
    """Test cases for WindowEvaluator class."""

    def test_window_evaluator_initialization(self):
        """Test basic window evaluator initialization."""
        mock_backtester = Mock(spec=StrategyBacktester)
        evaluator = WindowEvaluator(backtester=mock_backtester)

        assert evaluator.data_cache == {}
        assert evaluator.backtester == mock_backtester

        # Test with custom cache
        custom_cache = {"test": "data"}
        evaluator_with_cache = WindowEvaluator(
            backtester=mock_backtester, data_cache=custom_cache
        )
        assert evaluator_with_cache.data_cache == custom_cache
        assert evaluator_with_cache.backtester == mock_backtester

    def test_evaluate_window_with_mock_strategy(self):
        """Test window evaluation with a mock strategy."""
        mock_backtester = Mock(spec=StrategyBacktester)
        evaluator = WindowEvaluator(backtester=mock_backtester)

        # Mock the backtester to return proper data
        mock_result = Mock()
        mock_result.returns = [0.01, 0.02]  # 2 days of returns
        mock_result.metrics = {
            "total_return": 0.03,
            "sharpe_ratio": 1.5,
            "volatility": 0.1,
        }
        mock_result.trade_history = pd.DataFrame()
        mock_result.performance_stats = {"final_weights": {"TLT": 0.5, "SPY": 0.5}}
        mock_backtester.backtest_strategy.return_value = mock_result

        # Create test window
        window = WFOWindow(
            train_start=pd.Timestamp("2024-01-01"),
            train_end=pd.Timestamp("2024-12-31"),
            test_start=pd.Timestamp("2025-01-01"),
            test_end=pd.Timestamp("2025-01-03"),  # Short window for testing
            evaluation_frequency="D",
        )

        # Create test data
        dates = pd.date_range("2024-01-01", "2025-01-10", freq="D")
        daily_data = pd.DataFrame(
            {
                "TLT": np.random.randn(len(dates)) * 0.01
                + 1.0001,  # Small random returns around 1
                "SPY": np.random.randn(len(dates)) * 0.01 + 1.0001,
            },
            index=dates,
        )
        daily_data = (
            daily_data.cumprod() * 100
        )  # Convert to price series starting at 100

        benchmark_data = daily_data[["SPY"]].copy()

        # Create mock strategy
        mock_strategy = Mock()

        # Mock strategy returns simple signals (buy TLT on first day, hold, then sell)
        def mock_generate_signals(*args, **kwargs):
            current_date = kwargs.get("current_date")
            if current_date == pd.Timestamp("2025-01-01"):
                return pd.DataFrame({"TLT": [1.0], "SPY": [0.0]}, index=[current_date])
            elif current_date == pd.Timestamp("2025-01-03"):
                return pd.DataFrame({"TLT": [0.0], "SPY": [0.0]}, index=[current_date])
            else:
                return pd.DataFrame({"TLT": [1.0], "SPY": [0.0]}, index=[current_date])

        mock_strategy.generate_signals = mock_generate_signals

        # Evaluate window
        result = evaluator.evaluate_window(
            window=window,
            strategy=mock_strategy,
            daily_data=daily_data,
            full_monthly_data=daily_data.resample("ME").last(),
            full_rets_daily=daily_data.pct_change().dropna(),
            benchmark_data=benchmark_data,
            universe_tickers=["TLT", "SPY"],
            benchmark_ticker="SPY",
        )

        # Check result structure
        assert isinstance(result, WindowResult)
        assert result.train_start == window.train_start
        assert result.train_end == window.train_end
        assert result.test_start == window.test_start
        assert result.test_end == window.test_end

        # Should have some returns (2 days of returns for 3-day window)
        assert len(result.window_returns) == 2

        # Should have metrics
        assert "total_return" in result.metrics
        assert "sharpe_ratio" in result.metrics
        assert "volatility" in result.metrics

    def test_evaluate_window_no_evaluation_dates(self):
        """Test window evaluation when no evaluation dates are available."""
        mock_backtester = Mock(spec=StrategyBacktester)
        evaluator = WindowEvaluator(backtester=mock_backtester)

        # Mock the backtester to return proper data
        mock_result = Mock()
        mock_result.returns = []  # Empty returns for out-of-range dates
        mock_result.metrics = {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "volatility": 0.0,
        }
        mock_result.trade_history = pd.DataFrame()
        mock_result.performance_stats = {"final_weights": {}}
        mock_backtester.backtest_strategy.return_value = mock_result

        # Create window with dates outside available data
        window = WFOWindow(
            train_start=pd.Timestamp("2024-01-01"),
            train_end=pd.Timestamp("2024-12-31"),
            test_start=pd.Timestamp("2026-01-01"),  # Future date
            test_end=pd.Timestamp("2026-01-31"),
            evaluation_frequency="D",
        )

        # Create test data that doesn't cover the test window
        dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
        daily_data = pd.DataFrame(
            {"TLT": np.ones(len(dates)) * 100, "SPY": np.ones(len(dates)) * 400},
            index=dates,
        )

        benchmark_data = daily_data[["SPY"]].copy()
        mock_strategy = Mock()

        # Evaluate window
        result = evaluator.evaluate_window(
            window=window,
            strategy=mock_strategy,
            daily_data=daily_data,
            full_monthly_data=daily_data.resample("ME").last(),
            full_rets_daily=daily_data.pct_change().dropna(),
            benchmark_data=benchmark_data,
            universe_tickers=["TLT", "SPY"],
            benchmark_ticker="SPY",
        )

        # Should return empty result
        assert len(result.window_returns) == 0
        assert result.trades == []
        assert result.final_weights == {}
        assert result.metrics["total_return"] == 0.0
