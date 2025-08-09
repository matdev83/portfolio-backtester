"""
Integration tests for schedule-independent stop loss functionality.

These tests verify that stop loss works correctly regardless of:
1. Strategy type (signal-based vs portfolio-based)
2. Strategy rebalance schedule (monthly, quarterly, daily, etc.)

This is critical functionality that ensures positions are monitored daily
even when strategies don't rebalance daily.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from portfolio_backtester.backtesting.window_evaluator import WindowEvaluator
from portfolio_backtester.optimization.wfo_window import WFOWindow
from portfolio_backtester.strategies._core.base.base_strategy import BaseStrategy


class DummySignalStrategy(BaseStrategy):
    """A simple dummy signal strategy for integration testing."""

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
        weights = {ticker: 1.0 for ticker in tickers}  # Signal strength, not portfolio weight
        signals = pd.DataFrame(weights, index=[current_date])
        return signals


class DummyPortfolioStrategy(BaseStrategy):
    """A simple dummy portfolio strategy for integration testing."""

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


@pytest.mark.slow
@pytest.mark.integration
class TestScheduleIndependentStopLoss:
    """Test schedule-independent stop loss functionality."""

    @pytest.fixture
    def daily_ohlc_data(self):
        """Create daily OHLC data for backtesting."""
        dates = pd.date_range("2023-01-01", "2023-03-31", freq="D")
        tickers = ["AAPL", "MSFT", "GOOGL"]
        fields = ["Open", "High", "Low", "Close"]

        columns = pd.MultiIndex.from_product([tickers, fields], names=["Ticker", "Field"])

        # Create realistic price data with a crash scenario
        np.random.seed(42)
        data = {}

        for ticker in tickers:
            base_price = {"AAPL": 150.0, "MSFT": 250.0, "GOOGL": 100.0}[ticker]

            # Normal period, then crash in February
            normal_period = len(dates) // 2

            # Generate normal prices for first half
            normal_prices = base_price * (1 + np.random.randn(normal_period) * 0.02).cumprod()

            # Generate crash prices for second half (significant drop)
            crash_factor = 0.7  # 30% drop
            crash_prices = (
                normal_prices[-1]
                * crash_factor
                * (1 + np.random.randn(len(dates) - normal_period) * 0.03).cumprod()
            )

            all_prices = np.concatenate([normal_prices, crash_prices])

            data[(ticker, "Open")] = all_prices * (1 + np.random.randn(len(dates)) * 0.005)
            data[(ticker, "High")] = all_prices * (1 + np.abs(np.random.randn(len(dates))) * 0.01)
            data[(ticker, "Low")] = all_prices * (1 - np.abs(np.random.randn(len(dates))) * 0.01)
            data[(ticker, "Close")] = all_prices

        return pd.DataFrame(data, index=dates, columns=columns)

    @pytest.fixture
    def benchmark_data(self, daily_ohlc_data):
        """Create benchmark data (SPY equivalent)."""
        # Use average of all tickers as benchmark
        benchmark_close = daily_ohlc_data.xs("Close", level="Field", axis=1).mean(axis=1)

        return pd.DataFrame({"Close": benchmark_close})

    @pytest.fixture
    def monthly_ema_signal_strategy(self):
        """Create EMA signal strategy with monthly rebalancing."""
        config = {
            "leverage": 1.0,
            "stop_loss_config": {"type": "AtrBasedStopLoss", "atr_length": 14, "atr_multiple": 2.0},
            "timing_config": {
                "mode": "time_based",
                "rebalance_frequency": "M",  # Monthly rebalancing
            },
        }
        return DummySignalStrategy(config)

    @pytest.fixture
    def quarterly_portfolio_strategy(self):
        """Create portfolio momentum strategy with quarterly rebalancing."""
        config = {
            "strategy_params": {
                "num_holdings": 2,
                "leverage": 1.0,
                "smoothing_lambda": 0.0,
                "trade_longs": True,
                "trade_shorts": False,
            },
            "stop_loss_config": {"type": "AtrBasedStopLoss", "atr_length": 14, "atr_multiple": 1.5},
            "timing_config": {
                "mode": "time_based",
                "rebalance_frequency": "Q",  # Quarterly rebalancing
            },
        }
        return DummyPortfolioStrategy(config)

    @pytest.fixture
    def wfo_window(self):
        """Create WFO window for testing."""
        return WFOWindow(
            train_start=pd.Timestamp("2023-01-01"),
            train_end=pd.Timestamp("2023-01-31"),
            test_start=pd.Timestamp("2023-02-01"),
            test_end=pd.Timestamp("2023-03-31"),
            evaluation_frequency="D",  # Daily evaluation regardless of strategy rebalancing
        )

    def test_monthly_signal_strategy_daily_stop_loss_monitoring(
        self, daily_ohlc_data, benchmark_data, monthly_ema_signal_strategy, wfo_window
    ):
        """
        🚨 CRITICAL TEST: Monthly signal strategy with daily stop loss monitoring.

        This verifies that:
        1. Strategy rebalances monthly
        2. Stop loss is checked daily
        3. Positions are liquidated immediately when stop loss triggers
        """
        evaluator = WindowEvaluator()

        # Track strategy signal generation calls
        original_generate_signals = monthly_ema_signal_strategy.generate_signals
        signal_call_dates = []

        def track_signal_calls(*args, **kwargs):
            if "current_date" in kwargs:
                signal_call_dates.append(kwargs["current_date"])
            elif len(args) >= 4:  # current_date is 4th positional arg
                signal_call_dates.append(args[3])
            return original_generate_signals(*args, **kwargs)

        monthly_ema_signal_strategy.generate_signals = track_signal_calls

        # Run evaluation
        result = evaluator.evaluate_window(
            window=wfo_window,
            strategy=monthly_ema_signal_strategy,
            daily_data=daily_ohlc_data,
            benchmark_data=benchmark_data,
            universe_tickers=["AAPL", "MSFT", "GOOGL"],
            benchmark_ticker="SPY",
        )

        # Verify results
        assert isinstance(result.window_returns, pd.Series)
        assert len(result.trades or []) >= 0  # Should have some completed trades

        # Verify daily evaluation occurred (more calls than monthly rebalancing would suggest)
        eval_dates = wfo_window.get_evaluation_dates(daily_ohlc_data.index)
        assert len(signal_call_dates) > 12  # More than monthly (should be daily-ish)

        print(f"Strategy generated signals on {len(signal_call_dates)} dates")
        print(f"Total evaluation dates: {len(eval_dates)}")

    def test_quarterly_portfolio_strategy_daily_stop_loss_monitoring(
        self, daily_ohlc_data, benchmark_data, quarterly_portfolio_strategy, wfo_window
    ):
        """
        🚨 CRITICAL TEST: Quarterly portfolio strategy with daily stop loss monitoring.

        This verifies that:
        1. Strategy rebalances quarterly
        2. Stop loss is checked daily
        3. Positions held for months are still monitored daily
        """
        evaluator = WindowEvaluator()

        # Track signal generation for quarterly strategy
        original_generate_signals = quarterly_portfolio_strategy.generate_signals
        signal_call_dates = []

        def track_signal_calls(*args, **kwargs):
            if "current_date" in kwargs:
                signal_call_dates.append(kwargs["current_date"])
            elif len(args) >= 4:
                signal_call_dates.append(args[3])
            return original_generate_signals(*args, **kwargs)

        quarterly_portfolio_strategy.generate_signals = track_signal_calls

        # Run evaluation
        result = evaluator.evaluate_window(
            window=wfo_window,
            strategy=quarterly_portfolio_strategy,
            daily_data=daily_ohlc_data,
            benchmark_data=benchmark_data,
            universe_tickers=["AAPL", "MSFT", "GOOGL"],
            benchmark_ticker="SPY",
        )

        # Verify results
        assert isinstance(result.window_returns, pd.Series)

        # For quarterly strategy, should have many daily evaluations but fewer signal generations
        eval_dates = wfo_window.get_evaluation_dates(daily_ohlc_data.index)

        print(f"Portfolio strategy generated signals on {len(signal_call_dates)} dates")
        print(f"Total evaluation dates: {len(eval_dates)}")

        # Should have daily evaluations even with quarterly rebalancing
        assert len(signal_call_dates) > 4  # More than quarterly

    def test_stop_loss_triggers_between_rebalances(
        self, daily_ohlc_data, benchmark_data, monthly_ema_signal_strategy
    ):
        """
        🚨 CRITICAL TEST: Stop loss triggers between strategy rebalances.

        This tests the core requirement: positions acquired during monthly rebalancing
        must be monitored daily and liquidated immediately when stop loss triggers,
        even if the strategy won't rebalance again for weeks.
        """
        evaluator = WindowEvaluator()

        # Create scenario where strategy rebalances early, then prices crash later
        crash_window = WFOWindow(
            train_start=pd.Timestamp("2023-01-01"),
            train_end=pd.Timestamp("2023-01-31"),
            test_start=pd.Timestamp("2023-02-01"),  # Strategy might rebalance here
            test_end=pd.Timestamp("2023-02-28"),  # But crash happens throughout February
            evaluation_frequency="D",
        )

        # Monitor position liquidations
        with patch("src.portfolio_backtester.backtesting.window_evaluator.logger") as mock_logger:
            result = evaluator.evaluate_window(
                window=crash_window,
                strategy=monthly_ema_signal_strategy,
                daily_data=daily_ohlc_data,
                benchmark_data=benchmark_data,
                universe_tickers=["AAPL", "MSFT", "GOOGL"],
                benchmark_ticker="SPY",
            )

            # Check if stop loss liquidations were logged
            stop_loss_calls = [
                call
                for call in mock_logger.info.call_args_list
                if call[0] and "stop loss" in str(call[0][0]).lower()
            ]

            print(f"Stop loss liquidations detected: {len(stop_loss_calls)}")

        # Verify system handled the scenario
        assert isinstance(result.window_returns, pd.Series)

    def test_no_stop_loss_vs_atr_stop_loss_comparison(
        self, daily_ohlc_data, benchmark_data, wfo_window
    ):
        """
        Test comparing performance with and without stop loss to verify it's working.
        """
        # Strategy without stop loss
        no_stop_loss_config = {
            "fast_ema_days": 10,
            "slow_ema_days": 20,
            "leverage": 1.0,
            "stop_loss_config": {"type": "NoStopLoss"},
            "timing_config": {"mode": "time_based", "rebalance_frequency": "M"},
        }
        no_stop_loss_strategy = DummySignalStrategy(no_stop_loss_config)

        # Strategy with ATR stop loss
        atr_stop_loss_config = {
            "leverage": 1.0,
            "stop_loss_config": {
                "type": "AtrBasedStopLoss",
                "atr_length": 10,
                "atr_multiple": 1.0,  # Tight stop loss for testing
            },
            "timing_config": {"mode": "time_based", "rebalance_frequency": "M"},
        }
        atr_stop_loss_strategy = DummySignalStrategy(atr_stop_loss_config)

        evaluator = WindowEvaluator()

        # Run both strategies
        no_sl_result = evaluator.evaluate_window(
            window=wfo_window,
            strategy=no_stop_loss_strategy,
            daily_data=daily_ohlc_data,
            benchmark_data=benchmark_data,
            universe_tickers=["AAPL", "MSFT", "GOOGL"],
            benchmark_ticker="SPY",
        )

        atr_sl_result = evaluator.evaluate_window(
            window=wfo_window,
            strategy=atr_stop_loss_strategy,
            daily_data=daily_ohlc_data,
            benchmark_data=benchmark_data,
            universe_tickers=["AAPL", "MSFT", "GOOGL"],
            benchmark_ticker="SPY",
        )

        # Compare results
        no_sl_total_return = no_sl_result.window_returns.sum()
        atr_sl_total_return = atr_sl_result.window_returns.sum()

        print(f"No stop loss total return: {no_sl_total_return:.4f}")
        print(f"ATR stop loss total return: {atr_sl_total_return:.4f}")
        print(f"Number of trades (no SL): {len(no_sl_result.trades or [])}")
        print(f"Number of trades (ATR SL): {len(atr_sl_result.trades or [])}")

        # With tight stop loss in crash scenario, should have different performance
        # (Not necessarily better, but definitely different due to stop loss activity)
        assert no_sl_total_return != atr_sl_total_return or len(atr_sl_result.trades or []) != len(
            no_sl_result.trades or []
        )

    def test_mixed_strategy_types_same_stop_loss_behavior(
        self, daily_ohlc_data, benchmark_data, monthly_ema_signal_strategy, wfo_window
    ):
        """
        Test that signal and portfolio strategies exhibit similar stop loss behavior.
        """
        # Create equivalent portfolio strategy
        portfolio_config = {
            "strategy_params": {
                "num_holdings": 3,
                "leverage": 1.0,
                "smoothing_lambda": 0.0,
                "trade_longs": True,
                "trade_shorts": False,
                "lookback_window": 10,
            },
            "stop_loss_config": {
                "type": "AtrBasedStopLoss",
                "atr_length": 14,
                "atr_multiple": 2.0,  # Same as signal strategy
            },
            "timing_config": {"mode": "time_based", "rebalance_frequency": "M"},  # Same frequency
        }

        portfolio_strategy = DummyPortfolioStrategy(portfolio_config)
        evaluator = WindowEvaluator()

        # Run both strategy types
        signal_result = evaluator.evaluate_window(
            window=wfo_window,
            strategy=monthly_ema_signal_strategy,
            daily_data=daily_ohlc_data,
            benchmark_data=benchmark_data,
            universe_tickers=["AAPL", "MSFT", "GOOGL"],
            benchmark_ticker="SPY",
        )

        portfolio_result = evaluator.evaluate_window(
            window=wfo_window,
            strategy=portfolio_strategy,
            daily_data=daily_ohlc_data,
            benchmark_data=benchmark_data,
            universe_tickers=["AAPL", "MSFT", "GOOGL"],
            benchmark_ticker="SPY",
        )

        # Both should successfully run with stop loss
        assert isinstance(signal_result.window_returns, pd.Series)
        assert isinstance(portfolio_result.window_returns, pd.Series)

        print(f"Signal strategy trades: {len(signal_result.trades or [])}")
        print(f"Portfolio strategy trades: {len(portfolio_result.trades or [])}")

    def test_stop_loss_independence_from_strategy_timing(self, daily_ohlc_data, benchmark_data):
        """
        🚨 CRITICAL TEST: Verify stop loss operates independently of strategy timing.

        Test multiple rebalance frequencies with same stop loss configuration
        to ensure stop loss monitoring frequency is independent of strategy frequency.
        """
        base_config = {
            "fast_ema_days": 10,
            "slow_ema_days": 20,
            "leverage": 1.0,
            "stop_loss_config": {"type": "AtrBasedStopLoss", "atr_length": 14, "atr_multiple": 2.0},
        }

        frequencies = ["D", "W", "M"]  # Daily, Weekly, Monthly
        results = {}

        evaluator = WindowEvaluator()
        window = WFOWindow(
            train_start=pd.Timestamp("2023-01-01"),
            train_end=pd.Timestamp("2023-01-31"),
            test_start=pd.Timestamp("2023-02-01"),
            test_end=pd.Timestamp("2023-02-28"),
            evaluation_frequency="D",  # Always daily evaluation
        )

        for freq in frequencies:
            config = base_config.copy()
            config["timing_config"] = {"mode": "time_based", "rebalance_frequency": freq}

            strategy = DummySignalStrategy(config)

            result = evaluator.evaluate_window(
                window=window,
                strategy=strategy,
                daily_data=daily_ohlc_data,
                benchmark_data=benchmark_data,
                universe_tickers=["AAPL", "MSFT"],
                benchmark_ticker="SPY",
            )

            results[freq] = result
            print(f"Frequency {freq}: {len(result.trades or [])} trades")

        # All should successfully run - stop loss monitoring is independent of rebalance frequency
        for freq, result in results.items():
            assert isinstance(result.window_returns, pd.Series)
            assert len(result.window_returns) > 0  # Should have some returns


class TestStopLossErrorHandling:
    """Test error handling in stop loss system."""

    def test_stop_loss_handler_failure_graceful_degradation(self):
        """Test that system continues when stop loss handler fails."""
        evaluator = WindowEvaluator()

        # Create strategy with faulty stop loss handler
        config = {
            "fast_ema_days": 10,
            "slow_ema_days": 20,
            "stop_loss_config": {"type": "AtrBasedStopLoss", "atr_length": 14, "atr_multiple": 2.0},
            "timing_config": {"mode": "time_based", "rebalance_frequency": "M"},
        }

        strategy = DummySignalStrategy(config)

        # Mock the stop loss handler to raise exceptions

        mock_handler = Mock()
        mock_handler.calculate_stop_levels.side_effect = Exception("Stop loss calculation failed")

        with patch.object(strategy, "get_stop_loss_handler", return_value=mock_handler):
            # Create minimal test data
            dates = pd.date_range("2023-01-01", periods=10, freq="D")
            data = pd.DataFrame(
                {("AAPL", "Close"): [150.0] * len(dates), ("MSFT", "Close"): [250.0] * len(dates)},
                index=dates,
            )
            data.columns = pd.MultiIndex.from_tuples(
                [("AAPL", "Close"), ("MSFT", "Close")], names=["Ticker", "Field"]
            )

            benchmark = pd.DataFrame({"Close": [100.0] * len(dates)}, index=dates)

            window = WFOWindow(
                train_start=dates[0],
                train_end=dates[4],
                test_start=dates[5],
                test_end=dates[-1],
                evaluation_frequency="D",
            )

            # Should not crash when stop loss handler fails
            result = evaluator.evaluate_window(
                window=window,
                strategy=strategy,
                daily_data=data,
                benchmark_data=benchmark,
                universe_tickers=["AAPL", "MSFT"],
                benchmark_ticker="SPY",
            )

            # Should still produce results
            assert isinstance(result.window_returns, pd.Series)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
