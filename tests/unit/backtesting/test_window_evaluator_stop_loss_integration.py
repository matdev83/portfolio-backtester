"""
Unit tests for WindowEvaluator stop loss integration.

These tests focus specifically on the integration of daily stop loss monitoring
into the WindowEvaluator's daily evaluation loop.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch

from portfolio_backtester.backtesting.window_evaluator import WindowEvaluator
from portfolio_backtester.optimization.wfo_window import WFOWindow
from portfolio_backtester.risk_management.stop_loss_handlers import AtrBasedStopLoss


class TestWindowEvaluatorStopLossIntegration:
    """Test WindowEvaluator integration with daily stop loss monitoring."""

    @pytest.fixture
    def evaluator(self):
        """Create WindowEvaluator instance."""
        return WindowEvaluator()

    @pytest.fixture
    def mock_strategy(self):
        """Create mock strategy with stop loss handler."""
        strategy = Mock()
        strategy.generate_signals.return_value = pd.DataFrame(
            {"AAPL": [0.5], "MSFT": [0.3], "GOOGL": [0.2]}, index=[pd.Timestamp("2023-01-15")]
        )

        # Mock stop loss handler
        stop_loss_handler = Mock(spec=AtrBasedStopLoss)
        stop_loss_handler.calculate_stop_levels.return_value = pd.Series(
            {"AAPL": 140.0, "MSFT": 240.0, "GOOGL": 95.0}
        )
        stop_loss_handler.apply_stop_loss.return_value = pd.Series(
            {
                "AAPL": 0.0,  # Stopped out
                "MSFT": 0.3,  # Not stopped out
                "GOOGL": 0.2,  # Not stopped out
            }
        )

        strategy.get_stop_loss_handler.return_value = stop_loss_handler
        return strategy

    @pytest.fixture
    def test_window(self):
        """Create test WFO window."""
        return WFOWindow(
            train_start=pd.Timestamp("2023-01-01"),
            train_end=pd.Timestamp("2023-01-10"),
            test_start=pd.Timestamp("2023-01-11"),
            test_end=pd.Timestamp("2023-01-20"),
            evaluation_frequency="D",
        )

    @pytest.fixture
    def test_data(self):
        """Create test daily data."""
        dates = pd.date_range("2023-01-01", "2023-01-20", freq="D")
        tickers = ["AAPL", "MSFT", "GOOGL"]
        fields = ["Open", "High", "Low", "Close"]

        columns = pd.MultiIndex.from_product([tickers, fields], names=["Ticker", "Field"])

        data = {}
        for ticker in tickers:
            base_price = {"AAPL": 150.0, "MSFT": 250.0, "GOOGL": 100.0}[ticker]
            prices = [base_price] * len(dates)

            data[(ticker, "Open")] = prices
            data[(ticker, "High")] = prices
            data[(ticker, "Low")] = prices
            data[(ticker, "Close")] = prices

        return pd.DataFrame(data, index=dates, columns=columns)

    @pytest.fixture
    def benchmark_data(self):
        """Create benchmark data."""
        dates = pd.date_range("2023-01-01", "2023-01-20", freq="D")
        return pd.DataFrame({"Close": [100.0] * len(dates)}, index=dates)

    def test_daily_stop_loss_monitor_initialization(
        self, evaluator, mock_strategy, test_window, test_data, benchmark_data
    ):
        """Test that DailyStopLossMonitor is properly initialized."""
        with patch(
            "portfolio_backtester.backtesting.window_evaluator.DailyStopLossMonitor"
        ) as mock_monitor_class:
            mock_monitor_instance = Mock()
            mock_monitor_class.return_value = mock_monitor_instance
            mock_monitor_instance.check_positions_for_stop_loss.return_value = pd.DataFrame()

            evaluator.evaluate_window(
                window=test_window,
                strategy=mock_strategy,
                daily_data=test_data,
                benchmark_data=benchmark_data,
                universe_tickers=["AAPL", "MSFT", "GOOGL"],
                benchmark_ticker="SPY",
            )

            # Verify monitor was created
            mock_monitor_class.assert_called_once()

    def test_daily_stop_loss_check_called_every_day(
        self, evaluator, mock_strategy, test_window, test_data, benchmark_data
    ):
        """Test that stop loss check is called on every evaluation date."""
        with patch(
            "portfolio_backtester.backtesting.window_evaluator.DailyStopLossMonitor"
        ) as mock_monitor_class:
            mock_monitor_instance = Mock()
            mock_monitor_class.return_value = mock_monitor_instance
            mock_monitor_instance.check_positions_for_stop_loss.return_value = pd.DataFrame()

            evaluator.evaluate_window(
                window=test_window,
                strategy=mock_strategy,
                daily_data=test_data,
                benchmark_data=benchmark_data,
                universe_tickers=["AAPL", "MSFT", "GOOGL"],
                benchmark_ticker="SPY",
            )

            # Get evaluation dates
            eval_dates = test_window.get_evaluation_dates(test_data.index)

            # Verify stop loss check was called for each evaluation date
            assert mock_monitor_instance.check_positions_for_stop_loss.call_count == len(eval_dates)

    def test_stop_loss_signals_applied_to_position_tracker(
        self, evaluator, mock_strategy, test_window, test_data, benchmark_data
    ):
        """Test that stop loss liquidation signals are applied to position tracker."""
        # Create liquidation signals
        liquidation_signals = pd.DataFrame(
            {"AAPL": [0.0], "MSFT": [0.0]},  # Liquidate AAPL position  # Liquidate MSFT position
            index=[pd.Timestamp("2023-01-15")],
        )

        with patch(
            "portfolio_backtester.backtesting.window_evaluator.DailyStopLossMonitor"
        ) as mock_monitor_class:
            mock_monitor_instance = Mock()
            mock_monitor_class.return_value = mock_monitor_instance
            mock_monitor_instance.check_positions_for_stop_loss.return_value = liquidation_signals

            with patch(
                "portfolio_backtester.backtesting.window_evaluator.PositionTracker"
            ) as mock_tracker_class:
                mock_tracker_instance = Mock()
                mock_tracker_class.return_value = mock_tracker_instance
                mock_tracker_instance.update_positions.return_value = pd.Series(
                    {"AAPL": 0.0, "MSFT": 0.0, "GOOGL": 0.2}
                )
                mock_tracker_instance.get_completed_trades.return_value = []
                mock_tracker_instance.get_current_weights.return_value = pd.Series({"GOOGL": 0.2})

                evaluator.evaluate_window(
                    window=test_window,
                    strategy=mock_strategy,
                    daily_data=test_data,
                    benchmark_data=benchmark_data,
                    universe_tickers=["AAPL", "MSFT", "GOOGL"],
                    benchmark_ticker="SPY",
                )

                # Verify position tracker was called to apply liquidation signals
                # Should be called twice per evaluation: once for strategy signals, once for stop loss signals
                call_count = mock_tracker_instance.update_positions.call_count
                eval_dates = test_window.get_evaluation_dates(test_data.index)

                # Should have at least as many calls as evaluation dates (strategy signals)
                # Plus additional calls for stop loss signals
                assert call_count >= len(eval_dates)

    def test_stop_loss_error_handling_continues_evaluation(
        self, evaluator, mock_strategy, test_window, test_data, benchmark_data
    ):
        """Test that evaluation continues when stop loss monitoring fails."""
        with patch(
            "portfolio_backtester.backtesting.window_evaluator.DailyStopLossMonitor"
        ) as mock_monitor_class:
            mock_monitor_instance = Mock()
            mock_monitor_class.return_value = mock_monitor_instance
            mock_monitor_instance.check_positions_for_stop_loss.side_effect = Exception(
                "Stop loss check failed"
            )

            # Should not raise exception
            result = evaluator.evaluate_window(
                window=test_window,
                strategy=mock_strategy,
                daily_data=test_data,
                benchmark_data=benchmark_data,
                universe_tickers=["AAPL", "MSFT", "GOOGL"],
                benchmark_ticker="SPY",
            )

            # Should still produce valid results
            assert isinstance(result.window_returns, pd.Series)

    def test_stop_loss_parameters_passed_correctly(
        self, evaluator, mock_strategy, test_window, test_data, benchmark_data
    ):
        """Test that correct parameters are passed to stop loss monitor."""
        with patch(
            "portfolio_backtester.backtesting.window_evaluator.DailyStopLossMonitor"
        ) as mock_monitor_class:
            mock_monitor_instance = Mock()
            mock_monitor_class.return_value = mock_monitor_instance
            mock_monitor_instance.check_positions_for_stop_loss.return_value = pd.DataFrame()

            evaluator.evaluate_window(
                window=test_window,
                strategy=mock_strategy,
                daily_data=test_data,
                benchmark_data=benchmark_data,
                universe_tickers=["AAPL", "MSFT", "GOOGL"],
                benchmark_ticker="SPY",
            )

            # Verify parameters passed to stop loss check
            calls = mock_monitor_instance.check_positions_for_stop_loss.call_args_list

            for call in calls:
                args, kwargs = call

                # Check that required parameters are present
                assert "current_date" in kwargs
                assert "position_tracker" in kwargs
                assert "current_prices" in kwargs
                assert "stop_loss_handler" in kwargs
                assert "historical_data" in kwargs

                # Verify types
                assert isinstance(kwargs["current_date"], pd.Timestamp)
                assert isinstance(kwargs["current_prices"], pd.Series)
                assert isinstance(kwargs["historical_data"], pd.DataFrame)

    def test_stop_loss_liquidation_logging(
        self, evaluator, mock_strategy, test_window, test_data, benchmark_data
    ):
        """Test that stop loss liquidations are properly logged."""
        # Create liquidation signals
        liquidation_signals = pd.DataFrame({"AAPL": [0.0]}, index=[pd.Timestamp("2023-01-15")])

        with patch(
            "portfolio_backtester.backtesting.window_evaluator.DailyStopLossMonitor"
        ) as mock_monitor_class:
            mock_monitor_instance = Mock()
            mock_monitor_class.return_value = mock_monitor_instance
            mock_monitor_instance.check_positions_for_stop_loss.return_value = liquidation_signals

            with patch("portfolio_backtester.backtesting.window_evaluator.logger") as mock_logger:
                evaluator.evaluate_window(
                    window=test_window,
                    strategy=mock_strategy,
                    daily_data=test_data,
                    benchmark_data=benchmark_data,
                    universe_tickers=["AAPL", "MSFT", "GOOGL"],
                    benchmark_ticker="SPY",
                )

                # Verify liquidation was logged
                info_calls = mock_logger.info.call_args_list
                liquidation_logged = any(
                    "stop loss liquidations" in str(call).lower() for call in info_calls
                )
                assert liquidation_logged

    def test_empty_stop_loss_signals_handled_correctly(
        self, evaluator, mock_strategy, test_window, test_data, benchmark_data
    ):
        """Test handling when no stop loss signals are generated."""
        with patch(
            "portfolio_backtester.backtesting.window_evaluator.DailyStopLossMonitor"
        ) as mock_monitor_class:
            mock_monitor_instance = Mock()
            mock_monitor_class.return_value = mock_monitor_instance
            mock_monitor_instance.check_positions_for_stop_loss.return_value = (
                pd.DataFrame()
            )  # Empty signals

            with patch(
                "portfolio_backtester.backtesting.window_evaluator.PositionTracker"
            ) as mock_tracker_class:
                mock_tracker_instance = Mock()
                mock_tracker_class.return_value = mock_tracker_instance
                mock_tracker_instance.update_positions.return_value = pd.Series(
                    {"AAPL": 0.5, "MSFT": 0.3}
                )
                mock_tracker_instance.get_completed_trades.return_value = []
                mock_tracker_instance.get_current_weights.return_value = pd.Series(
                    {"AAPL": 0.5, "MSFT": 0.3}
                )

                result = evaluator.evaluate_window(
                    window=test_window,
                    strategy=mock_strategy,
                    daily_data=test_data,
                    benchmark_data=benchmark_data,
                    universe_tickers=["AAPL", "MSFT", "GOOGL"],
                    benchmark_ticker="SPY",
                )

                # Should handle empty signals gracefully
                assert isinstance(result.window_returns, pd.Series)

    def test_multiple_stop_loss_signals_same_day(
        self, evaluator, mock_strategy, test_window, test_data, benchmark_data
    ):
        """Test handling multiple stop loss signals on the same day."""
        # Multiple assets liquidated on same day
        liquidation_signals = pd.DataFrame(
            {"AAPL": [0.0], "MSFT": [0.0], "GOOGL": [0.0]}, index=[pd.Timestamp("2023-01-15")]
        )

        with patch(
            "portfolio_backtester.backtesting.window_evaluator.DailyStopLossMonitor"
        ) as mock_monitor_class:
            mock_monitor_instance = Mock()
            mock_monitor_class.return_value = mock_monitor_instance
            mock_monitor_instance.check_positions_for_stop_loss.return_value = liquidation_signals

            result = evaluator.evaluate_window(
                window=test_window,
                strategy=mock_strategy,
                daily_data=test_data,
                benchmark_data=benchmark_data,
                universe_tickers=["AAPL", "MSFT", "GOOGL"],
                benchmark_ticker="SPY",
            )

            # Should handle multiple liquidations gracefully
            assert isinstance(result.window_returns, pd.Series)

    def test_stop_loss_integration_preserves_existing_functionality(self, evaluator):
        """Test that adding stop loss doesn't break existing WindowEvaluator functionality."""
        # Create minimal viable test scenario
        strategy = Mock()
        strategy.generate_signals.return_value = pd.DataFrame(
            {"AAPL": [1.0]}, index=[pd.Timestamp("2023-01-15")]
        )

        stop_loss_handler = Mock()
        stop_loss_handler.calculate_stop_levels.return_value = pd.Series({"AAPL": 140.0})
        stop_loss_handler.apply_stop_loss.return_value = pd.Series({"AAPL": 1.0})  # No liquidation
        strategy.get_stop_loss_handler.return_value = stop_loss_handler

        window = WFOWindow(
            train_start=pd.Timestamp("2023-01-01"),
            train_end=pd.Timestamp("2023-01-10"),
            test_start=pd.Timestamp("2023-01-11"),
            test_end=pd.Timestamp("2023-01-16"),
            evaluation_frequency="D",
        )

        dates = pd.date_range("2023-01-01", "2023-01-16", freq="D")
        data = pd.DataFrame({("AAPL", "Close"): [150.0] * len(dates)}, index=dates)
        data.columns = pd.MultiIndex.from_tuples([("AAPL", "Close")], names=["Ticker", "Field"])

        benchmark = pd.DataFrame({"Close": [100.0] * len(dates)}, index=dates)

        result = evaluator.evaluate_window(
            window=window,
            strategy=strategy,
            daily_data=data,
            benchmark_data=benchmark,
            universe_tickers=["AAPL"],
            benchmark_ticker="SPY",
        )

        # Should produce valid WindowResult
        assert hasattr(result, "window_returns")
        assert hasattr(result, "metrics")
        assert hasattr(result, "trades")
        assert hasattr(result, "final_weights")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
