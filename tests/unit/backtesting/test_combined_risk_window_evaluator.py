"""
Unit tests for WindowEvaluator with combined stop loss and take profit functionality.

These tests focus specifically on the integration of both daily stop loss
and take profit monitoring into the WindowEvaluator's daily evaluation loop.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch

from src.portfolio_backtester.backtesting.window_evaluator import WindowEvaluator
from src.portfolio_backtester.optimization.wfo_window import WFOWindow
from src.portfolio_backtester.risk_management.stop_loss_handlers import AtrBasedStopLoss
from src.portfolio_backtester.risk_management.take_profit_handlers import AtrBasedTakeProfit


class TestCombinedRiskWindowEvaluator:
    """Test WindowEvaluator with combined stop loss and take profit monitoring."""

    @pytest.fixture
    def evaluator(self):
        """Create WindowEvaluator instance."""
        return WindowEvaluator()

    @pytest.fixture
    def mock_strategy_combined_risk(self):
        """Create mock strategy with both stop loss and take profit handlers."""
        strategy = Mock()
        strategy.generate_signals.return_value = pd.DataFrame(
            {"AAPL": [0.6], "MSFT": [0.4]}, index=[pd.Timestamp("2023-01-15")]
        )

        # Mock stop loss handler
        stop_loss_handler = Mock(spec=AtrBasedStopLoss)
        stop_loss_handler.calculate_stop_levels.return_value = pd.Series(
            {"AAPL": 140.0, "MSFT": 240.0}
        )
        stop_loss_handler.apply_stop_loss.return_value = pd.Series(
            {"AAPL": 0.6, "MSFT": 0.4}  # No stop loss triggered
        )

        # Mock take profit handler
        take_profit_handler = Mock(spec=AtrBasedTakeProfit)
        take_profit_handler.calculate_take_profit_levels.return_value = pd.Series(
            {"AAPL": 180.0, "MSFT": 280.0}
        )
        take_profit_handler.apply_take_profit.return_value = pd.Series(
            {"AAPL": 0.0, "MSFT": 0.4}  # AAPL take profit triggered
        )

        strategy.get_stop_loss_handler.return_value = stop_loss_handler
        strategy.get_take_profit_handler.return_value = take_profit_handler
        return strategy

    @pytest.fixture
    def test_window(self):
        """Create test WFO window."""
        return WFOWindow(
            train_start=pd.Timestamp("2023-01-01"),
            train_end=pd.Timestamp("2023-01-10"),
            test_start=pd.Timestamp("2023-01-11"),
            test_end=pd.Timestamp("2023-01-15"),
            evaluation_frequency="D",
        )

    @pytest.fixture
    def test_data(self):
        """Create test daily data."""
        dates = pd.date_range("2023-01-01", "2023-01-15", freq="D")
        tickers = ["AAPL", "MSFT"]
        fields = ["Open", "High", "Low", "Close"]

        columns = pd.MultiIndex.from_product([tickers, fields], names=["Ticker", "Field"])

        data = {}
        for ticker in tickers:
            base_price = {"AAPL": 150.0, "MSFT": 250.0}[ticker]
            prices = [base_price] * len(dates)

            data[(ticker, "Open")] = prices
            data[(ticker, "High")] = prices
            data[(ticker, "Low")] = prices
            data[(ticker, "Close")] = prices

        return pd.DataFrame(data, index=dates, columns=columns)

    @pytest.fixture
    def benchmark_data(self):
        """Create benchmark data."""
        dates = pd.date_range("2023-01-01", "2023-01-15", freq="D")
        return pd.DataFrame({"Close": [100.0] * len(dates)}, index=dates)

    def test_both_monitors_initialized(
        self, evaluator, mock_strategy_combined_risk, test_window, test_data, benchmark_data
    ):
        """Test that both DailyStopLossMonitor and DailyTakeProfitMonitor are initialized."""
        with (
            patch(
                "src.portfolio_backtester.backtesting.window_evaluator.DailyStopLossMonitor"
            ) as mock_sl_monitor_class,
            patch(
                "src.portfolio_backtester.backtesting.window_evaluator.DailyTakeProfitMonitor"
            ) as mock_tp_monitor_class,
        ):

            mock_sl_monitor = Mock()
            mock_tp_monitor = Mock()
            mock_sl_monitor_class.return_value = mock_sl_monitor
            mock_tp_monitor_class.return_value = mock_tp_monitor

            mock_sl_monitor.check_positions_for_stop_loss.return_value = pd.DataFrame()
            mock_tp_monitor.check_positions_for_take_profit.return_value = pd.DataFrame()

            evaluator.evaluate_window(
                window=test_window,
                strategy=mock_strategy_combined_risk,
                daily_data=test_data,
                benchmark_data=benchmark_data,
                universe_tickers=["AAPL", "MSFT"],
                benchmark_ticker="SPY",
            )

            # Verify both monitors were created
            mock_sl_monitor_class.assert_called_once()
            mock_tp_monitor_class.assert_called_once()

    def test_both_monitors_called_daily(
        self, evaluator, mock_strategy_combined_risk, test_window, test_data, benchmark_data
    ):
        """Test that both monitors are called for each evaluation date."""
        with (
            patch(
                "src.portfolio_backtester.backtesting.window_evaluator.DailyStopLossMonitor"
            ) as mock_sl_monitor_class,
            patch(
                "src.portfolio_backtester.backtesting.window_evaluator.DailyTakeProfitMonitor"
            ) as mock_tp_monitor_class,
        ):

            mock_sl_monitor = Mock()
            mock_tp_monitor = Mock()
            mock_sl_monitor_class.return_value = mock_sl_monitor
            mock_tp_monitor_class.return_value = mock_tp_monitor

            mock_sl_monitor.check_positions_for_stop_loss.return_value = pd.DataFrame()
            mock_tp_monitor.check_positions_for_take_profit.return_value = pd.DataFrame()

            evaluator.evaluate_window(
                window=test_window,
                strategy=mock_strategy_combined_risk,
                daily_data=test_data,
                benchmark_data=benchmark_data,
                universe_tickers=["AAPL", "MSFT"],
                benchmark_ticker="SPY",
            )

            # Both monitors should be called for each evaluation date
            eval_dates = test_window.get_evaluation_dates(test_data.index)
            expected_calls = len(eval_dates)

            assert mock_sl_monitor.check_positions_for_stop_loss.call_count >= expected_calls
            assert mock_tp_monitor.check_positions_for_take_profit.call_count >= expected_calls

    def test_execution_order_stop_loss_then_take_profit(
        self, evaluator, mock_strategy_combined_risk, test_window, test_data, benchmark_data
    ):
        """Test that stop loss is checked before take profit (execution order matters)."""
        call_order = []

        def mock_stop_loss_check(*args, **kwargs):
            call_order.append("stop_loss")
            return pd.DataFrame()

        def mock_take_profit_check(*args, **kwargs):
            call_order.append("take_profit")
            return pd.DataFrame()

        with (
            patch(
                "src.portfolio_backtester.backtesting.window_evaluator.DailyStopLossMonitor"
            ) as mock_sl_monitor_class,
            patch(
                "src.portfolio_backtester.backtesting.window_evaluator.DailyTakeProfitMonitor"
            ) as mock_tp_monitor_class,
        ):

            mock_sl_monitor = Mock()
            mock_tp_monitor = Mock()
            mock_sl_monitor_class.return_value = mock_sl_monitor
            mock_tp_monitor_class.return_value = mock_tp_monitor

            mock_sl_monitor.check_positions_for_stop_loss.side_effect = mock_stop_loss_check
            mock_tp_monitor.check_positions_for_take_profit.side_effect = mock_take_profit_check

            evaluator.evaluate_window(
                window=test_window,
                strategy=mock_strategy_combined_risk,
                daily_data=test_data,
                benchmark_data=benchmark_data,
                universe_tickers=["AAPL", "MSFT"],
                benchmark_ticker="SPY",
            )

            # Verify execution order for each day
            eval_dates = test_window.get_evaluation_dates(test_data.index)
            expected_pattern = ["stop_loss", "take_profit"] * len(eval_dates)

            # Check that stop loss always comes before take profit
            assert len(call_order) >= len(expected_pattern)
            for i in range(0, len(call_order), 2):
                if i + 1 < len(call_order):
                    assert call_order[i] == "stop_loss"
                    assert call_order[i + 1] == "take_profit"

    def test_both_systems_trigger_liquidations(
        self, evaluator, mock_strategy_combined_risk, test_window, test_data, benchmark_data
    ):
        """Test that liquidations from both systems are applied to position tracker."""
        with (
            patch(
                "src.portfolio_backtester.backtesting.window_evaluator.DailyStopLossMonitor"
            ) as mock_sl_monitor_class,
            patch(
                "src.portfolio_backtester.backtesting.window_evaluator.DailyTakeProfitMonitor"
            ) as mock_tp_monitor_class,
            patch(
                "src.portfolio_backtester.backtesting.window_evaluator.PositionTracker"
            ) as mock_position_tracker_class,
        ):

            mock_sl_monitor = Mock()
            mock_tp_monitor = Mock()
            mock_position_tracker = Mock()

            mock_sl_monitor_class.return_value = mock_sl_monitor
            mock_tp_monitor_class.return_value = mock_tp_monitor
            mock_position_tracker_class.return_value = mock_position_tracker

            # Mock stop loss signals (liquidate MSFT)
            stop_loss_signals = pd.DataFrame({"MSFT": [0.0]}, index=[pd.Timestamp("2023-01-15")])
            mock_sl_monitor.check_positions_for_stop_loss.return_value = stop_loss_signals

            # Mock take profit signals (liquidate AAPL)
            take_profit_signals = pd.DataFrame({"AAPL": [0.0]}, index=[pd.Timestamp("2023-01-15")])
            mock_tp_monitor.check_positions_for_take_profit.return_value = take_profit_signals

            mock_position_tracker.update_positions.return_value = pd.Series(
                {"AAPL": 0.6, "MSFT": 0.4}
            )

            evaluator.evaluate_window(
                window=test_window,
                strategy=mock_strategy_combined_risk,
                daily_data=test_data,
                benchmark_data=benchmark_data,
                universe_tickers=["AAPL", "MSFT"],
                benchmark_ticker="SPY",
            )

            # Verify that position tracker was called to apply both types of signals
            # Should be called at least 3 times per day: strategy signals, stop loss signals, take profit signals
            update_calls = mock_position_tracker.update_positions.call_count
            eval_dates = test_window.get_evaluation_dates(test_data.index)
            expected_min_calls = len(eval_dates) * 3  # strategy + stop loss + take profit
            assert update_calls >= expected_min_calls

    def test_one_system_error_doesnt_affect_other(
        self, evaluator, mock_strategy_combined_risk, test_window, test_data, benchmark_data
    ):
        """Test that errors in one system don't prevent the other from working."""
        with (
            patch(
                "src.portfolio_backtester.backtesting.window_evaluator.DailyStopLossMonitor"
            ) as mock_sl_monitor_class,
            patch(
                "src.portfolio_backtester.backtesting.window_evaluator.DailyTakeProfitMonitor"
            ) as mock_tp_monitor_class,
        ):

            mock_sl_monitor = Mock()
            mock_tp_monitor = Mock()
            mock_sl_monitor_class.return_value = mock_sl_monitor
            mock_tp_monitor_class.return_value = mock_tp_monitor

            # Stop loss monitor raises exception
            mock_sl_monitor.check_positions_for_stop_loss.side_effect = Exception("Stop loss error")

            # Take profit monitor works normally
            mock_tp_monitor.check_positions_for_take_profit.return_value = pd.DataFrame(
                {"AAPL": [0.0]}, index=[pd.Timestamp("2023-01-15")]
            )

            # Should complete without crashing despite stop loss error
            result = evaluator.evaluate_window(
                window=test_window,
                strategy=mock_strategy_combined_risk,
                daily_data=test_data,
                benchmark_data=benchmark_data,
                universe_tickers=["AAPL", "MSFT"],
                benchmark_ticker="SPY",
            )

            # Should still produce results
            assert isinstance(result.window_returns, pd.Series)

            # Take profit monitor should still be called
            assert mock_tp_monitor.check_positions_for_take_profit.called

    def test_correct_handlers_passed_to_monitors(
        self, evaluator, mock_strategy_combined_risk, test_window, test_data, benchmark_data
    ):
        """Test that the correct handlers are passed to the respective monitors."""
        with (
            patch(
                "src.portfolio_backtester.backtesting.window_evaluator.DailyStopLossMonitor"
            ) as mock_sl_monitor_class,
            patch(
                "src.portfolio_backtester.backtesting.window_evaluator.DailyTakeProfitMonitor"
            ) as mock_tp_monitor_class,
        ):

            mock_sl_monitor = Mock()
            mock_tp_monitor = Mock()
            mock_sl_monitor_class.return_value = mock_sl_monitor
            mock_tp_monitor_class.return_value = mock_tp_monitor

            mock_sl_monitor.check_positions_for_stop_loss.return_value = pd.DataFrame()
            mock_tp_monitor.check_positions_for_take_profit.return_value = pd.DataFrame()

            evaluator.evaluate_window(
                window=test_window,
                strategy=mock_strategy_combined_risk,
                daily_data=test_data,
                benchmark_data=benchmark_data,
                universe_tickers=["AAPL", "MSFT"],
                benchmark_ticker="SPY",
            )

            # Verify that the correct handlers were passed to the monitors
            sl_calls = mock_sl_monitor.check_positions_for_stop_loss.call_args_list
            tp_calls = mock_tp_monitor.check_positions_for_take_profit.call_args_list

            if sl_calls:
                # Check that stop loss handler was passed to stop loss monitor
                assert "stop_loss_handler" in sl_calls[0].kwargs
                assert (
                    sl_calls[0].kwargs["stop_loss_handler"]
                    == mock_strategy_combined_risk.get_stop_loss_handler.return_value
                )

            if tp_calls:
                # Check that take profit handler was passed to take profit monitor
                assert "take_profit_handler" in tp_calls[0].kwargs
                assert (
                    tp_calls[0].kwargs["take_profit_handler"]
                    == mock_strategy_combined_risk.get_take_profit_handler.return_value
                )

    def test_logging_both_systems_liquidations(
        self, evaluator, mock_strategy_combined_risk, test_window, test_data, benchmark_data
    ):
        """Test that liquidations from both systems are logged appropriately."""
        with (
            patch(
                "src.portfolio_backtester.backtesting.window_evaluator.DailyStopLossMonitor"
            ) as mock_sl_monitor_class,
            patch(
                "src.portfolio_backtester.backtesting.window_evaluator.DailyTakeProfitMonitor"
            ) as mock_tp_monitor_class,
            patch("src.portfolio_backtester.backtesting.window_evaluator.logger") as mock_logger,
        ):

            mock_sl_monitor = Mock()
            mock_tp_monitor = Mock()
            mock_sl_monitor_class.return_value = mock_sl_monitor
            mock_tp_monitor_class.return_value = mock_tp_monitor

            # Both systems trigger liquidations
            stop_loss_signals = pd.DataFrame({"MSFT": [0.0]}, index=[pd.Timestamp("2023-01-15")])
            take_profit_signals = pd.DataFrame({"AAPL": [0.0]}, index=[pd.Timestamp("2023-01-15")])

            mock_sl_monitor.check_positions_for_stop_loss.return_value = stop_loss_signals
            mock_tp_monitor.check_positions_for_take_profit.return_value = take_profit_signals

            evaluator.evaluate_window(
                window=test_window,
                strategy=mock_strategy_combined_risk,
                daily_data=test_data,
                benchmark_data=benchmark_data,
                universe_tickers=["AAPL", "MSFT"],
                benchmark_ticker="SPY",
            )

            # Check that both types of liquidations were logged
            logged_messages = [call[0][0] for call in mock_logger.info.call_args_list]

            stop_loss_logged = any(
                "stop loss liquidations" in msg.lower() for msg in logged_messages
            )
            take_profit_logged = any(
                "take profit liquidations" in msg.lower() for msg in logged_messages
            )

            assert stop_loss_logged, "Stop loss liquidations should be logged"
            assert take_profit_logged, "Take profit liquidations should be logged"


if __name__ == "__main__":
    pytest.main([__file__])
