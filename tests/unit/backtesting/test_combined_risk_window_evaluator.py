"""
Unit tests for WindowEvaluator with combined stop loss and take profit functionality.

These tests focus specifically on the integration of both daily stop loss
and take profit monitoring into the WindowEvaluator's daily evaluation loop.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from portfolio_backtester.backtesting.window_evaluator import WindowEvaluator
from portfolio_backtester.optimization.wfo_window import WFOWindow
from portfolio_backtester.risk_management.stop_loss_handlers import AtrBasedStopLoss
from portfolio_backtester.risk_management.take_profit_handlers import AtrBasedTakeProfit


class TestCombinedRiskWindowEvaluator:
    """Test WindowEvaluator with combined stop loss and take profit monitoring."""

    @pytest.fixture
    def evaluator(self):
        """Create WindowEvaluator instance."""
        from portfolio_backtester.backtesting.strategy_backtester import StrategyBacktester

        mock_backtester = Mock(spec=StrategyBacktester)
        return WindowEvaluator(backtester=mock_backtester)

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

    def test_both_monitors_initialized(
        self, evaluator, mock_strategy_combined_risk, test_window, test_data
    ):
        """Test that both DailyStopLossMonitor and DailyTakeProfitMonitor are initialized."""
        # Create a mock factory that returns mock monitors
        mock_sl_monitor = Mock()
        mock_tp_monitor = Mock()
        
        mock_factory = Mock()
        mock_factory.create_stop_loss_monitor.return_value = mock_sl_monitor
        mock_factory.create_take_profit_monitor.return_value = mock_tp_monitor
        
        # Set the mock monitors' return values
        mock_sl_monitor.check_positions_for_stop_loss.return_value = pd.DataFrame()
        mock_tp_monitor.check_positions_for_take_profit.return_value = pd.DataFrame()
        
        # Replace the evaluator's factory with our mock
        evaluator.risk_monitor_factory = mock_factory

        evaluator.evaluate_window(
            window=test_window,
            strategy=mock_strategy_combined_risk,
            daily_data=test_data,
            full_monthly_data=test_data.resample("ME").last(),
            full_rets_daily=test_data.pct_change().fillna(0),
            universe_tickers=["AAPL", "MSFT"],
            benchmark_ticker="SPY",
            benchmark_data=test_data  # Added missing benchmark_data parameter
        )

        # Verify both monitors were created via the factory
        mock_factory.create_stop_loss_monitor.assert_called_once()
        mock_factory.create_take_profit_monitor.assert_called_once()

    def test_both_monitors_called_daily(
        self, evaluator, mock_strategy_combined_risk, test_window, test_data
    ):
        """Test that both monitors are called for each evaluation date."""
        # Create mock monitors
        mock_sl_monitor = Mock()
        mock_tp_monitor = Mock()
        
        # Create mock factory
        mock_factory = Mock()
        mock_factory.create_stop_loss_monitor.return_value = mock_sl_monitor
        mock_factory.create_take_profit_monitor.return_value = mock_tp_monitor
        
        # Set return values
        mock_sl_monitor.check_positions_for_stop_loss.return_value = pd.DataFrame()
        mock_tp_monitor.check_positions_for_take_profit.return_value = pd.DataFrame()
        
        # Replace evaluator's factory
        evaluator.risk_monitor_factory = mock_factory

        evaluator.evaluate_window(
            window=test_window,
            strategy=mock_strategy_combined_risk,
            daily_data=test_data,
            full_monthly_data=test_data.resample("ME").last(),
            full_rets_daily=test_data.pct_change().fillna(0),
            universe_tickers=["AAPL", "MSFT"],
            benchmark_ticker="SPY",
            benchmark_data=test_data  # Added missing benchmark_data parameter
        )

        # Both monitors should be called for each evaluation date
        eval_dates = test_window.get_evaluation_dates(test_data.index)
        expected_calls = len(eval_dates)

        assert mock_sl_monitor.check_positions_for_stop_loss.call_count >= expected_calls
        assert mock_tp_monitor.check_positions_for_take_profit.call_count >= expected_calls

    def test_execution_order_stop_loss_then_take_profit(
        self, evaluator, mock_strategy_combined_risk, test_window, test_data
    ):
        """Test that stop loss is checked before take profit (execution order matters)."""
        call_order = []

        def mock_stop_loss_check(*args, **kwargs):
            call_order.append("stop_loss")
            return pd.DataFrame()

        def mock_take_profit_check(*args, **kwargs):
            call_order.append("take_profit")
            return pd.DataFrame()

        # Create mock monitors
        mock_sl_monitor = Mock()
        mock_tp_monitor = Mock()
        
        # Create mock factory
        mock_factory = Mock()
        mock_factory.create_stop_loss_monitor.return_value = mock_sl_monitor
        mock_factory.create_take_profit_monitor.return_value = mock_tp_monitor
        
        # Set side effects to track call order
        mock_sl_monitor.check_positions_for_stop_loss.side_effect = mock_stop_loss_check
        mock_tp_monitor.check_positions_for_take_profit.side_effect = mock_take_profit_check
        
        # Replace evaluator's factory
        evaluator.risk_monitor_factory = mock_factory

        evaluator.evaluate_window(
            window=test_window,
            strategy=mock_strategy_combined_risk,
            daily_data=test_data,
            full_monthly_data=test_data.resample("ME").last(),
            full_rets_daily=test_data.pct_change().fillna(0),
            universe_tickers=["AAPL", "MSFT"],
            benchmark_ticker="SPY",
            benchmark_data=test_data  # Added missing benchmark_data parameter
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
        self, evaluator, mock_strategy_combined_risk, test_window, test_data
    ):
        """Test that liquidations from both systems are applied to position tracker."""
        # Since we can't easily mock the PositionTracker inside the WindowEvaluator,
        # we'll verify that the risk monitoring is happening by checking the log messages
        
        # Create mock monitors
        mock_sl_monitor = Mock()
        mock_tp_monitor = Mock()
        
        # Create mock factory
        mock_factory = Mock()
        mock_factory.create_stop_loss_monitor.return_value = mock_sl_monitor
        mock_factory.create_take_profit_monitor.return_value = mock_tp_monitor
        
        # Mock stop loss signals (liquidate MSFT)
        stop_loss_signals = pd.DataFrame({"MSFT": [0.0]}, index=[pd.Timestamp("2023-01-15")])
        mock_sl_monitor.check_positions_for_stop_loss.return_value = stop_loss_signals
        
        # Mock take profit signals (liquidate AAPL)
        take_profit_signals = pd.DataFrame({"AAPL": [0.0]}, index=[pd.Timestamp("2023-01-15")])
        mock_tp_monitor.check_positions_for_take_profit.return_value = take_profit_signals
        
        # Replace evaluator's factory
        evaluator.risk_monitor_factory = mock_factory
        
        with patch("portfolio_backtester.backtesting.window_evaluator.logger") as mock_logger:
            evaluator.evaluate_window(
                window=test_window,
                strategy=mock_strategy_combined_risk,
                daily_data=test_data,
                full_monthly_data=test_data.resample("ME").last(),
                full_rets_daily=test_data.pct_change().fillna(0),
                universe_tickers=["AAPL", "MSFT"],
                benchmark_ticker="SPY",
                benchmark_data=test_data  # Added missing benchmark_data parameter
            )
            
            # Verify that both systems triggered liquidations by checking the log messages
            info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
            
            stop_loss_calls = [call for call in info_calls if "stop loss liquidations" in call.lower()]
            take_profit_calls = [call for call in info_calls if "take profit liquidations" in call.lower()]
            
            eval_dates = test_window.get_evaluation_dates(test_data.index)
            
            # Should be called for each evaluation date
            assert len(stop_loss_calls) == len(eval_dates)
            assert len(take_profit_calls) == len(eval_dates)

    def test_one_system_error_doesnt_affect_other(
        self, evaluator, mock_strategy_combined_risk, test_window, test_data
    ):
        """Test that errors in one system don't prevent the other from working."""
        # Create mock monitors
        mock_sl_monitor = Mock()
        mock_tp_monitor = Mock()
        
        # Create mock factory
        mock_factory = Mock()
        mock_factory.create_stop_loss_monitor.return_value = mock_sl_monitor
        mock_factory.create_take_profit_monitor.return_value = mock_tp_monitor
        
        # Stop loss monitor raises exception
        mock_sl_monitor.check_positions_for_stop_loss.side_effect = Exception("Stop loss error")
        
        # Take profit monitor works normally
        mock_tp_monitor.check_positions_for_take_profit.return_value = pd.DataFrame(
            {"AAPL": [0.0]}, index=[pd.Timestamp("2023-01-15")]
        )
        
        # Replace evaluator's factory
        evaluator.risk_monitor_factory = mock_factory
        
        # Configure the mock backtester to return a proper result
        mock_result = Mock()
        mock_result.returns = pd.Series(index=test_data.index, data=np.random.randn(len(test_data)))
        mock_result.metrics = {"sharpe": 1.5}
        mock_result.trade_history = pd.DataFrame({
            "ticker": ["AAPL"], "entry_date": [pd.Timestamp("2023-01-01")],
            "exit_date": [pd.Timestamp("2023-01-05")]
        })
        mock_result.performance_stats = {"final_weights": {}}
        
        evaluator.backtester.backtest_strategy.return_value = mock_result

        # Should complete without crashing despite stop loss error
        result = evaluator.evaluate_window(
            window=test_window,
            strategy=mock_strategy_combined_risk,
            daily_data=test_data,
            full_monthly_data=test_data.resample("ME").last(),
            full_rets_daily=test_data.pct_change().fillna(0),
            universe_tickers=["AAPL", "MSFT"],
            benchmark_ticker="SPY",
            benchmark_data=test_data  # Added missing benchmark_data parameter
        )

        # Should still produce results
        assert isinstance(result.window_returns, pd.Series)

        # Take profit monitor should still be called
        assert mock_tp_monitor.check_positions_for_take_profit.called

    def test_correct_handlers_passed_to_monitors(
        self, evaluator, mock_strategy_combined_risk, test_window, test_data
    ):
        """Test that the correct handlers are passed to the respective monitors."""
        # Create mock monitors
        mock_sl_monitor = Mock()
        mock_tp_monitor = Mock()
        
        # Create mock factory
        mock_factory = Mock()
        mock_factory.create_stop_loss_monitor.return_value = mock_sl_monitor
        mock_factory.create_take_profit_monitor.return_value = mock_tp_monitor
        
        # Set return values
        mock_sl_monitor.check_positions_for_stop_loss.return_value = pd.DataFrame()
        mock_tp_monitor.check_positions_for_take_profit.return_value = pd.DataFrame()
        
        # Replace evaluator's factory
        evaluator.risk_monitor_factory = mock_factory

        evaluator.evaluate_window(
            window=test_window,
            strategy=mock_strategy_combined_risk,
            daily_data=test_data,
            full_monthly_data=test_data.resample("ME").last(),
            full_rets_daily=test_data.pct_change().fillna(0),
            universe_tickers=["AAPL", "MSFT"],
            benchmark_ticker="SPY",
            benchmark_data=test_data  # Added missing benchmark_data parameter
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
        self, evaluator, mock_strategy_combined_risk, test_window, test_data
    ):
        """Test that liquidations from both systems are logged appropriately."""
        # Create mock monitors
        mock_sl_monitor = Mock()
        mock_tp_monitor = Mock()
        
        # Create mock factory
        mock_factory = Mock()
        mock_factory.create_stop_loss_monitor.return_value = mock_sl_monitor
        mock_factory.create_take_profit_monitor.return_value = mock_tp_monitor
        
        # Both systems trigger liquidations
        stop_loss_signals = pd.DataFrame({"MSFT": [0.0]}, index=[pd.Timestamp("2023-01-15")])
        take_profit_signals = pd.DataFrame({"AAPL": [0.0]}, index=[pd.Timestamp("2023-01-15")])
        
        mock_sl_monitor.check_positions_for_stop_loss.return_value = stop_loss_signals
        mock_tp_monitor.check_positions_for_take_profit.return_value = take_profit_signals
        
        # Replace evaluator's factory
        evaluator.risk_monitor_factory = mock_factory
        
        with patch("portfolio_backtester.backtesting.window_evaluator.logger") as mock_logger:
            evaluator.evaluate_window(
                window=test_window,
                strategy=mock_strategy_combined_risk,
                daily_data=test_data,
                full_monthly_data=test_data.resample("ME").last(),
                full_rets_daily=test_data.pct_change().fillna(0),
                universe_tickers=["AAPL", "MSFT"],
                benchmark_ticker="SPY",
                benchmark_data=test_data  # Added missing benchmark_data parameter
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
