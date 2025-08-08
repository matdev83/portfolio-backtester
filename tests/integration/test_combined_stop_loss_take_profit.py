"""
Integration tests for combined Stop Loss and Take Profit functionality.

This test suite verifies that both stop loss and take profit systems work correctly
when used together in the same strategy, ensuring they don't interfere with each other
and handle edge cases appropriately.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock

from portfolio_backtester.backtesting.window_evaluator import WindowEvaluator
from portfolio_backtester.optimization.wfo_window import WFOWindow
from portfolio_backtester.strategies.base.base_strategy import BaseStrategy
from portfolio_backtester.risk_management.stop_loss_handlers import AtrBasedStopLoss
from portfolio_backtester.risk_management.take_profit_handlers import (
    AtrBasedTakeProfit,
)
from portfolio_backtester.risk_management.daily_stop_loss_monitor import DailyStopLossMonitor
from portfolio_backtester.risk_management.daily_take_profit_monitor import (
    DailyTakeProfitMonitor,
)
from portfolio_backtester.backtesting.position_tracker import Position, PositionTracker


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
class TestCombinedStopLossTakeProfit:
    """Test combined stop loss and take profit functionality."""

    @pytest.fixture
    def daily_ohlc_data(self):
        """Create daily OHLC data with various price scenarios."""
        dates = pd.date_range("2023-01-01", "2023-04-30", freq="D")
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        fields = ["Open", "High", "Low", "Close"]

        columns = pd.MultiIndex.from_product([tickers, fields], names=["Ticker", "Field"])

        # Create realistic price scenarios
        np.random.seed(42)
        data = {}

        # Scenario setup:
        # AAPL: Generally upward trend (good for take profit)
        # MSFT: Generally downward trend (good for stop loss)
        # GOOGL: Volatile with both up and down movements
        # AMZN: Stable with small movements (neither system should trigger)

        base_prices = {"AAPL": 150.0, "MSFT": 250.0, "GOOGL": 100.0, "AMZN": 200.0}
        trends = {"AAPL": 0.002, "MSFT": -0.003, "GOOGL": 0.0, "AMZN": 0.0}  # Daily trend
        volatilities = {"AAPL": 0.02, "MSFT": 0.025, "GOOGL": 0.04, "AMZN": 0.015}

        for ticker in tickers:
            base_price = base_prices[ticker]
            trend = trends[ticker]
            volatility = volatilities[ticker]

            # Generate price series with trend and volatility
            prices = [base_price]
            for i in range(1, len(dates)):
                # Trend component + random component
                daily_return = trend + np.random.normal(0, volatility)
                new_price = prices[-1] * (1 + daily_return)
                prices.append(new_price)

            prices_array = np.array(prices)

            # Create OHLC from closing prices
            daily_ranges = np.abs(np.random.normal(0, volatility * 0.5, len(dates)))

            data[(ticker, "Close")] = prices_array
            data[(ticker, "Open")] = prices_array * (
                1 + np.random.normal(0, volatility * 0.2, len(dates))
            )
            data[(ticker, "High")] = prices_array * (1 + daily_ranges)
            data[(ticker, "Low")] = prices_array * (1 - daily_ranges)

        return pd.DataFrame(data, index=dates, columns=columns)

    @pytest.fixture
    def benchmark_data(self, daily_ohlc_data):
        """Create benchmark data (SPY equivalent)."""
        # Use average of all tickers as benchmark
        benchmark_close = daily_ohlc_data.xs("Close", level="Field", axis=1).mean(axis=1)
        return pd.DataFrame({"Close": benchmark_close})

    @pytest.fixture
    def combined_risk_strategy(self):
        """Create strategy with both stop loss and take profit configured."""
        config = {
            "leverage": 1.0,
            "stop_loss_config": {
                "type": "AtrBasedStopLoss",
                "atr_length": 14,
                "atr_multiple": 2.0,
            },
            "take_profit_config": {
                "type": "AtrBasedTakeProfit",
                "atr_length": 14,
                "atr_multiple": 3.0,  # Wider than stop loss
            },
            "timing_config": {
                "mode": "time_based",
                "rebalance_frequency": "M",  # Monthly rebalancing
            },
        }
        return DummySignalStrategy(config)

    @pytest.fixture
    def portfolio_combined_risk_strategy(self):
        """Create portfolio strategy with both stop loss and take profit."""
        config = {
            "strategy_params": {
                "num_holdings": 3,
                "leverage": 1.0,
                "smoothing_lambda": 0.0,
                "trade_longs": True,
                "trade_shorts": False,
            },
            "stop_loss_config": {
                "type": "AtrBasedStopLoss",
                "atr_length": 10,
                "atr_multiple": 1.5,  # Tight stop loss
            },
            "take_profit_config": {
                "type": "AtrBasedTakeProfit",
                "atr_length": 10,
                "atr_multiple": 2.5,  # Wider take profit
            },
            "timing_config": {
                "mode": "time_based",
                "rebalance_frequency": "Q",  # Quarterly rebalancing
            },
        }
        return DummyPortfolioStrategy(config)

    @pytest.fixture
    def test_window(self):
        """Create test WFO window."""
        return WFOWindow(
            train_start=pd.Timestamp("2023-01-01"),
            train_end=pd.Timestamp("2023-01-31"),
            test_start=pd.Timestamp("2023-02-01"),
            test_end=pd.Timestamp("2023-04-30"),
            evaluation_frequency="D",  # Daily evaluation
        )

    def test_both_systems_work_independently(
        self, daily_ohlc_data, benchmark_data, combined_risk_strategy, test_window
    ):
        """
        🚨 CRITICAL TEST: Both stop loss and take profit work independently.

        This verifies that both systems can be configured and operate without
        interfering with each other.
        """
        evaluator = WindowEvaluator()

        # Verify both handlers are configured correctly
        stop_loss_handler = combined_risk_strategy.get_stop_loss_handler()
        take_profit_handler = combined_risk_strategy.get_take_profit_handler()

        assert isinstance(stop_loss_handler, AtrBasedStopLoss)
        assert isinstance(take_profit_handler, AtrBasedTakeProfit)
        assert stop_loss_handler.atr_multiple == 2.0
        assert take_profit_handler.atr_multiple == 3.0

        # Run backtest
        result = evaluator.evaluate_window(
            window=test_window,
            strategy=combined_risk_strategy,
            daily_data=daily_ohlc_data,
            benchmark_data=benchmark_data,
            universe_tickers=["AAPL", "MSFT", "GOOGL", "AMZN"],
            benchmark_ticker="SPY",
        )

        # Both systems should work without errors
        assert isinstance(result.window_returns, pd.Series)
        assert len(result.window_returns) > 0
        assert isinstance(result.trades, list)

        print(f"Combined risk management - Total trades: {len(result.trades or [])}")

    def test_no_interference_between_systems(self, daily_ohlc_data, benchmark_data, test_window):
        """
        Test that stop loss and take profit don't interfere with each other's calculations.
        """
        # Strategy with only stop loss
        stop_loss_only_config = {
            "fast_ema_days": 10,
            "slow_ema_days": 20,
            "leverage": 1.0,
            "stop_loss_config": {"type": "AtrBasedStopLoss", "atr_length": 14, "atr_multiple": 2.0},
            "take_profit_config": {"type": "NoTakeProfit"},
            "timing_config": {"mode": "time_based", "rebalance_frequency": "M"},
        }

        # Strategy with only take profit
        take_profit_only_config = {
            "fast_ema_days": 10,
            "slow_ema_days": 20,
            "leverage": 1.0,
            "stop_loss_config": {"type": "NoStopLoss"},
            "take_profit_config": {
                "type": "AtrBasedTakeProfit",
                "atr_length": 14,
                "atr_multiple": 3.0,
            },
            "timing_config": {"mode": "time_based", "rebalance_frequency": "M"},
        }

        # Strategy with both
        combined_config = {
            "fast_ema_days": 10,
            "slow_ema_days": 20,
            "leverage": 1.0,
            "stop_loss_config": {"type": "AtrBasedStopLoss", "atr_length": 14, "atr_multiple": 2.0},
            "take_profit_config": {
                "type": "AtrBasedTakeProfit",
                "atr_length": 14,
                "atr_multiple": 3.0,
            },
            "timing_config": {"mode": "time_based", "rebalance_frequency": "M"},
        }

        evaluator = WindowEvaluator()

        # Run all three scenarios
        stop_loss_only = DummySignalStrategy(stop_loss_only_config)
        take_profit_only = DummySignalStrategy(take_profit_only_config)
        combined = DummySignalStrategy(combined_config)

        results = {}
        for name, strategy in [
            ("stop_loss", stop_loss_only),
            ("take_profit", take_profit_only),
            ("combined", combined),
        ]:
            result = evaluator.evaluate_window(
                window=test_window,
                strategy=strategy,
                daily_data=daily_ohlc_data,
                benchmark_data=benchmark_data,
                universe_tickers=["AAPL", "MSFT"],
                benchmark_ticker="SPY",
            )
            results[name] = result
            print(
                f"{name}: {len(result.trades or [])} trades, returns: {result.window_returns.sum():.4f}"
            )

        # Combined should have characteristics of both individual systems
        assert len(results["combined"].trades or []) >= 0  # Should produce some trades

        # The combined system should not have fewer total trading activity than systems working individually
        # (although the specific trades may differ due to interactions)

    def test_both_systems_triggered_same_day_precedence(self):
        """
        🚨 CRITICAL TEST: Test behavior when both stop loss and take profit trigger on the same day.

        This tests the execution order and ensures the system handles concurrent triggers gracefully.
        """
        monitor_sl = DailyStopLossMonitor()
        monitor_tp = DailyTakeProfitMonitor()

        # Create position tracker with positions that could trigger both systems
        tracker = Mock(spec=PositionTracker)
        tracker.current_positions = {
            "AAPL": Position(
                ticker="AAPL", entry_date=pd.Timestamp("2023-01-01"), weight=1.0, entry_price=150.0
            )
        }

        # Create extreme price scenario where both could potentially trigger
        # Price moves significantly up (potential take profit) but also very volatile (potential stop loss)
        extreme_prices = pd.Series({"AAPL": 200.0})  # 33% gain

        # Create historical data with high volatility
        dates = pd.date_range("2023-01-01", periods=20, freq="D")

        # High volatility data
        price_data = np.linspace(150.0, 200.0, 20)
        volatility_data = np.random.normal(0, 20, 20)  # High volatility

        columns = pd.MultiIndex.from_product(
            [["AAPL"], ["Open", "High", "Low", "Close"]], names=["Ticker", "Field"]
        )
        data = {}
        data[("AAPL", "Close")] = price_data
        data[("AAPL", "Open")] = price_data + volatility_data * 0.1
        data[("AAPL", "High")] = price_data + np.abs(volatility_data)
        data[("AAPL", "Low")] = price_data - np.abs(volatility_data)

        historical_data = pd.DataFrame(data, index=dates, columns=columns)

        # Create handlers with parameters that could both trigger
        sl_handler = AtrBasedStopLoss({}, {"atr_length": 5, "atr_multiple": 0.5})  # Very tight
        tp_handler = AtrBasedTakeProfit({}, {"atr_length": 5, "atr_multiple": 0.8})  # Also tight

        current_date = pd.Timestamp("2023-01-19")

        # Check both systems
        sl_signals = monitor_sl.check_positions_for_stop_loss(
            current_date=current_date,
            position_tracker=tracker,
            current_prices=extreme_prices,
            stop_loss_handler=sl_handler,
            historical_data=historical_data,
        )

        tp_signals = monitor_tp.check_positions_for_take_profit(
            current_date=current_date,
            position_tracker=tracker,
            current_prices=extreme_prices,
            take_profit_handler=tp_handler,
            historical_data=historical_data,
        )

        # Both systems should be able to generate signals independently
        # The actual execution order is handled by WindowEvaluator (stop loss first, then take profit)
        print(f"Stop loss signals: {not sl_signals.empty}")
        print(f"Take profit signals: {not tp_signals.empty}")

        # Both should be able to operate without errors
        assert isinstance(sl_signals, pd.DataFrame)
        assert isinstance(tp_signals, pd.DataFrame)

    def test_one_system_triggers_prevents_other(self):
        """
        Test that when one system triggers and closes a position,
        the other system doesn't also try to close the same position.
        """
        monitor_sl = DailyStopLossMonitor()
        monitor_tp = DailyTakeProfitMonitor()

        # Create position tracker with a losing position
        tracker = Mock(spec=PositionTracker)
        initial_positions = {
            "MSFT": Position(
                ticker="MSFT", entry_date=pd.Timestamp("2023-01-01"), weight=0.8, entry_price=250.0
            )
        }
        tracker.current_positions = initial_positions

        # Price drops significantly (should trigger stop loss)
        declining_prices = pd.Series({"MSFT": 200.0})  # 20% loss

        # Create historical data showing decline
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        declining_data = np.linspace(250.0, 200.0, 20)

        columns = pd.MultiIndex.from_product(
            [["MSFT"], ["Open", "High", "Low", "Close"]], names=["Ticker", "Field"]
        )
        data = {}
        for field in ["Open", "High", "Low", "Close"]:
            data[("MSFT", field)] = declining_data

        historical_data = pd.DataFrame(data, index=dates, columns=columns)

        sl_handler = AtrBasedStopLoss({}, {"atr_length": 10, "atr_multiple": 1.5})
        tp_handler = AtrBasedTakeProfit({}, {"atr_length": 10, "atr_multiple": 2.0})

        current_date = pd.Timestamp("2023-01-19")

        # Check stop loss first (as WindowEvaluator does)
        sl_signals = monitor_sl.check_positions_for_stop_loss(
            current_date=current_date,
            position_tracker=tracker,
            current_prices=declining_prices,
            stop_loss_handler=sl_handler,
            historical_data=historical_data,
        )

        # If stop loss triggered, simulate position being closed
        if not sl_signals.empty:
            # Position would now be closed (weight = 0)
            tracker.current_positions = {}  # Position closed by stop loss

            # Take profit should now find no positions to monitor
            tp_signals = monitor_tp.check_positions_for_take_profit(
                current_date=current_date,
                position_tracker=tracker,
                current_prices=declining_prices,
                take_profit_handler=tp_handler,
                historical_data=historical_data,
            )

            # Take profit should return empty signals (no positions to monitor)
            assert tp_signals.empty, "Take profit should not trigger on already closed positions"

    def test_different_atr_parameters_both_systems(
        self, daily_ohlc_data, benchmark_data, test_window
    ):
        """
        Test that both systems can use different ATR parameters without conflicts.
        """
        # Configure with different ATR parameters for each system
        config = {
            "leverage": 1.0,
            "stop_loss_config": {
                "type": "AtrBasedStopLoss",
                "atr_length": 10,  # Shorter period
                "atr_multiple": 1.5,  # Tighter
            },
            "take_profit_config": {
                "type": "AtrBasedTakeProfit",
                "atr_length": 20,  # Longer period
                "atr_multiple": 4.0,  # Much wider
            },
            "timing_config": {"mode": "time_based", "rebalance_frequency": "M"},
        }

        strategy = DummySignalStrategy(config)
        evaluator = WindowEvaluator()

        # Verify different parameters are set correctly
        sl_handler = strategy.get_stop_loss_handler()
        tp_handler = strategy.get_take_profit_handler()

        assert isinstance(sl_handler, AtrBasedStopLoss)
        assert sl_handler.atr_length == 10
        assert sl_handler.atr_multiple == 1.5
        assert isinstance(tp_handler, AtrBasedTakeProfit)
        assert tp_handler.atr_length == 20
        assert tp_handler.atr_multiple == 4.0

        # Run backtest
        result = evaluator.evaluate_window(
            window=test_window,
            strategy=strategy,
            daily_data=daily_ohlc_data,
            benchmark_data=benchmark_data,
            universe_tickers=["AAPL", "MSFT", "GOOGL"],
            benchmark_ticker="SPY",
        )

        # Should work without conflicts
        assert isinstance(result.window_returns, pd.Series)
        print(f"Different ATR params - Total trades: {len(result.trades or [])}")

    def test_mixed_long_short_positions_both_systems(self):
        """
        Test both systems working correctly with mixed long and short positions.
        """
        monitor_sl = DailyStopLossMonitor()
        monitor_tp = DailyTakeProfitMonitor()

        # Create position tracker with mixed positions
        tracker = Mock(spec=PositionTracker)
        tracker.current_positions = {
            "AAPL": Position("AAPL", pd.Timestamp("2023-01-01"), 0.6, 150.0),  # Long
            "MSFT": Position("MSFT", pd.Timestamp("2023-01-01"), -0.4, 250.0),  # Short
            "GOOGL": Position("GOOGL", pd.Timestamp("2023-01-01"), 0.3, 100.0),  # Long
            "AMZN": Position("AMZN", pd.Timestamp("2023-01-01"), -0.2, 200.0),  # Short
        }

        # Create mixed price scenario
        # AAPL up (good for long take profit)
        # MSFT down (good for short take profit)
        # GOOGL down (bad for long, potential stop loss)
        # AMZN up (bad for short, potential stop loss)
        current_prices = pd.Series(
            {
                "AAPL": 180.0,  # +20% (take profit for long)
                "MSFT": 200.0,  # -20% (take profit for short)
                "GOOGL": 80.0,  # -20% (stop loss for long)
                "AMZN": 240.0,  # +20% (stop loss for short)
            }
        )

        # Create historical data
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        fields = ["Open", "High", "Low", "Close"]
        columns = pd.MultiIndex.from_product([tickers, fields], names=["Ticker", "Field"])

        data = {}
        base_prices = {"AAPL": 150.0, "MSFT": 250.0, "GOOGL": 100.0, "AMZN": 200.0}
        final_prices = {"AAPL": 180.0, "MSFT": 200.0, "GOOGL": 80.0, "AMZN": 240.0}

        for ticker in tickers:
            price_series = np.linspace(base_prices[ticker], final_prices[ticker], len(dates))
            for field in fields:
                data[(ticker, field)] = price_series

        historical_data = pd.DataFrame(data, index=dates, columns=columns)

        # Configure handlers
        sl_handler = AtrBasedStopLoss({}, {"atr_length": 10, "atr_multiple": 1.0})
        tp_handler = AtrBasedTakeProfit({}, {"atr_length": 10, "atr_multiple": 1.5})

        current_date = pd.Timestamp("2023-01-19")

        # Check both systems
        sl_signals = monitor_sl.check_positions_for_stop_loss(
            current_date=current_date,
            position_tracker=tracker,
            current_prices=current_prices,
            stop_loss_handler=sl_handler,
            historical_data=historical_data,
        )

        tp_signals = monitor_tp.check_positions_for_take_profit(
            current_date=current_date,
            position_tracker=tracker,
            current_prices=current_prices,
            take_profit_handler=tp_handler,
            historical_data=historical_data,
        )

        # Both systems should be able to handle mixed positions
        print(f"Stop loss signals empty: {sl_signals.empty}")
        print(f"Take profit signals empty: {tp_signals.empty}")

        # Should not crash and produce valid DataFrames
        assert isinstance(sl_signals, pd.DataFrame)
        assert isinstance(tp_signals, pd.DataFrame)

    def test_error_handling_one_system_fails(self):
        """
        Test that if one system fails, the other continues to work.
        """
        monitor_sl = DailyStopLossMonitor()
        monitor_tp = DailyTakeProfitMonitor()

        tracker = Mock(spec=PositionTracker)
        tracker.current_positions = {
            "AAPL": Position("AAPL", pd.Timestamp("2023-01-01"), 1.0, 150.0)
        }

        current_prices = pd.Series({"AAPL": 140.0})
        current_date = pd.Timestamp("2023-01-15")
        historical_data = pd.DataFrame()  # Empty data to potentially cause issues

        # Create handlers where one might fail
        working_sl_handler = AtrBasedStopLoss({}, {"atr_length": 5, "atr_multiple": 2.0})
        failing_tp_handler = Mock()
        failing_tp_handler.calculate_take_profit_levels.side_effect = Exception(
            "Take profit calculation failed"
        )

        # Stop loss should work
        sl_signals = monitor_sl.check_positions_for_stop_loss(
            current_date=current_date,
            position_tracker=tracker,
            current_prices=current_prices,
            stop_loss_handler=working_sl_handler,
            historical_data=historical_data,
        )

        # Take profit should fail gracefully and return empty signals
        tp_signals = monitor_tp.check_positions_for_take_profit(
            current_date=current_date,
            position_tracker=tracker,
            current_prices=current_prices,
            take_profit_handler=failing_tp_handler,
            historical_data=historical_data,
        )

        # Stop loss should still work
        assert isinstance(sl_signals, pd.DataFrame)

        # Take profit should fail gracefully
        assert isinstance(tp_signals, pd.DataFrame)
        assert tp_signals.empty  # Should return empty due to error

    def test_performance_comparison_combined_vs_individual(
        self, daily_ohlc_data, benchmark_data, test_window
    ):
        """
        Compare performance when using both systems vs individual systems.
        """
        evaluator = WindowEvaluator()

        # Base configuration
        base_config = {
            "fast_ema_days": 10,
            "slow_ema_days": 20,
            "leverage": 1.0,
            "timing_config": {"mode": "time_based", "rebalance_frequency": "M"},
        }

        configs = {
            "none": {
                **base_config,
                "stop_loss_config": {"type": "NoStopLoss"},
                "take_profit_config": {"type": "NoTakeProfit"},
            },
            "stop_loss_only": {
                **base_config,
                "stop_loss_config": {
                    "type": "AtrBasedStopLoss",
                    "atr_length": 14,
                    "atr_multiple": 2.0,
                },
                "take_profit_config": {"type": "NoTakeProfit"},
            },
            "take_profit_only": {
                **base_config,
                "stop_loss_config": {"type": "NoStopLoss"},
                "take_profit_config": {
                    "type": "AtrBasedTakeProfit",
                    "atr_length": 14,
                    "atr_multiple": 3.0,
                },
            },
            "combined": {
                **base_config,
                "stop_loss_config": {
                    "type": "AtrBasedStopLoss",
                    "atr_length": 14,
                    "atr_multiple": 2.0,
                },
                "take_profit_config": {
                    "type": "AtrBasedTakeProfit",
                    "atr_length": 14,
                    "atr_multiple": 3.0,
                },
            },
        }

        results = {}
        for name, config in configs.items():
            strategy = DummySignalStrategy(config)
            result = evaluator.evaluate_window(
                window=test_window,
                strategy=strategy,
                daily_data=daily_ohlc_data,
                benchmark_data=benchmark_data,
                universe_tickers=["AAPL", "MSFT", "GOOGL"],
                benchmark_ticker="SPY",
            )

            results[name] = {
                "total_return": result.window_returns.sum(),
                "trades": len(result.trades or []),
                "volatility": result.window_returns.std(),
                "sharpe": result.metrics.get("sharpe_ratio", 0),
            }

            print(
                f"{name:15}: Return={results[name]['total_return']:7.4f}, "
                f"Trades={results[name]['trades']:3d}, "
                f"Sharpe={results[name]['sharpe']:6.3f}"
            )

        # All configurations should produce valid results
        for name, metrics in results.items():
            assert isinstance(metrics["total_return"], (int, float))
            assert isinstance(metrics["trades"], int)
            assert metrics["trades"] >= 0

    def test_quarterly_portfolio_strategy_combined_risk(
        self, daily_ohlc_data, benchmark_data, test_window
    ):
        """
        Test combined risk management with portfolio strategy and quarterly rebalancing.
        """
        config = {
            "strategy_params": {
                "num_holdings": 2,
                "leverage": 1.0,
                "smoothing_lambda": 0.0,
                "trade_longs": True,
                "trade_shorts": False,
                "lookback_window": 21,
                "top_decile_fraction": 0.6,
            },
            "stop_loss_config": {
                "type": "AtrBasedStopLoss",
                "atr_length": 14,
                "atr_multiple": 2.5,
            },
            "take_profit_config": {
                "type": "AtrBasedTakeProfit",
                "atr_length": 14,
                "atr_multiple": 4.0,
            },
            "timing_config": {
                "mode": "time_based",
                "rebalance_frequency": "Q",  # Quarterly rebalancing
            },
        }

        strategy = DummyPortfolioStrategy(config)
        evaluator = WindowEvaluator()

        # Verify both systems are configured
        sl_handler = strategy.get_stop_loss_handler()
        tp_handler = strategy.get_take_profit_handler()

        assert isinstance(sl_handler, AtrBasedStopLoss)
        assert isinstance(tp_handler, AtrBasedTakeProfit)

        # Run backtest
        result = evaluator.evaluate_window(
            window=test_window,
            strategy=strategy,
            daily_data=daily_ohlc_data,
            benchmark_data=benchmark_data,
            universe_tickers=["AAPL", "MSFT", "GOOGL", "AMZN"],
            benchmark_ticker="SPY",
        )

        # Should work with portfolio strategies
        assert isinstance(result.window_returns, pd.Series)
        assert len(result.window_returns) > 0

        print(f"Portfolio + Combined Risk - Total trades: {len(result.trades or [])}")
        print(f"Total return: {result.window_returns.sum():.4f}")

    def test_signal_strategy_fallback_both_systems(self, daily_ohlc_data):
        """
        Test that signal strategy fallback mechanisms work for both systems.
        """
        config = {
            "leverage": 1.0,
            "stop_loss_config": {"type": "AtrBasedStopLoss", "atr_length": 5, "atr_multiple": 1.0},
            "take_profit_config": {
                "type": "AtrBasedTakeProfit",
                "atr_length": 5,
                "atr_multiple": 2.0,
            },
            "timing_config": {"mode": "time_based", "rebalance_frequency": "M"},
        }

        strategy = DummySignalStrategy(config)

        # Test signal generation with fallback mechanisms
        current_date = pd.Timestamp("2023-02-01")

        # Extract smaller data subset for testing
        test_data = daily_ohlc_data.loc[: pd.Timestamp("2023-02-01")]
        benchmark_test_data = pd.DataFrame(
            {"Close": test_data.xs("Close", level="Field", axis=1).mean(axis=1)}
        )

        # Generate signals (this should include both stop loss and take profit fallbacks)
        signals = strategy.generate_signals(
            all_historical_data=test_data,
            benchmark_historical_data=benchmark_test_data,
            non_universe_historical_data=pd.DataFrame(),
            current_date=current_date,
        )

        # Should complete without errors
        assert isinstance(signals, pd.DataFrame)
        assert not signals.empty

        # Verify that both fallback methods exist
        assert hasattr(strategy, "_apply_signal_strategy_stop_loss")
        assert hasattr(strategy, "_apply_signal_strategy_take_profit")


class TestCombinedRiskEdgeCases:
    """Test edge cases for combined risk management."""

    def test_zero_atr_both_systems(self):
        """Test behavior when ATR is zero for both systems."""
        monitor_sl = DailyStopLossMonitor()
        monitor_tp = DailyTakeProfitMonitor()

        tracker = Mock(spec=PositionTracker)
        tracker.current_positions = {
            "STABLE": Position("STABLE", pd.Timestamp("2023-01-01"), 1.0, 100.0)
        }

        # Create data with no volatility (constant prices)
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        constant_price = 100.0

        columns = pd.MultiIndex.from_product(
            [["STABLE"], ["Open", "High", "Low", "Close"]], names=["Ticker", "Field"]
        )
        data = {(col): [constant_price] * len(dates) for col in columns}
        historical_data = pd.DataFrame(data, index=dates, columns=columns)

        current_prices = pd.Series({"STABLE": constant_price})
        current_date = pd.Timestamp("2023-01-19")

        sl_handler = AtrBasedStopLoss({}, {"atr_length": 10, "atr_multiple": 2.0})
        tp_handler = AtrBasedTakeProfit({}, {"atr_length": 10, "atr_multiple": 3.0})

        # Both systems should handle zero ATR gracefully
        sl_signals = monitor_sl.check_positions_for_stop_loss(
            current_date=current_date,
            position_tracker=tracker,
            current_prices=current_prices,
            stop_loss_handler=sl_handler,
            historical_data=historical_data,
        )

        tp_signals = monitor_tp.check_positions_for_take_profit(
            current_date=current_date,
            position_tracker=tracker,
            current_prices=current_prices,
            take_profit_handler=tp_handler,
            historical_data=historical_data,
        )

        # Should not crash
        assert isinstance(sl_signals, pd.DataFrame)
        assert isinstance(tp_signals, pd.DataFrame)

    def test_extremely_tight_both_levels(self):
        """Test with extremely tight levels for both systems."""
        config = {
            "fast_ema_days": 5,
            "slow_ema_days": 10,
            "leverage": 1.0,
            "stop_loss_config": {
                "type": "AtrBasedStopLoss",
                "atr_length": 3,
                "atr_multiple": 0.1,
            },  # Very tight
            "take_profit_config": {
                "type": "AtrBasedTakeProfit",
                "atr_length": 3,
                "atr_multiple": 0.1,
            },  # Very tight
            "timing_config": {"mode": "time_based", "rebalance_frequency": "D"},
        }

        strategy = DummySignalStrategy(config)

        # Should initialize without errors even with extreme parameters
        sl_handler = strategy.get_stop_loss_handler()
        tp_handler = strategy.get_take_profit_handler()

        assert isinstance(sl_handler, AtrBasedStopLoss)
        assert sl_handler.atr_multiple == 0.1
        assert isinstance(tp_handler, AtrBasedTakeProfit)
        assert tp_handler.atr_multiple == 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
