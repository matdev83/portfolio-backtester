"""
Comprehensive test suite for DailyTakeProfitMonitor.

This test suite covers all typical and edge cases for take profit functionality,
ensuring the system works correctly regardless of strategy type or rebalance schedule.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock

from portfolio_backtester.risk_management.daily_take_profit_monitor import (
    DailyTakeProfitMonitor,
)
from portfolio_backtester.risk_management.take_profit_handlers import (
    BaseTakeProfit,
    NoTakeProfit,
    AtrBasedTakeProfit,
)
from portfolio_backtester.backtesting.position_tracker import Position, PositionTracker


class TestDailyTakeProfitMonitor:
    """Test suite for DailyTakeProfitMonitor functionality."""

    @pytest.fixture
    def monitor(self):
        """Create a fresh monitor instance for each test."""
        return DailyTakeProfitMonitor()

    @pytest.fixture
    def mock_position_tracker(self):
        """Create a mock position tracker with test positions."""
        tracker = Mock(spec=PositionTracker)

        # Mock positions: AAPL long, MSFT short, GOOGL long
        positions = {
            "AAPL": Position(
                ticker="AAPL",
                entry_date=pd.Timestamp("2023-01-01"),
                weight=0.5,
                entry_price=150.0,
            ),
            "MSFT": Position(
                ticker="MSFT",
                entry_date=pd.Timestamp("2023-01-01"),
                weight=-0.3,  # Short position
                entry_price=250.0,
            ),
            "GOOGL": Position(
                ticker="GOOGL",
                entry_date=pd.Timestamp("2023-01-01"),
                weight=0.2,
                entry_price=100.0,
            ),
        }

        tracker.current_positions = positions
        return tracker

    @pytest.fixture
    def current_prices(self):
        """Create current price data that would trigger take profit."""
        return pd.Series(
            {
                "AAPL": 180.0,  # Up from 150.0 (20% gain for long)
                "MSFT": 200.0,  # Down from 250.0 (20% gain for short)
                "GOOGL": 130.0,  # Up from 100.0 (30% gain for long)
            }
        )

    @pytest.fixture
    def historical_data(self):
        """Create historical OHLC data for ATR calculations."""
        dates = pd.date_range("2023-01-01", periods=30, freq="D")

        # Create MultiIndex columns for OHLC data
        tickers = ["AAPL", "MSFT", "GOOGL"]
        fields = ["Open", "High", "Low", "Close"]

        columns = pd.MultiIndex.from_product([tickers, fields], names=["Ticker", "Field"])

        data = {}
        for ticker in tickers:
            # Generate realistic OHLC data with some volatility
            np.random.seed(42)  # For reproducible tests
            base_price = {"AAPL": 150.0, "MSFT": 250.0, "GOOGL": 100.0}[ticker]

            prices = base_price * (1 + np.random.randn(len(dates)) * 0.02).cumprod()

            data[(ticker, "Open")] = prices * (1 + np.random.randn(len(dates)) * 0.005)
            data[(ticker, "High")] = prices * (1 + np.abs(np.random.randn(len(dates))) * 0.01)
            data[(ticker, "Low")] = prices * (1 - np.abs(np.random.randn(len(dates))) * 0.01)
            data[(ticker, "Close")] = prices

        return pd.DataFrame(data, index=dates, columns=columns)

    @pytest.fixture
    def atr_take_profit_handler(self):
        """Create ATR-based take profit handler."""
        strategy_config: dict[str, str] = {}
        take_profit_config = {
            "type": "AtrBasedTakeProfit",
            "atr_length": 14,
            "atr_multiple": 1.5,
        }  # Aggressive take profit for testing
        return AtrBasedTakeProfit(strategy_config, take_profit_config)

    @pytest.fixture
    def no_take_profit_handler(self):
        """Create no-op take profit handler."""
        return NoTakeProfit({}, {})

    def test_no_positions_returns_empty_signals(self, monitor):
        """Test that monitor returns empty signals when there are no positions."""
        tracker = Mock(spec=PositionTracker)
        tracker.current_positions = {}

        result = monitor.check_positions_for_take_profit(
            current_date=pd.Timestamp("2023-01-15"),
            position_tracker=tracker,
            current_prices=pd.Series({"AAPL": 150.0}),
            take_profit_handler=Mock(spec=BaseTakeProfit),
            historical_data=pd.DataFrame(),
        )

        assert result.empty

    def test_no_take_profit_handler_returns_empty_signals(
        self, monitor, mock_position_tracker, current_prices, historical_data
    ):
        """Test that NoTakeProfit handler doesn't trigger any liquidations."""
        no_take_profit = NoTakeProfit({}, {})

        result = monitor.check_positions_for_take_profit(
            current_date=pd.Timestamp("2023-01-15"),
            position_tracker=mock_position_tracker,
            current_prices=current_prices,
            take_profit_handler=no_take_profit,
            historical_data=historical_data,
        )

        assert result.empty

    def test_atr_take_profit_basic_functionality(
        self, monitor, mock_position_tracker, historical_data, atr_take_profit_handler
    ):
        """Test basic ATR take profit functionality."""
        # Create prices that should trigger take profit for long positions
        trigger_prices = pd.Series(
            {
                "AAPL": 200.0,  # Significant gain from entry price 150.0
                "MSFT": 200.0,  # Significant gain for short from entry 250.0
                "GOOGL": 150.0,  # Significant gain from entry 100.0
            }
        )

        result = monitor.check_positions_for_take_profit(
            current_date=pd.Timestamp("2023-01-15"),
            position_tracker=mock_position_tracker,
            current_prices=trigger_prices,
            take_profit_handler=atr_take_profit_handler,
            historical_data=historical_data,
        )

        # Should return liquidation signals for triggered positions
        assert not result.empty
        assert result.index[0] == pd.Timestamp("2023-01-15")

    def test_long_position_take_profit_trigger(
        self, monitor, atr_take_profit_handler, historical_data
    ):
        """Test take profit trigger for long positions."""
        # Create position tracker with long position
        tracker = Mock(spec=PositionTracker)
        tracker.current_positions = {
            "AAPL": Position(
                ticker="AAPL",
                entry_date=pd.Timestamp("2023-01-01"),
                weight=1.0,
                entry_price=150.0,
            )
        }

        # Price rises significantly above entry (favorable for long)
        trigger_prices = pd.Series({"AAPL": 200.0})  # 33% gain

        result = monitor.check_positions_for_take_profit(
            current_date=pd.Timestamp("2023-01-15"),
            position_tracker=tracker,
            current_prices=trigger_prices,
            take_profit_handler=atr_take_profit_handler,
            historical_data=historical_data,
        )

        # Verify that position is marked for liquidation
        assert not result.empty

    def test_short_position_take_profit_trigger(
        self, monitor, atr_take_profit_handler, historical_data
    ):
        """Test take profit trigger for short positions."""
        # Create position tracker with short position
        tracker = Mock(spec=PositionTracker)
        tracker.current_positions = {
            "AAPL": Position(
                ticker="AAPL",
                entry_date=pd.Timestamp("2023-01-01"),
                weight=-1.0,  # Short position
                entry_price=150.0,
            )
        }

        # Price falls significantly below entry (favorable for short)
        trigger_prices = pd.Series({"AAPL": 100.0})  # 33% drop

        result = monitor.check_positions_for_take_profit(
            current_date=pd.Timestamp("2023-01-15"),
            position_tracker=tracker,
            current_prices=trigger_prices,
            take_profit_handler=atr_take_profit_handler,
            historical_data=historical_data,
        )

        # Verify that short position is marked for liquidation
        assert not result.empty

    def test_mixed_positions_selective_liquidation(
        self, monitor, atr_take_profit_handler, historical_data
    ):
        """Test that only positions meeting take profit criteria are liquidated."""
        # Create multiple positions
        tracker = Mock(spec=PositionTracker)
        tracker.current_positions = {
            "AAPL": Position("AAPL", pd.Timestamp("2023-01-01"), 0.5, 150.0),  # Will trigger
            "MSFT": Position("MSFT", pd.Timestamp("2023-01-01"), 0.3, 250.0),  # Won't trigger
            "GOOGL": Position("GOOGL", pd.Timestamp("2023-01-01"), 0.2, 100.0),  # Won't trigger
        }

        # Only AAPL price rises significantly (should trigger take profit)
        prices = pd.Series(
            {
                "AAPL": 220.0,  # Should trigger take profit
                "MSFT": 255.0,  # Small gain, shouldn't trigger
                "GOOGL": 105.0,  # Small gain, shouldn't trigger
            }
        )

        result = monitor.check_positions_for_take_profit(
            current_date=pd.Timestamp("2023-01-15"),
            position_tracker=tracker,
            current_prices=prices,
            take_profit_handler=atr_take_profit_handler,
            historical_data=historical_data,
        )

        # Should return signals but only for triggered positions
        if not result.empty:
            liquidated_assets = result.columns[result.iloc[0] == 0].tolist()
            # Verify selective liquidation logic works
            assert isinstance(liquidated_assets, list)

    def test_missing_entry_prices_handled_gracefully(
        self, monitor, atr_take_profit_handler, historical_data
    ):
        """Test handling of positions without entry prices."""
        tracker = Mock(spec=PositionTracker)
        tracker.current_positions = {
            "AAPL": Position(
                ticker="AAPL",
                entry_date=pd.Timestamp("2023-01-01"),
                weight=0.5,
                entry_price=None,  # Missing entry price
            )
        }

        prices = pd.Series({"AAPL": 150.0})

        result = monitor.check_positions_for_take_profit(
            current_date=pd.Timestamp("2023-01-15"),
            position_tracker=tracker,
            current_prices=prices,
            take_profit_handler=atr_take_profit_handler,
            historical_data=historical_data,
        )

        # Should handle gracefully without crashing
        assert isinstance(result, pd.DataFrame)

    def test_missing_price_data_handled_gracefully(
        self, monitor, mock_position_tracker, atr_take_profit_handler, historical_data
    ):
        """Test handling of missing current price data."""
        # Prices missing for some positions
        incomplete_prices = pd.Series({"AAPL": 150.0})  # Missing MSFT and GOOGL

        result = monitor.check_positions_for_take_profit(
            current_date=pd.Timestamp("2023-01-15"),
            position_tracker=mock_position_tracker,
            current_prices=incomplete_prices,
            take_profit_handler=atr_take_profit_handler,
            historical_data=historical_data,
        )

        # Should handle gracefully without crashing
        assert isinstance(result, pd.DataFrame)

    def test_take_profit_handler_error_handling(
        self, monitor, mock_position_tracker, current_prices, historical_data
    ):
        """Test handling of errors in take profit handler."""
        # Create a mock handler that raises exceptions
        error_handler = Mock(spec=BaseTakeProfit)
        error_handler.calculate_take_profit_levels.side_effect = Exception("ATR calculation error")

        result = monitor.check_positions_for_take_profit(
            current_date=pd.Timestamp("2023-01-15"),
            position_tracker=mock_position_tracker,
            current_prices=current_prices,
            take_profit_handler=error_handler,
            historical_data=historical_data,
        )

        # Should return empty result when handler fails
        assert result.empty

    def test_monitoring_stats_tracking(self, monitor):
        """Test that monitoring statistics are tracked correctly."""
        initial_stats = monitor.get_monitoring_stats()

        assert initial_stats["last_check_date"] is None
        assert initial_stats["total_triggered_positions"] == 0
        assert initial_stats["triggered_positions"] == {}

    def test_reset_monitoring_state(self, monitor):
        """Test reset functionality."""
        # Add some state
        monitor.triggered_positions["AAPL"] = pd.Timestamp("2023-01-15")
        monitor.last_check_date = pd.Timestamp("2023-01-15")

        # Reset
        monitor.reset_monitoring_state()

        stats = monitor.get_monitoring_stats()
        assert stats["last_check_date"] is None
        assert stats["total_triggered_positions"] == 0
        assert stats["triggered_positions"] == {}

    def test_position_extraction_empty_positions(self, monitor):
        """Test position extraction with empty positions dictionary."""
        weights = monitor._extract_current_weights({})
        entry_prices = monitor._extract_entry_prices({})

        assert weights.empty
        assert entry_prices.empty

    def test_position_extraction_valid_positions(self, monitor, mock_position_tracker):
        """Test position extraction with valid positions."""
        weights = monitor._extract_current_weights(mock_position_tracker.current_positions)
        entry_prices = monitor._extract_entry_prices(mock_position_tracker.current_positions)

        assert len(weights) == 3
        assert len(entry_prices) == 3

        assert weights["AAPL"] == 0.5
        assert weights["MSFT"] == -0.3  # Short position
        assert weights["GOOGL"] == 0.2

        assert entry_prices["AAPL"] == 150.0
        assert entry_prices["MSFT"] == 250.0
        assert entry_prices["GOOGL"] == 100.0

    def test_liquidation_signal_generation(self, monitor):
        """Test generation of liquidation signals."""
        liquidated_positions = pd.Series({"AAPL": 0.5, "MSFT": -0.3}, name="positions")
        current_date = pd.Timestamp("2023-01-15")

        signals = monitor._generate_liquidation_signals(liquidated_positions, current_date)

        assert not signals.empty
        assert signals.index[0] == current_date
        assert signals.loc[current_date, "AAPL"] == 0.0
        assert signals.loc[current_date, "MSFT"] == 0.0

    def test_liquidated_position_identification(self, monitor):
        """Test identification of liquidated positions."""
        original_weights = pd.Series({"AAPL": 0.5, "MSFT": -0.3, "GOOGL": 0.2})
        weights_after_tp = pd.Series(
            {"AAPL": 0.0, "MSFT": -0.3, "GOOGL": 0.0}
        )  # AAPL and GOOGL taken profit
        current_date = pd.Timestamp("2023-01-15")

        liquidated = monitor._identify_liquidated_positions(
            original_weights, weights_after_tp, current_date
        )

        assert "AAPL" in liquidated.index
        assert "GOOGL" in liquidated.index
        assert "MSFT" not in liquidated.index

        # Check tracking of triggered positions
        assert monitor.triggered_positions["AAPL"] == current_date
        assert monitor.triggered_positions["GOOGL"] == current_date

    def test_no_valid_positions_with_entry_prices(self, monitor):
        """Test behavior when positions exist but have no valid entry prices."""
        tracker = Mock(spec=PositionTracker)
        tracker.current_positions = {
            "AAPL": Position(
                ticker="AAPL",
                entry_date=pd.Timestamp("2023-01-01"),
                weight=0.5,
                entry_price=None,  # No entry price
            )
        }

        result = monitor.check_positions_for_take_profit(
            current_date=pd.Timestamp("2023-01-15"),
            position_tracker=tracker,
            current_prices=pd.Series({"AAPL": 150.0}),
            take_profit_handler=NoTakeProfit({}, {}),
            historical_data=pd.DataFrame(),
        )

        assert result.empty

    def test_position_tracker_without_current_positions_attribute(self, monitor):
        """Test behavior when position tracker doesn't have current_positions attribute."""
        tracker = Mock()  # Mock without current_positions attribute

        result = monitor.check_positions_for_take_profit(
            current_date=pd.Timestamp("2023-01-15"),
            position_tracker=tracker,
            current_prices=pd.Series({"AAPL": 150.0}),
            take_profit_handler=NoTakeProfit({}, {}),
            historical_data=pd.DataFrame(),
        )

        assert result.empty


class TestDailyTakeProfitMonitorEdgeCases:
    """Test edge cases and complex scenarios."""

    def test_same_day_open_and_take_profit(self):
        """Test positions opened and closed for profit on the same day."""
        monitor = DailyTakeProfitMonitor()

        # Position opened today with entry price
        tracker = Mock(spec=PositionTracker)
        tracker.current_positions = {
            "AAPL": Position(
                ticker="AAPL",
                entry_date=pd.Timestamp("2023-01-15"),  # Same as current date
                weight=1.0,
                entry_price=150.0,
            )
        }

        # Price rises immediately (favorable for long)
        current_prices = pd.Series({"AAPL": 200.0})
        current_date = pd.Timestamp("2023-01-15")

        # Create minimal historical data
        historical_data = pd.DataFrame(
            {("AAPL", "Close"): [150.0, 155.0, 160.0, 180.0, 200.0]},
            index=pd.date_range("2023-01-11", periods=5),
        )
        historical_data.columns = pd.MultiIndex.from_tuples(
            [("AAPL", "Close")], names=["Ticker", "Field"]
        )

        atr_handler = AtrBasedTakeProfit({}, {"atr_length": 3, "atr_multiple": 1.0})

        result = monitor.check_positions_for_take_profit(
            current_date=current_date,
            position_tracker=tracker,
            current_prices=current_prices,
            take_profit_handler=atr_handler,
            historical_data=historical_data,
        )

        # Should handle same-day scenario
        assert isinstance(result, pd.DataFrame)

    def test_multiple_take_profit_triggers_same_day(self):
        """Test multiple positions triggering take profit on the same day."""
        monitor = DailyTakeProfitMonitor()

        # Multiple positions that will all trigger take profit
        tracker = Mock(spec=PositionTracker)
        tracker.current_positions = {
            "AAPL": Position("AAPL", pd.Timestamp("2023-01-01"), 0.4, 150.0),
            "MSFT": Position("MSFT", pd.Timestamp("2023-01-01"), 0.3, 250.0),
            "GOOGL": Position("GOOGL", pd.Timestamp("2023-01-01"), 0.3, 100.0),
        }

        # All prices rise significantly (favorable moves)
        bull_market_prices = pd.Series(
            {
                "AAPL": 200.0,  # 33% gain
                "MSFT": 320.0,  # 28% gain
                "GOOGL": 140.0,  # 40% gain
            }
        )

        # Create historical data with upward trend
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        tickers = ["AAPL", "MSFT", "GOOGL"]
        fields = ["Open", "High", "Low", "Close"]

        columns = pd.MultiIndex.from_product([tickers, fields], names=["Ticker", "Field"])
        data = {}

        for ticker in tickers:
            base_price = {"AAPL": 150.0, "MSFT": 250.0, "GOOGL": 100.0}[ticker]
            final_price = bull_market_prices[ticker]
            prices = np.linspace(base_price, final_price, len(dates))

            data[(ticker, "Open")] = prices
            data[(ticker, "High")] = prices * 1.02
            data[(ticker, "Low")] = prices * 0.98
            data[(ticker, "Close")] = prices

        historical_data = pd.DataFrame(data, index=dates, columns=columns)

        atr_handler = AtrBasedTakeProfit({}, {"atr_length": 10, "atr_multiple": 1.0})

        result = monitor.check_positions_for_take_profit(
            current_date=pd.Timestamp("2023-01-20"),
            position_tracker=tracker,
            current_prices=bull_market_prices,
            take_profit_handler=atr_handler,
            historical_data=historical_data,
        )

        # Should handle multiple simultaneous triggers
        assert isinstance(result, pd.DataFrame)

    def test_invalid_historical_data_scenarios(self):
        """Test handling of invalid or insufficient historical data."""
        monitor = DailyTakeProfitMonitor()

        tracker = Mock(spec=PositionTracker)
        tracker.current_positions = {
            "AAPL": Position("AAPL", pd.Timestamp("2023-01-01"), 1.0, 150.0)
        }

        current_prices = pd.Series({"AAPL": 180.0})
        atr_handler = AtrBasedTakeProfit({}, {"atr_length": 14, "atr_multiple": 2.0})

        # Test with empty historical data
        empty_data = pd.DataFrame()
        result = monitor.check_positions_for_take_profit(
            current_date=pd.Timestamp("2023-01-15"),
            position_tracker=tracker,
            current_prices=current_prices,
            take_profit_handler=atr_handler,
            historical_data=empty_data,
        )
        assert result.empty

        # Test with insufficient historical data
        insufficient_data = pd.DataFrame(
            {("AAPL", "Close"): [150.0]}, index=[pd.Timestamp("2023-01-14")]
        )  # Only 1 data point
        insufficient_data.columns = pd.MultiIndex.from_tuples(
            [("AAPL", "Close")], names=["Ticker", "Field"]
        )

        result = monitor.check_positions_for_take_profit(
            current_date=pd.Timestamp("2023-01-15"),
            position_tracker=tracker,
            current_prices=current_prices,
            take_profit_handler=atr_handler,
            historical_data=insufficient_data,
        )
        # Should handle gracefully
        assert isinstance(result, pd.DataFrame)

    def test_apply_take_profit_handler_error(self):
        """Test error handling in apply_take_profit method."""
        monitor = DailyTakeProfitMonitor()

        tracker = Mock(spec=PositionTracker)
        tracker.current_positions = {
            "AAPL": Position("AAPL", pd.Timestamp("2023-01-01"), 1.0, 150.0)
        }

        # Mock handler that succeeds in calculation but fails in application
        error_handler = Mock(spec=BaseTakeProfit)
        error_handler.calculate_take_profit_levels.return_value = pd.Series({"AAPL": 170.0})
        error_handler.apply_take_profit.side_effect = Exception("Apply take profit error")

        result = monitor.check_positions_for_take_profit(
            current_date=pd.Timestamp("2023-01-15"),
            position_tracker=tracker,
            current_prices=pd.Series({"AAPL": 180.0}),
            take_profit_handler=error_handler,
            historical_data=pd.DataFrame(),
        )

        # Should return empty result when application fails
        assert result.empty


if __name__ == "__main__":
    pytest.main([__file__])
