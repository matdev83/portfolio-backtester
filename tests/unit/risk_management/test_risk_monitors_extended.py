import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, Mock
from src.portfolio_backtester.risk_management.daily_stop_loss_monitor import DailyStopLossMonitor
from src.portfolio_backtester.risk_management.daily_take_profit_monitor import DailyTakeProfitMonitor
from src.portfolio_backtester.backtesting.position_tracker import Position

class TestDailyStopLossMonitor:
    @pytest.fixture
    def monitor(self):
        return DailyStopLossMonitor()

    @pytest.fixture
    def mock_position_tracker(self):
        tracker = MagicMock()
        # Setup some dummy positions
        pos1 = Position(ticker="AAPL", weight=0.5, entry_price=100.0, entry_date=pd.Timestamp("2020-01-01"))
        pos2 = Position(ticker="MSFT", weight=0.5, entry_price=200.0, entry_date=pd.Timestamp("2020-01-01"))
        tracker.current_positions = {"AAPL": pos1, "MSFT": pos2}
        return tracker

    @pytest.fixture
    def mock_stop_loss_handler(self):
        handler = MagicMock()
        return handler

    def test_no_positions(self, monitor, mock_stop_loss_handler):
        tracker = MagicMock()
        tracker.current_positions = {}
        
        current_date = pd.Timestamp("2020-01-02")
        current_prices = pd.Series({"AAPL": 105.0, "MSFT": 205.0})
        historical_data = pd.DataFrame() # Not used in this path
        
        result = monitor.check_positions_for_stop_loss(
            current_date, tracker, current_prices, mock_stop_loss_handler, historical_data
        )
        
        assert result.empty
        assert not monitor.triggered_positions

    def test_stop_loss_triggered(self, monitor, mock_position_tracker, mock_stop_loss_handler):
        current_date = pd.Timestamp("2020-01-02")
        current_prices = pd.Series({"AAPL": 90.0, "MSFT": 210.0}) # AAPL dropped
        historical_data = pd.DataFrame()
        
        # Mock calculate_stop_levels return
        # Say AAPL stop is 95, MSFT stop is 190
        mock_stop_loss_handler.calculate_stop_levels.return_value = pd.Series(
            {"AAPL": 95.0, "MSFT": 190.0}, index=["AAPL", "MSFT"]
        )
        
        # Mock apply_stop_loss return
        # Should return weights where AAPL is 0 (liquidated) and MSFT is 0.5 (kept)
        mock_stop_loss_handler.apply_stop_loss.return_value = pd.Series(
            {"AAPL": 0.0, "MSFT": 0.5}, index=["AAPL", "MSFT"]
        )
        
        result = monitor.check_positions_for_stop_loss(
            current_date, mock_position_tracker, current_prices, mock_stop_loss_handler, historical_data
        )
        
        assert not result.empty
        assert isinstance(result, pd.DataFrame)
        assert result.index[0] == current_date
        assert "AAPL" in result.columns
        assert result.loc[current_date, "AAPL"] == 0.0
        # MSFT should not be in liquidation signals or have 0 weight signal?
        # The monitor returns liquidation signals for *triggered* positions.
        # It creates a row with ALL liquidated positions set to 0.
        # But wait, `_generate_liquidation_signals` creates a Series of 0.0 for liquidated positions.
        # So result should only contain columns for AAPL.
        
        assert "AAPL" in result.columns
        assert "MSFT" not in result.columns
        
        assert "AAPL" in monitor.triggered_positions
        assert monitor.triggered_positions["AAPL"] == current_date
        assert "MSFT" not in monitor.triggered_positions

    def test_handler_exception_handling(self, monitor, mock_position_tracker, mock_stop_loss_handler):
        current_date = pd.Timestamp("2020-01-02")
        current_prices = pd.Series({"AAPL": 100.0, "MSFT": 200.0})
        
        mock_stop_loss_handler.calculate_stop_levels.side_effect = Exception("Calculation error")
        
        result = monitor.check_positions_for_stop_loss(
            current_date, mock_position_tracker, current_prices, mock_stop_loss_handler, pd.DataFrame()
        )
        
        assert result.empty


class TestDailyTakeProfitMonitor:
    @pytest.fixture
    def monitor(self):
        return DailyTakeProfitMonitor()

    @pytest.fixture
    def mock_position_tracker(self):
        tracker = MagicMock()
        pos1 = Position(ticker="AAPL", weight=0.5, entry_price=100.0, entry_date=pd.Timestamp("2020-01-01"))
        pos2 = Position(ticker="MSFT", weight=0.5, entry_price=200.0, entry_date=pd.Timestamp("2020-01-01"))
        tracker.current_positions = {"AAPL": pos1, "MSFT": pos2}
        return tracker

    @pytest.fixture
    def mock_take_profit_handler(self):
        handler = MagicMock()
        return handler

    def test_take_profit_triggered(self, monitor, mock_position_tracker, mock_take_profit_handler):
        current_date = pd.Timestamp("2020-01-02")
        # AAPL rose significantly to 150
        current_prices = pd.Series({"AAPL": 150.0, "MSFT": 205.0})
        historical_data = pd.DataFrame()
        
        # Mock calculate_take_profit_levels
        # Say AAPL TP is 140, MSFT TP is 250
        mock_take_profit_handler.calculate_take_profit_levels.return_value = pd.Series(
            {"AAPL": 140.0, "MSFT": 250.0}, index=["AAPL", "MSFT"]
        )
        
        # Mock apply_take_profit
        # AAPL liquidated (0.0), MSFT kept (0.5)
        mock_take_profit_handler.apply_take_profit.return_value = pd.Series(
            {"AAPL": 0.0, "MSFT": 0.5}, index=["AAPL", "MSFT"]
        )
        
        result = monitor.check_positions_for_take_profit(
            current_date, mock_position_tracker, current_prices, mock_take_profit_handler, historical_data
        )
        
        assert not result.empty
        assert "AAPL" in result.columns
        assert result.loc[current_date, "AAPL"] == 0.0
        assert "MSFT" not in result.columns
        
        assert "AAPL" in monitor.triggered_positions
        assert monitor.triggered_positions["AAPL"] == current_date

    def test_tracker_missing_attributes(self, monitor, mock_take_profit_handler):
        tracker = MagicMock()
        del tracker.current_positions # Simulate missing attribute
        
        current_date = pd.Timestamp("2020-01-02")
        
        result = monitor.check_positions_for_take_profit(
            current_date, tracker, pd.Series(), mock_take_profit_handler, pd.DataFrame()
        )
        
        assert result.empty
