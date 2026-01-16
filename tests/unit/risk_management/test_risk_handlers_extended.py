import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.portfolio_backtester.risk_management.stop_loss_handlers import AtrBasedStopLoss, NoStopLoss
from src.portfolio_backtester.risk_management.take_profit_handlers import AtrBasedTakeProfit, NoTakeProfit

class TestStopLossHandlers:
    @pytest.fixture
    def strategy_config(self):
        return {}

    def test_no_stop_loss(self, strategy_config):
        handler = NoStopLoss(strategy_config, {})
        current_date = pd.Timestamp("2020-01-01")
        current_weights = pd.Series({"AAPL": 0.5}, index=["AAPL"])
        entry_prices = pd.Series({"AAPL": 100.0}, index=["AAPL"])
        
        levels = handler.calculate_stop_levels(current_date, pd.DataFrame(), current_weights, entry_prices)
        assert levels.isna().all()
        
        adjusted = handler.apply_stop_loss(
            current_date, pd.Series({"AAPL": 90.0}), current_weights, entry_prices, levels
        )
        pd.testing.assert_series_equal(adjusted, current_weights)

    @patch('src.portfolio_backtester.risk_management.stop_loss_handlers.calculate_atr_fast')
    def test_atr_stop_loss_long(self, mock_atr, strategy_config):
        # Setup
        sl_config = {"atr_length": 14, "atr_multiple": 2.0}
        handler = AtrBasedStopLoss(strategy_config, sl_config)
        
        current_date = pd.Timestamp("2020-01-01")
        # Long position in AAPL
        current_weights = pd.Series({"AAPL": 0.5}, index=["AAPL"])
        entry_prices = pd.Series({"AAPL": 100.0}, index=["AAPL"])
        
        # Mock ATR return: ATR = 5.0
        mock_atr.return_value = pd.Series({"AAPL": 5.0}, index=["AAPL"])
        
        # 1. Calculate Levels
        # Stop should be Entry - (ATR * Multiple) = 100 - (5 * 2) = 90
        levels = handler.calculate_stop_levels(current_date, pd.DataFrame(), current_weights, entry_prices)
        
        assert levels["AAPL"] == 90.0
        
        # 2. Apply Stop Loss - Price above stop
        current_prices_safe = pd.Series({"AAPL": 95.0}, index=["AAPL"])
        adjusted_safe = handler.apply_stop_loss(
            current_date, current_prices_safe, current_weights, entry_prices, levels
        )
        assert adjusted_safe["AAPL"] == 0.5 # Not triggered
        
        # 3. Apply Stop Loss - Price below stop
        current_prices_triggered = pd.Series({"AAPL": 89.0}, index=["AAPL"])
        adjusted_triggered = handler.apply_stop_loss(
            current_date, current_prices_triggered, current_weights, entry_prices, levels
        )
        assert adjusted_triggered["AAPL"] == 0.0 # Triggered

    @patch('src.portfolio_backtester.risk_management.stop_loss_handlers.calculate_atr_fast')
    def test_atr_stop_loss_short(self, mock_atr, strategy_config):
        sl_config = {"atr_length": 14, "atr_multiple": 2.0}
        handler = AtrBasedStopLoss(strategy_config, sl_config)
        
        current_date = pd.Timestamp("2020-01-01")
        # Short position in TSLA
        current_weights = pd.Series({"TSLA": -0.5}, index=["TSLA"])
        entry_prices = pd.Series({"TSLA": 200.0}, index=["TSLA"])
        
        # ATR = 10.0
        mock_atr.return_value = pd.Series({"TSLA": 10.0}, index=["TSLA"])
        
        # Stop should be Entry + (ATR * Multiple) = 200 + (10 * 2) = 220
        levels = handler.calculate_stop_levels(current_date, pd.DataFrame(), current_weights, entry_prices)
        assert levels["TSLA"] == 220.0
        
        # Triggered if price goes ABOVE 220
        current_prices = pd.Series({"TSLA": 225.0}, index=["TSLA"])
        adjusted = handler.apply_stop_loss(
            current_date, current_prices, current_weights, entry_prices, levels
        )
        assert adjusted["TSLA"] == 0.0

    @patch('src.portfolio_backtester.risk_management.stop_loss_handlers.calculate_atr_fast')
    def test_atr_fallback(self, mock_atr, strategy_config):
        # Test fallback when ATR is NaN
        sl_config = {"atr_length": 14, "atr_multiple": 2.0}
        handler = AtrBasedStopLoss(strategy_config, sl_config)
        
        current_date = pd.Timestamp("2020-01-01")
        current_weights = pd.Series({"AAPL": 0.5}, index=["AAPL"])
        entry_prices = pd.Series({"AAPL": 100.0}, index=["AAPL"])
        
        # Return NaN for ATR
        mock_atr.return_value = pd.Series({"AAPL": np.nan}, index=["AAPL"])
        
        # Fallback logic: ATR = Entry * 0.02 = 100 * 0.02 = 2.0
        # Stop = 100 - (2.0 * 2.0) = 96.0
        levels = handler.calculate_stop_levels(current_date, pd.DataFrame(), current_weights, entry_prices)
        assert levels["AAPL"] == 96.0


class TestTakeProfitHandlers:
    @pytest.fixture
    def strategy_config(self):
        return {}

    def test_no_take_profit(self, strategy_config):
        handler = NoTakeProfit(strategy_config, {})
        current_date = pd.Timestamp("2020-01-01")
        current_weights = pd.Series({"AAPL": 0.5}, index=["AAPL"])
        entry_prices = pd.Series({"AAPL": 100.0}, index=["AAPL"])
        
        levels = handler.calculate_take_profit_levels(current_date, pd.DataFrame(), current_weights, entry_prices)
        assert levels.isna().all()

    @patch('src.portfolio_backtester.risk_management.take_profit_handlers.calculate_atr_fast')
    def test_atr_take_profit_long(self, mock_atr, strategy_config):
        tp_config = {"atr_length": 14, "atr_multiple": 3.0}
        handler = AtrBasedTakeProfit(strategy_config, tp_config)
        
        current_date = pd.Timestamp("2020-01-01")
        current_weights = pd.Series({"AAPL": 0.5}, index=["AAPL"])
        entry_prices = pd.Series({"AAPL": 100.0}, index=["AAPL"])
        
        mock_atr.return_value = pd.Series({"AAPL": 5.0}, index=["AAPL"])
        
        # Target = Entry + (ATR * Multiple) = 100 + (5 * 3) = 115
        levels = handler.calculate_take_profit_levels(current_date, pd.DataFrame(), current_weights, entry_prices)
        assert levels["AAPL"] == 115.0
        
        # Check trigger
        current_prices = pd.Series({"AAPL": 116.0}, index=["AAPL"])
        adjusted = handler.apply_take_profit(
            current_date, current_prices, current_weights, entry_prices, levels
        )
        assert adjusted["AAPL"] == 0.0

    @patch('src.portfolio_backtester.risk_management.take_profit_handlers.calculate_atr_fast')
    def test_atr_take_profit_short(self, mock_atr, strategy_config):
        tp_config = {"atr_length": 14, "atr_multiple": 3.0}
        handler = AtrBasedTakeProfit(strategy_config, tp_config)
        
        current_date = pd.Timestamp("2020-01-01")
        current_weights = pd.Series({"TSLA": -0.5}, index=["TSLA"])
        entry_prices = pd.Series({"TSLA": 200.0}, index=["TSLA"])
        
        mock_atr.return_value = pd.Series({"TSLA": 10.0}, index=["TSLA"])
        
        # Target = Entry - (ATR * Multiple) = 200 - (10 * 3) = 170
        levels = handler.calculate_take_profit_levels(current_date, pd.DataFrame(), current_weights, entry_prices)
        assert levels["TSLA"] == 170.0
        
        # Check trigger (price below target)
        current_prices = pd.Series({"TSLA": 165.0}, index=["TSLA"])
        adjusted = handler.apply_take_profit(
            current_date, current_prices, current_weights, entry_prices, levels
        )
        assert adjusted["TSLA"] == 0.0
