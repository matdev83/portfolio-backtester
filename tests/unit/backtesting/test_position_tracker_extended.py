import pytest
import pandas as pd
import numpy as np
from src.portfolio_backtester.backtesting.position_tracker import PositionTracker, Position, Trade

class TestPositionTracker:
    @pytest.fixture
    def tracker(self):
        return PositionTracker()

    @pytest.fixture
    def sample_prices(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        prices = pd.DataFrame({
            "AAPL": [100.0, 101.0, 102.0, 103.0, 102.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            "MSFT": [200.0, 202.0, 204.0, 201.0, 205.0, 208.0, 210.0, 212.0, 215.0, 218.0]
        }, index=dates)
        return prices

    def test_open_position(self, tracker, sample_prices):
        date = sample_prices.index[0]
        signals = pd.DataFrame({"AAPL": [0.5]}, index=[date])
        
        weights = tracker.update_positions(signals, date, sample_prices)
        
        assert "AAPL" in tracker.current_positions
        pos = tracker.current_positions["AAPL"]
        assert pos.weight == 0.5
        assert pos.entry_price == 100.0
        assert pos.entry_date == date
        
        assert weights["AAPL"] == 0.5

    def test_close_position_and_pnl(self, tracker, sample_prices):
        # Open
        date_open = sample_prices.index[0]
        tracker.update_positions(pd.DataFrame({"AAPL": [0.5]}, index=[date_open]), date_open, sample_prices)
        
        # Close
        date_close = sample_prices.index[5] # Price 105.0
        # Signal 0.0 means close
        tracker.update_positions(pd.DataFrame({"AAPL": [0.0]}, index=[date_close]), date_close, sample_prices)
        
        assert "AAPL" not in tracker.current_positions
        assert len(tracker.completed_trades) == 1
        
        trade = tracker.completed_trades[0]
        assert trade.ticker == "AAPL"
        assert trade.entry_date == date_open
        assert trade.exit_date == date_close
        assert trade.entry_weight == 0.5
        assert trade.exit_weight == 0.0
        
        # PnL = (Exit - Entry) / Entry * Weight
        # (105 - 100) / 100 * 0.5 = 0.05 * 0.5 = 0.025
        assert trade.pnl == pytest.approx(0.025)
        
        # Duration: business days from index[0] to index[5] is 5 days diff
        # bdate_range includes both ends, so length is 6, -1 = 5
        assert trade.duration_days == 5

    def test_adjust_position(self, tracker, sample_prices):
        date_1 = sample_prices.index[0]
        tracker.update_positions(pd.DataFrame({"AAPL": [0.5]}, index=[date_1]), date_1, sample_prices)
        
        date_2 = sample_prices.index[2]
        tracker.update_positions(pd.DataFrame({"AAPL": [0.7]}, index=[date_2]), date_2, sample_prices)
        
        assert tracker.current_positions["AAPL"].weight == 0.7
        # Entry price/date should remain from INITIAL entry (FIFO/Average logic or just update? implementation updates weight only)
        # Looking at code: current_position.weight = target_weight. It does NOT update entry_price/date.
        assert tracker.current_positions["AAPL"].entry_price == 100.0
        assert tracker.current_positions["AAPL"].entry_date == date_1

    def test_daily_weights_tracking(self, tracker, sample_prices):
        date_1 = sample_prices.index[0]
        tracker.update_positions(pd.DataFrame({"AAPL": [0.5]}, index=[date_1]), date_1, sample_prices)
        
        date_2 = sample_prices.index[1]
        tracker.update_positions(pd.DataFrame({"AAPL": [0.5]}, index=[date_2]), date_2, sample_prices)
        
        df = tracker.get_daily_weights_df()
        assert len(df) == 2
        assert "AAPL" in df.columns
        assert df.iloc[0]["AAPL"] == 0.5
        assert df.iloc[1]["AAPL"] == 0.5

    def test_missing_prices(self, tracker):
        date = pd.Timestamp("2020-01-01")
        signals = pd.DataFrame({"AAPL": [0.5]}, index=[date])
        # Pass None for prices
        tracker.update_positions(signals, date, None)
        
        pos = tracker.current_positions["AAPL"]
        assert pos.entry_price is None
        
        # Close without price
        tracker.update_positions(pd.DataFrame({"AAPL": [0.0]}, index=[date]), date, None)
        trade = tracker.completed_trades[0]
        assert trade.pnl is None

    def test_multiindex_prices(self, tracker):
        date = pd.Timestamp("2020-01-01")
        # MultiIndex columns (Ticker, Field)
        cols = pd.MultiIndex.from_product([["AAPL"], ["Close", "Open"]])
        prices = pd.DataFrame([[150.0, 148.0]], index=[date], columns=cols)
        
        signals = pd.DataFrame({"AAPL": [1.0]}, index=[date])
        tracker.update_positions(signals, date, prices)
        
        assert tracker.current_positions["AAPL"].entry_price == 150.0

    def test_trade_summary(self, tracker, sample_prices):
        # Create a profitable trade
        date_open = sample_prices.index[0]
        tracker.update_positions(pd.DataFrame({"AAPL": [1.0]}, index=[date_open]), date_open, sample_prices)
        
        date_close = sample_prices.index[1] # 100 -> 101 (+1%)
        tracker.update_positions(pd.DataFrame({"AAPL": [0.0]}, index=[date_close]), date_close, sample_prices)
        
        summary = tracker.get_trade_summary()
        assert summary["total_trades"] == 1
        assert summary["avg_pnl"] == pytest.approx(0.01)
        assert summary["max_duration"] == 1.0
