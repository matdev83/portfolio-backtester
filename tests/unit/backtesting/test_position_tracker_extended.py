import pytest
import pandas as pd
import numpy as np
from portfolio_backtester.backtesting.position_tracker import PositionTracker, Position, Trade

class TestPositionTrackerExtended:
    """Extended test cases for PositionTracker class."""
    
    @pytest.fixture
    def tracker(self):
        return PositionTracker()
    
    def test_partial_exit(self, tracker):
        """Test partial exit of a position."""
        dates = pd.DatetimeIndex(["2025-01-01", "2025-01-02", "2025-01-03"])
        
        # Day 1: Open full position
        signals_open = pd.DataFrame({"TLT": [1.0]}, index=[dates[0]])
        tracker.update_positions(signals_open, dates[0])
        
        assert tracker.current_positions["TLT"].weight == 1.0
        
        # Day 2: Reduce position to 0.5 (Partial Exit - technically an adjustment in this tracker)
        # Note: The current PositionTracker logic treats adjustments as updating the weight,
        # but does NOT create a Trade record for the partial closure.
        # It only creates a Trade record when weight goes to 0 (or crosses 0, effectively close & reopen).
        signals_partial = pd.DataFrame({"TLT": [0.5]}, index=[dates[1]])
        tracker.update_positions(signals_partial, dates[1])
        
        assert tracker.current_positions["TLT"].weight == 0.5
        # No trade should be recorded for partial exit yet based on current logic
        assert len(tracker.completed_trades) == 0
        
        # Day 3: Close remainder
        signals_close = pd.DataFrame({"TLT": [0.0]}, index=[dates[2]])
        tracker.update_positions(signals_close, dates[2])
        
        assert "TLT" not in tracker.current_positions
        assert len(tracker.completed_trades) == 1
        
        # The recorded trade covers the *entire* lifecycle from original entry date to final exit date
        # The weight logged is the *current* weight at time of closure (which was 0.5 before closing)
        # Wait, let's verify logic. _close_position uses `position.weight`.
        # When we adjusted to 0.5, we updated position.weight.
        # So the trade record will show entry_weight=0.5 (the weight at time of closure logic).
        # The entry_date remains the original entry date.
        
        trade = tracker.completed_trades[0]
        assert trade.entry_date == dates[0]
        assert trade.exit_date == dates[2]
        assert trade.entry_weight == 0.5 # Updated weight

    def test_flip_position(self, tracker):
        """Test flipping from long to short."""
        dates = pd.DatetimeIndex(["2025-01-01", "2025-01-02"])
        
        # Day 1: Long 1.0
        signals_long = pd.DataFrame({"TLT": [1.0]}, index=[dates[0]])
        tracker.update_positions(signals_long, dates[0])
        
        # Day 2: Short -1.0
        # This is an adjustment in the current logic, unless we explicitly check for sign change?
        # Let's see update_positions logic:
        # It does `current_position.weight = target_weight`
        # It does NOT close and reopen on sign flip.
        signals_short = pd.DataFrame({"TLT": [-1.0]}, index=[dates[1]])
        tracker.update_positions(signals_short, dates[1])
        
        pos = tracker.current_positions["TLT"]
        assert pos.weight == -1.0
        assert pos.entry_date == dates[0] # Maintains original entry date? Yes.
        
        # This confirms current behavior: Sign flips are treated as weight adjustments, not trade lifecycle events.

    def test_duplicate_columns_in_signals(self, tracker):
        """Test handling of duplicate columns in signals DataFrame."""
        date = pd.Timestamp("2025-01-01")
        # DataFrame with duplicate columns
        signals = pd.DataFrame([[1.0, 1.0]], columns=["TLT", "TLT"], index=[date])
        
        # Should handle gracefully (e.g. take first, or error handled)
        # The code has specific handling: `if isinstance(raw_tw, pd.Series): ...`
        weights = tracker.update_positions(signals, date)
        
        assert "TLT" in tracker.current_positions
        assert tracker.current_positions["TLT"].weight == 1.0
        assert len(weights) == 1 # Result series should handle duplicates by deduplication or last-write?
        # Actually current_weights dict construction overwrites keys, so we get one entry.

    def test_scalar_extraction_robustness(self, tracker):
        """Test extracting numeric scalars from various inputs."""
        date = pd.Timestamp("2025-01-01")
        
        # Case 1: Series with single value
        signals_series = pd.DataFrame({"A": [1.0]}, index=[date])
        # Force the accessor to return a Series-like object if possible, or just standard DF behavior
        
        tracker.update_positions(signals_series, date)
        assert tracker.current_positions["A"].weight == 1.0
        
        # Case 2: NaN handling
        signals_nan = pd.DataFrame({"B": [np.nan]}, index=[date])
        tracker.update_positions(signals_nan, date)
        # Nan -> 0.0 -> Close/Do nothing
        assert "B" not in tracker.current_positions

    def test_get_price_robustness(self, tracker):
        """Test _get_price with various DataFrame structures."""
        date = pd.Timestamp("2025-01-01")
        
        # Single level
        prices_single = pd.DataFrame({"A": [100.0]}, index=[date])
        p = tracker._get_price(prices_single, date, "A")
        assert p == 100.0
        
        p_missing = tracker._get_price(prices_single, date, "B")
        assert p_missing is None
        
        # Multi level
        cols = pd.MultiIndex.from_tuples([("A", "Close"), ("B", "Open")])
        prices_multi = pd.DataFrame([[100.0, 50.0]], index=[date], columns=cols)
        
        p_multi = tracker._get_price(prices_multi, date, "A")
        assert p_multi == 100.0
        
        p_wrong_field = tracker._get_price(prices_multi, date, "B") # Only "Open" available
        assert p_wrong_field is None

    def test_trade_pnl_with_missing_prices(self, tracker):
        """Test trade creation when price data is missing for entry or exit."""
        dates = pd.DatetimeIndex(["2025-01-01", "2025-01-02"])
        
        # Entry with price
        prices_entry = pd.DataFrame({"A": [100.0]}, index=[dates[0]])
        signals_entry = pd.DataFrame({"A": [1.0]}, index=[dates[0]])
        tracker.update_positions(signals_entry, dates[0], prices_entry)
        
        # Exit without price
        signals_exit = pd.DataFrame({"A": [0.0]}, index=[dates[1]])
        tracker.update_positions(signals_exit, dates[1], prices=None)
        
        trade = tracker.completed_trades[0]
        assert trade.pnl is None