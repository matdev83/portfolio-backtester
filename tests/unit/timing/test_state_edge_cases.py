"""
Tests for edge cases and error conditions in state management.
Split from test_advanced_state_management.py for better organization.
"""

import pytest
import pandas as pd
from portfolio_backtester.interfaces.timing_state_interface import create_timing_state


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_state_operations(self):
        """Test operations on empty state."""
        state = create_timing_state()

        # Should not raise exceptions
        summary = state.get_portfolio_summary()
        assert summary["total_positions"] == 0

        stats = state.get_position_statistics()
        assert stats["total_trades"] == 0

        assert state.get_position_return("AAPL") is None
        assert state.get_position_info("AAPL") is None
        assert state.get_consecutive_periods("AAPL") == 0

    def test_invalid_position_operations(self):
        """Test operations on non-existent positions."""
        state = create_timing_state()
        current_date = pd.Timestamp("2023-01-01")

        assert state.get_position_holding_days("NONEXISTENT", current_date) is None
        assert not state.is_position_held("NONEXISTENT")
        assert state.get_position_return("NONEXISTENT") is None
        assert state.get_position_info("NONEXISTENT") is None

    def test_zero_price_handling(self):
        """Test handling of zero or invalid prices."""
        state = create_timing_state()
        test_date = pd.Timestamp("2023-01-01")
        weights = pd.Series([0.5], index=["AAPL"])
        prices = pd.Series([0.0], index=["AAPL"])  # Zero price

        state.update_positions(test_date, weights, prices)

        # Should handle gracefully
        position = state.positions["AAPL"]
        assert position.entry_price == 0.0

        # Return calculation should handle zero price
        assert state.get_position_return("AAPL", 100.0) is None


class TestTimingState:
    """Test basic TimingState functionality."""

    def test_initialization(self):
        """Test TimingState initialization."""
        state = create_timing_state()

        assert state.state_version == "1.0"
        assert state.last_signal_date is None
        assert len(state.positions) == 0
        assert len(state.position_history) == 0
        assert not state.debug_enabled
        assert len(state.debug_log) == 0

        # Legacy compatibility
        assert len(state.position_entry_dates) == 0
        assert len(state.position_entry_prices) == 0
        assert len(state.consecutive_periods) == 0

    def test_signal_update(self):
        """Test signal update functionality."""
        state = create_timing_state()
        test_date = pd.Timestamp("2023-01-01")
        weights = pd.Series([0.5, 0.5], index=["AAPL", "MSFT"])

        state.update_signal(test_date, weights)

        assert state.last_signal_date == test_date
        # Note: TimingState may not store last_signal_weights, just the date

    def test_basic_position_operations(self):
        """Test basic position operations."""
        state = create_timing_state()
        test_date = pd.Timestamp("2023-01-01")
        weights = pd.Series([0.6, 0.4], index=["AAPL", "MSFT"])
        prices = pd.Series([150.0, 250.0], index=["AAPL", "MSFT"])

        state.update_positions(test_date, weights, prices)

        # Test basic queries
        assert state.is_position_held("AAPL")
        assert state.is_position_held("MSFT")
        assert not state.is_position_held("GOOGL")

        held_assets = state.get_held_assets()
        assert len(held_assets) == 2
        assert "AAPL" in held_assets
        assert "MSFT" in held_assets

        # Test position info
        aapl_info = state.get_position_info("AAPL")
        assert aapl_info is not None
        assert aapl_info.entry_price == 150.0
        assert aapl_info.current_weight == 0.6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
