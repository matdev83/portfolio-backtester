"""
Tests for state persistence functionality.
Split from test_advanced_state_management.py for better organization.
"""

import pytest
import pandas as pd
import tempfile
import os
from portfolio_backtester.interfaces.timing_state_interface import create_timing_state
from portfolio_backtester.timing.timing_state import TimingState


class TestStatePersistence:
    """Test state persistence functionality."""

    def setup_method(self):
        """Set up test data."""
        self.state = create_timing_state()
        self.test_date = pd.Timestamp("2023-01-01")

        # Add some test data
        weights = pd.Series([0.4, 0.3, 0.3], index=["AAPL", "MSFT", "GOOGL"])
        prices = pd.Series([150.0, 250.0, 2500.0], index=["AAPL", "MSFT", "GOOGL"])

        self.state.update_signal(self.test_date, weights)
        self.state.update_positions(self.test_date, weights, prices)
        self.state.enable_debug(True)

        # Add some history
        self.state.position_history.append(
            {
                "asset": "TSLA",
                "entry_date": self.test_date - pd.Timedelta(days=5),
                "exit_date": self.test_date - pd.Timedelta(days=1),
                "holding_days": 4,
                "total_return": 0.1,
            }
        )

    def test_state_serialization(self):
        """Test state serialization to dictionary."""
        state_dict = self.state.to_dict()

        assert state_dict["state_version"] == "1.0"
        assert state_dict["last_signal_date"] == self.test_date.isoformat()
        assert state_dict["debug_enabled"]
        assert len(state_dict["positions"]) == 3
        assert len(state_dict["position_history"]) == 1

        # Check position data
        aapl_data = state_dict["positions"]["AAPL"]
        assert aapl_data["entry_date"] == self.test_date.isoformat()
        assert aapl_data["entry_price"] == 150.0
        assert aapl_data["entry_weight"] == 0.4

    def test_state_deserialization(self):
        """Test state deserialization from dictionary."""
        # Serialize first
        state_dict = self.state.to_dict()

        # Deserialize
        restored_state = TimingState.from_dict(state_dict)

        # Verify restoration
        assert restored_state.state_version == "1.0"
        assert restored_state.last_signal_date == self.test_date
        assert restored_state.debug_enabled
        assert len(restored_state.positions) == 3
        assert len(restored_state.position_history) == 1

        # Check position restoration
        aapl_position = restored_state.positions["AAPL"]
        assert aapl_position.entry_date == self.test_date
        assert aapl_position.entry_price == 150.0
        assert aapl_position.entry_weight == 0.4

        # Check legacy compatibility restoration
        assert len(restored_state.position_entry_dates) == 3
        assert restored_state.position_entry_dates["AAPL"] == self.test_date

    def test_file_persistence(self):
        """Test saving and loading state from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = f.name

        try:
            # Save to file
            self.state.save_to_file(temp_file)

            # Load from file
            restored_state = TimingState.load_from_file(temp_file)

            # Verify restoration
            assert restored_state.last_signal_date == self.test_date
            assert len(restored_state.positions) == 3
            assert len(restored_state.position_history) == 1

            # Verify position data
            aapl_position = restored_state.positions["AAPL"]
            assert aapl_position.entry_price == 150.0

        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
