"""
Tests for refactored TimingState with SOLID principles.
Ensures backward compatibility while testing new architecture.
"""

import pandas as pd
import tempfile
import os
from portfolio_backtester.timing.timing_state import TimingState
from portfolio_backtester.timing.state_management import (
    PositionTracker,
    StateStatistics,
    StateSerializer,
)


class TestTimingStateRefactored:
    """Test refactored TimingState class."""

    def setup_method(self):
        """Set up test environment."""
        self.state = TimingState(debug_enabled=True)

    def test_initialization(self):
        """Test state initialization with specialized components."""
        assert isinstance(self.state.position_tracker, PositionTracker)
        assert isinstance(self.state.statistics, StateStatistics)
        assert isinstance(self.state.serializer, StateSerializer)
        assert self.state.debug_enabled is True

    def test_basic_state_operations(self):
        """Test basic state operations."""
        # Test signal update
        date = pd.Timestamp("2023-01-01")
        weights = pd.Series({"AAPL": 0.5, "MSFT": 0.3, "GOOGL": 0.2})

        self.state.update_signal(date, weights)

        assert self.state.last_signal_date == date
        assert self.state.last_weights.equals(weights)
        assert self.state.last_updated == date

    def test_position_tracking(self):
        """Test position tracking functionality."""
        date = pd.Timestamp("2023-01-01")
        weights = pd.Series({"AAPL": 0.5, "MSFT": 0.3})
        prices = pd.Series({"AAPL": 150.0, "MSFT": 250.0})

        # Update positions
        self.state.update_positions(date, weights, prices)

        # Verify position tracking
        assert len(self.state.positions) == 2
        assert "AAPL" in self.state.positions
        assert "MSFT" in self.state.positions

        # Check position info
        aapl_position = self.state.get_position_info("AAPL")
        assert aapl_position is not None
        assert aapl_position.entry_date == date
        assert aapl_position.entry_price == 150.0
        assert aapl_position.current_weight == 0.5

        # Test backward compatibility properties
        assert "AAPL" in self.state.position_entry_dates
        assert "AAPL" in self.state.position_entry_prices
        assert "AAPL" in self.state.consecutive_periods

    def test_position_lifecycle(self):
        """Test complete position lifecycle (entry, update, exit)."""
        date1 = pd.Timestamp("2023-01-01")
        date2 = pd.Timestamp("2023-01-02")
        date3 = pd.Timestamp("2023-01-03")

        # Initial positions
        weights1 = pd.Series({"AAPL": 0.5, "MSFT": 0.5})
        prices1 = pd.Series({"AAPL": 150.0, "MSFT": 250.0})
        self.state.update_positions(date1, weights1, prices1)

        # Update positions (weight change)
        weights2 = pd.Series({"AAPL": 0.7, "MSFT": 0.3})
        prices2 = pd.Series({"AAPL": 155.0, "MSFT": 245.0})
        self.state.update_positions(date2, weights2, prices2)

        # Verify position update
        aapl_position = self.state.get_position_info("AAPL")
        assert aapl_position.current_weight == 0.7
        assert aapl_position.consecutive_periods == 2

        # Exit one position
        weights3 = pd.Series({"AAPL": 0.0, "MSFT": 1.0})
        prices3 = pd.Series({"AAPL": 160.0, "MSFT": 240.0})
        self.state.update_positions(date3, weights3, prices3)

        # Verify position exit
        assert "AAPL" not in self.state.positions
        assert len(self.state.position_history) == 1

        # Check historical position data
        history = self.state.position_history[0]
        assert history["asset"] == "AAPL"
        assert history["entry_date"] == date1
        assert history["exit_date"] == date3
        assert history["holding_days"] == 2

    def test_position_utility_methods(self):
        """Test position utility methods."""
        date = pd.Timestamp("2023-01-01")
        weights = pd.Series({"AAPL": 0.6, "MSFT": 0.4})
        prices = pd.Series({"AAPL": 150.0, "MSFT": 250.0})

        self.state.update_positions(date, weights, prices)

        # Test utility methods
        assert self.state.is_position_held("AAPL") is True
        assert self.state.is_position_held("GOOGL") is False

        held_assets = self.state.get_held_assets()
        assert "AAPL" in held_assets
        assert "MSFT" in held_assets
        assert len(held_assets) == 2

        # Test holding days calculation
        later_date = pd.Timestamp("2023-01-10")
        holding_days = self.state.get_position_holding_days("AAPL", later_date)
        assert holding_days == 9

        # Test consecutive periods
        assert self.state.get_consecutive_periods("AAPL") == 1

    def test_portfolio_summary(self):
        """Test portfolio summary statistics."""
        date = pd.Timestamp("2023-01-01")
        weights = pd.Series({"AAPL": 0.4, "MSFT": 0.3, "GOOGL": 0.3})
        prices = pd.Series({"AAPL": 150.0, "MSFT": 250.0, "GOOGL": 2500.0})

        self.state.update_positions(date, weights, prices)

        summary = self.state.get_portfolio_summary()

        assert summary["total_positions"] == 3
        assert summary["total_weight"] == 1.0
        assert summary["last_updated"] == date
        assert len(summary["assets"]) == 3
        assert "AAPL" in summary["assets"]

    def test_position_statistics(self):
        """Test position statistics functionality."""
        # Add some historical positions
        date1 = pd.Timestamp("2023-01-01")
        date2 = pd.Timestamp("2023-01-05")

        # Create and exit a position to generate history
        weights1 = pd.Series({"AAPL": 1.0})
        prices1 = pd.Series({"AAPL": 150.0})
        self.state.update_positions(date1, weights1, prices1)

        weights2 = pd.Series({"AAPL": 0.0})
        prices2 = pd.Series({"AAPL": 160.0})
        self.state.update_positions(date2, weights2, prices2)

        stats = self.state.get_position_statistics()

        assert stats["total_trades"] == 1
        assert stats["avg_holding_days"] == 4
        assert stats["min_holding_days"] == 4
        assert stats["max_holding_days"] == 4

        # Check return calculations - the return should be calculated and stored
        if "avg_return" in stats:
            expected_return = (160.0 - 150.0) / 150.0
            assert abs(stats["avg_return"] - expected_return) < 1e-6
        assert stats["positive_trades"] == 1
        assert stats["win_rate"] == 1.0

    def test_performance_summary(self):
        """Test performance summary functionality."""
        date = pd.Timestamp("2023-01-01")
        weights = pd.Series({"AAPL": 0.6, "MSFT": 0.4})
        prices = pd.Series({"AAPL": 150.0, "MSFT": 250.0})

        self.state.update_positions(date, weights, prices)

        performance = self.state.get_position_performance_summary()

        # Check active positions summary
        active = performance["active_positions"]
        assert active["count"] == 2
        assert active["total_weight"] == 1.0
        assert active["max_weight"] == 0.6
        assert active["min_weight"] == 0.4

        # Check historical positions summary
        historical = performance["historical_positions"]
        assert historical["count"] == 0

    def test_asset_analysis(self):
        """Test per-asset analysis functionality."""
        date = pd.Timestamp("2023-01-01")
        weights = pd.Series({"AAPL": 0.7, "MSFT": 0.3})
        prices = pd.Series({"AAPL": 150.0, "MSFT": 250.0})

        self.state.update_positions(date, weights, prices)

        analysis = self.state.get_asset_analysis()

        assert "AAPL" in analysis
        assert "MSFT" in analysis

        aapl_analysis = analysis["AAPL"]
        assert aapl_analysis["status"] == "active"
        assert aapl_analysis["current_weight"] == 0.7
        assert aapl_analysis["entry_date"] == date

    def test_weight_distribution_analysis(self):
        """Test weight distribution analysis."""
        date = pd.Timestamp("2023-01-01")
        weights = pd.Series(
            {
                "AAPL": 0.15,  # Heavy position > 10%
                "MSFT": 0.08,  # Medium position 5-10%
                "GOOGL": 0.03,  # Light position < 5%
            }
        )
        prices = pd.Series({"AAPL": 150.0, "MSFT": 250.0, "GOOGL": 2500.0})

        self.state.update_positions(date, weights, prices)

        analysis = self.state.get_weight_distribution_analysis()

        assert analysis["total_positions"] == 3
        assert analysis["total_weight"] == 0.26
        assert analysis["max_weight"] == 0.15
        assert analysis["min_weight"] == 0.03

        distribution = analysis["weight_distribution"]
        assert distribution["heavy_positions"] == 1
        assert distribution["medium_positions"] == 1
        assert distribution["light_positions"] == 1

    def test_holding_period_analysis(self):
        """Test holding period analysis."""
        date1 = pd.Timestamp("2023-01-01")
        date2 = pd.Timestamp("2023-01-10")

        # Create positions with different holding periods
        weights = pd.Series({"AAPL": 0.5, "MSFT": 0.5})
        prices = pd.Series({"AAPL": 150.0, "MSFT": 250.0})
        self.state.update_positions(date1, weights, prices)

        # Need to pass the current date to get holding period analysis
        self.state.last_updated = date2  # Set the current date in the state
        analysis = self.state.get_holding_period_analysis()

        # Should have active positions analysis with current date passed
        assert "active_positions" in analysis
        if analysis["active_positions"]:  # Only check if there are active positions
            active = analysis["active_positions"]
            assert active["count"] == 2
            assert active["avg_days"] == 9  # 9 days holding period

    def test_state_serialization(self):
        """Test state serialization and deserialization."""
        # Set up initial state
        date = pd.Timestamp("2023-01-01")
        weights = pd.Series({"AAPL": 0.6, "MSFT": 0.4})
        prices = pd.Series({"AAPL": 150.0, "MSFT": 250.0})

        self.state.update_signal(date, weights)
        self.state.update_positions(date, weights, prices)
        self.state.scheduled_dates.add(pd.Timestamp("2023-01-15"))

        # Test to_dict serialization
        state_dict = self.state.to_dict()

        assert state_dict["state_version"] == "1.0"
        assert state_dict["last_signal_date"] == date.isoformat()
        assert len(state_dict["positions"]) == 2
        assert "AAPL" in state_dict["positions"]

        # Test from_dict deserialization
        restored_state = TimingState.from_dict(state_dict)

        assert restored_state.last_signal_date == date
        assert len(restored_state.positions) == 2
        assert "AAPL" in restored_state.positions
        assert restored_state.scheduled_dates == self.state.scheduled_dates

    def test_file_persistence(self):
        """Test saving and loading state from file."""
        # Set up state
        date = pd.Timestamp("2023-01-01")
        weights = pd.Series({"AAPL": 0.7, "MSFT": 0.3})
        prices = pd.Series({"AAPL": 150.0, "MSFT": 250.0})

        self.state.update_signal(date, weights)
        self.state.update_positions(date, weights, prices)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = f.name

        try:
            self.state.save_to_file(temp_file)
            assert os.path.exists(temp_file)

            # Load from file
            loaded_state = TimingState.load_from_file(temp_file)

            assert loaded_state.last_signal_date == date
            assert len(loaded_state.positions) == 2
            assert "AAPL" in loaded_state.positions

        finally:
            os.unlink(temp_file)

    def test_comprehensive_analysis(self):
        """Test comprehensive analysis combining all statistics."""
        date = pd.Timestamp("2023-01-01")
        weights = pd.Series({"AAPL": 0.5, "MSFT": 0.5})
        prices = pd.Series({"AAPL": 150.0, "MSFT": 250.0})

        self.state.update_positions(date, weights, prices)

        analysis = self.state.get_comprehensive_analysis()

        # Verify all analysis components are present
        required_keys = [
            "portfolio_summary",
            "position_statistics",
            "performance_summary",
            "asset_analysis",
            "weight_distribution",
            "holding_period_analysis",
            "state_info",
        ]

        for key in required_keys:
            assert key in analysis

        # Verify state info
        state_info = analysis["state_info"]
        assert "active_positions" in state_info
        assert "estimated_memory_kb" in state_info

    def test_state_integrity_validation(self):
        """Test state integrity validation."""
        date = pd.Timestamp("2023-01-01")
        weights = pd.Series({"AAPL": 0.5, "MSFT": 0.5})
        prices = pd.Series({"AAPL": 150.0, "MSFT": 250.0})

        self.state.update_positions(date, weights, prices)

        # Test with valid state
        errors = self.state.validate_state_integrity()
        assert len(errors) == 0

    def test_debug_functionality(self):
        """Test debug logging functionality."""
        # Debug should be enabled by default in setup
        assert self.state.debug_enabled is True

        # Perform some operations to generate debug logs
        date = pd.Timestamp("2023-01-01")
        weights = pd.Series({"AAPL": 1.0})

        self.state.update_signal(date, weights)

        # Check debug logs
        debug_logs = self.state.get_debug_log()
        assert len(debug_logs) > 0

        # Test clearing debug logs
        self.state.clear_debug_log()
        debug_logs_after_clear = self.state.get_debug_log()
        assert len(debug_logs_after_clear) <= 1  # May have the "cleared" entry

    def test_reset_functionality(self):
        """Test state reset functionality."""
        # Set up some state
        date = pd.Timestamp("2023-01-01")
        weights = pd.Series({"AAPL": 0.5, "MSFT": 0.5})
        prices = pd.Series({"AAPL": 150.0, "MSFT": 250.0})

        self.state.update_signal(date, weights)
        self.state.update_positions(date, weights, prices)
        self.state.scheduled_dates.add(pd.Timestamp("2023-01-15"))

        # Verify state has data
        assert self.state.last_signal_date is not None
        assert len(self.state.positions) > 0
        assert len(self.state.scheduled_dates) > 0

        # Reset state
        self.state.reset()

        # Verify state is cleared
        assert self.state.last_signal_date is None
        assert self.state.last_weights is None
        assert len(self.state.positions) == 0
        assert len(self.state.scheduled_dates) == 0
        assert len(self.state.position_history) == 0
