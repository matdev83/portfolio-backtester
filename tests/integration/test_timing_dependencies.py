"""
Integration tests for timing system dependency inversion.

Tests that the interfaces work correctly and dependency injection is functioning properly.
"""

import pandas as pd

from portfolio_backtester.interfaces.timing_state_interface import (
    create_timing_state,
    ITimingState,
)
from portfolio_backtester.interfaces.time_based_timing_interface import (
    create_time_based_timing,
    ITimeBasedTiming,
)
from portfolio_backtester.timing.time_based_timing import TimeBasedTiming
from portfolio_backtester.timing.signal_based_timing import SignalBasedTiming


class TestTimingDependencyInversion:
    """Test timing system dependency inversion implementation."""

    def test_timing_state_interface_creation(self):
        """Test that timing state interface can be created and used."""
        timing_state = create_timing_state()

        assert isinstance(timing_state, ITimingState)
        assert timing_state.last_signal_date is None
        assert timing_state.last_weights is None
        assert len(timing_state.scheduled_dates) == 0

    def test_timing_state_operations(self):
        """Test that timing state interface operations work correctly."""
        timing_state = create_timing_state()

        # Test signal update
        test_date = pd.Timestamp("2023-01-01")
        test_weights = pd.Series({"AAPL": 0.5, "MSFT": 0.5})

        timing_state.update_signal(test_date, test_weights)

        assert timing_state.last_signal_date == test_date
        assert timing_state.last_weights is not None
        pd.testing.assert_series_equal(timing_state.last_weights, test_weights)

    def test_timing_controller_with_dependency_injection(self):
        """Test that TimingController accepts ITimingState through dependency injection."""
        # Create timing state via interface
        timing_state = create_timing_state()

        # Test with TimeBasedTiming
        config = {"rebalance_frequency": "M"}
        time_based = TimeBasedTiming(config, timing_state)

        # Verify the injected timing state is used
        assert time_based.get_timing_state() is timing_state

        # Test reset functionality
        time_based.reset_state()  # This should use the injected timing state

    def test_time_based_timing_interface(self):
        """Test that time-based timing interface works correctly."""
        config = {"rebalance_frequency": "M", "rebalance_offset": 0}
        time_based_timing = create_time_based_timing(config)

        assert isinstance(time_based_timing, ITimeBasedTiming)
        assert time_based_timing.get_frequency() == "M"
        assert time_based_timing.get_offset() == 0

        # Test frequency modification
        time_based_timing.set_frequency("W")
        assert time_based_timing.get_frequency() == "W"

    def test_adaptive_timing_controller_dependency_injection(self):
        """Test that AdaptiveTimingController uses dependency injection correctly."""
        from portfolio_backtester.timing.custom_timing_registry import (
            AdaptiveTimingController,
        )

        config = {"base_frequency": "M", "high_vol_frequency": "W", "low_vol_frequency": "Q"}

        # Create with dependency injection
        time_based_timing = create_time_based_timing(config)
        adaptive = AdaptiveTimingController(
            config, time_based_timing=time_based_timing, volatility_threshold=0.03
        )

        # Verify it uses the injected dependency
        assert adaptive.volatility_threshold == 0.03
        assert adaptive.base_frequency == "M"
        assert adaptive._time_based_timing is time_based_timing

    def test_momentum_timing_controller_dependency_injection(self):
        """Test that MomentumTimingController uses dependency injection correctly."""
        from portfolio_backtester.timing.custom_timing_registry import (
            MomentumTimingController,
        )

        config = {"rebalance_frequency": "W"}

        # Create with dependency injection
        time_based_timing = create_time_based_timing(config)
        momentum = MomentumTimingController(
            config, time_based_timing=time_based_timing, momentum_period=30
        )

        # Verify it uses the injected dependency
        assert momentum.momentum_period == 30
        assert momentum._time_based_timing is time_based_timing

    def test_signal_based_timing_with_interface(self):
        """Test that SignalBasedTiming works with interface dependency injection."""
        config = {"scan_frequency": "D", "min_holding_period": 1, "max_holding_period": 30}

        signal_based = SignalBasedTiming(config)

        # Verify it has a timing state
        assert isinstance(signal_based.get_timing_state(), ITimingState)

        # Test basic functionality
        test_date = pd.Timestamp("2023-01-01")
        assert not signal_based._is_within_min_holding_period(test_date)

    def test_interface_backward_compatibility(self):
        """Test that interfaces maintain backward compatibility."""
        # Test creating instances the old way still works
        config = {"rebalance_frequency": "M"}
        time_based = TimeBasedTiming(config)

        # Should have a timing state
        timing_state = time_based.get_timing_state()
        assert timing_state is not None

        # Test that it's compatible with the interface
        assert isinstance(timing_state, ITimingState)

    def test_end_to_end_timing_workflow(self):
        """Test end-to-end workflow with dependency injection."""
        config = {"rebalance_frequency": "M", "rebalance_offset": 0}
        time_based_timing = create_time_based_timing(config)

        # Use in adaptive controller
        from portfolio_backtester.timing.custom_timing_registry import (
            AdaptiveTimingController,
        )

        adaptive = AdaptiveTimingController(config, time_based_timing=time_based_timing)

        # Test date generation
        start_date = pd.Timestamp("2023-01-01")
        end_date = pd.Timestamp("2023-12-31")
        available_dates = pd.date_range(start_date, end_date, freq="B")  # Business days

        dates = adaptive.get_rebalance_dates(start_date, end_date, available_dates, None)

        # Should have generated some rebalance dates
        assert len(dates) > 0
        assert isinstance(dates, pd.DatetimeIndex)
        assert all(isinstance(date, pd.Timestamp) for date in dates)
