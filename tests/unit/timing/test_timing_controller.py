"""
Tests for TimingController abstract base class.
"""

import pytest
import pandas as pd
from unittest.mock import Mock
from portfolio_backtester.timing.timing_controller import TimingController
from portfolio_backtester.interfaces.timing_state_interface import ITimingState


class ConcreteTimingController(TimingController):
    """Concrete implementation for testing."""

    def get_rebalance_dates(self, start_date, end_date, available_dates, strategy_context):
        # Simple implementation for testing
        return available_dates[(available_dates >= start_date) & (available_dates <= end_date)]

    def should_generate_signal(self, current_date, strategy_context):
        # Simple implementation for testing
        return True


class TestTimingController:
    """Test cases for TimingController."""

    def test_initialization(self):
        """Test TimingController initialization."""
        config = {"test_param": "test_value"}
        controller = ConcreteTimingController(config)

        assert controller.config == config
        assert isinstance(controller.timing_state, ITimingState)

    def test_reset_state(self):
        """Test state reset functionality."""
        controller = ConcreteTimingController({})

        # Set some state
        controller.timing_state.last_signal_date = pd.Timestamp("2023-01-01")
        controller.timing_state.position_entry_dates["A"] = pd.Timestamp("2023-01-01")

        # Reset
        controller.reset_state()

        # Verify reset
        assert controller.timing_state.last_signal_date is None
        assert len(controller.timing_state.position_entry_dates) == 0

    def test_get_timing_state(self):
        """Test getting timing state."""
        controller = ConcreteTimingController({})
        state = controller.get_timing_state()

        assert isinstance(state, ITimingState)
        assert state is controller.timing_state

    def test_update_signal_state(self):
        """Test updating signal state."""
        controller = ConcreteTimingController({})
        date = pd.Timestamp("2023-01-01")
        weights = pd.Series([0.5, 0.5], index=["A", "B"])

        controller.update_signal_state(date, weights)

        assert controller.timing_state.last_signal_date == date
        assert controller.timing_state.last_weights.equals(weights)

    def test_update_position_state(self):
        """Test updating position state."""
        controller = ConcreteTimingController({})
        date = pd.Timestamp("2023-01-01")
        weights = pd.Series([0.5, 0.5], index=["A", "B"])
        prices = pd.Series([100.0, 200.0], index=["A", "B"])

        controller.update_position_state(date, weights, prices)

        assert controller.timing_state.position_entry_dates["A"] == date
        assert controller.timing_state.position_entry_dates["B"] == date
        assert controller.timing_state.position_entry_prices["A"] == 100.0
        assert controller.timing_state.position_entry_prices["B"] == 200.0

    def test_abstract_methods_must_be_implemented(self):
        """Test that abstract methods must be implemented."""
        with pytest.raises(TypeError):
            # This should fail because abstract methods are not implemented
            TimingController({})

    def test_concrete_implementation_works(self):
        """Test that concrete implementation works."""
        controller = ConcreteTimingController({})

        # Test get_rebalance_dates
        start_date = pd.Timestamp("2023-01-01")
        end_date = pd.Timestamp("2023-01-31")
        available_dates = pd.date_range("2023-01-01", "2023-01-31", freq="D")
        strategy_context = Mock()

        result = controller.get_rebalance_dates(
            start_date, end_date, available_dates, strategy_context
        )
        assert isinstance(result, pd.DatetimeIndex)
        assert len(result) == 31  # All days in January

        # Test should_generate_signal
        current_date = pd.Timestamp("2023-01-15")
        result = controller.should_generate_signal(current_date, strategy_context)
        assert result is True
