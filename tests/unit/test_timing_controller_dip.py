"""
Tests for TimingController Dependency Inversion Principle implementation.

This test suite verifies that the TimingController class properly implements
dependency inversion by accepting ITimingState abstractions instead of
concrete TimingState implementations.
"""

import pytest
import pandas as pd
from portfolio_backtester.timing.timing_controller import TimingController
from portfolio_backtester.interfaces.timing_state_interface import (
    ITimingState,
    create_timing_state,
    create_timing_state_factory,
)


class ConcreteTimingController(TimingController):
    """Concrete implementation for testing DIP."""

    def get_rebalance_dates(self, start_date, end_date, available_dates, strategy_context):
        return available_dates[(available_dates >= start_date) & (available_dates <= end_date)]

    def should_generate_signal(self, current_date, strategy_context):
        return True


class MockTimingState(ITimingState):
    """Mock implementation of ITimingState for testing."""

    def __init__(self):
        self._last_signal_date = None
        self._last_weights = None
        self._scheduled_dates = set()
        self.reset_called = False
        self.update_signal_called = False
        self.update_positions_called = False

    def reset(self) -> None:
        self.reset_called = True
        self._last_signal_date = None
        self._last_weights = None
        self._scheduled_dates.clear()

    def update_signal(self, date: pd.Timestamp, weights: pd.Series) -> None:
        self.update_signal_called = True
        self._last_signal_date = date
        self._last_weights = weights

    def update_positions(
        self, date: pd.Timestamp, new_weights: pd.Series, prices: pd.Series
    ) -> None:
        self.update_positions_called = True

    def get_position_holding_days(self, asset: str, current_date: pd.Timestamp):
        return 10

    def is_position_held(self, asset: str) -> bool:
        return True

    def get_held_assets(self):
        return {"AAPL", "GOOGL"}

    def get_consecutive_periods(self, asset: str) -> int:
        return 5

    def get_position_return(self, asset: str, current_price=None):
        return 0.05

    def get_portfolio_summary(self):
        return {"total_positions": 2}

    def get_position_statistics(self):
        return {"avg_holding_period": 15}

    def to_dict(self):
        return {"mock": True}

    def save_to_file(self, filepath: str) -> None:
        pass

    @property
    def last_signal_date(self):
        return self._last_signal_date

    @last_signal_date.setter
    def last_signal_date(self, value):
        self._last_signal_date = value

    @property
    def last_weights(self):
        return self._last_weights

    @last_weights.setter
    def last_weights(self, value):
        self._last_weights = value

    @property
    def scheduled_dates(self):
        return self._scheduled_dates

    @scheduled_dates.setter
    def scheduled_dates(self, value):
        self._scheduled_dates = value


class TestTimingControllerDIP:
    """Test cases for TimingController DIP implementation."""

    def test_timing_controller_accepts_timing_state_injection(self):
        """Test that TimingController accepts ITimingState via dependency injection."""
        # Create mock timing state
        mock_state = MockTimingState()

        # Create controller with injected state
        config = {"test_param": "test_value"}
        controller = ConcreteTimingController(config, timing_state=mock_state)

        # Verify the injected state is used
        assert controller.timing_state is mock_state
        assert isinstance(controller.timing_state, ITimingState)

    def test_timing_controller_creates_default_state_when_none_provided(self):
        """Test backward compatibility - creates default state when none provided."""
        config = {"test_param": "test_value"}
        controller = ConcreteTimingController(config)

        # Verify default state is created
        assert controller.timing_state is not None
        assert isinstance(controller.timing_state, ITimingState)

    def test_timing_controller_delegates_to_injected_state(self):
        """Test that TimingController methods delegate to the injected ITimingState."""
        mock_state = MockTimingState()
        config = {}
        controller = ConcreteTimingController(config, timing_state=mock_state)

        # Test reset_state delegation
        controller.reset_state()
        assert mock_state.reset_called

        # Test update_signal_state delegation
        test_date = pd.Timestamp("2023-01-01")
        test_weights = pd.Series([0.5, 0.5], index=["AAPL", "GOOGL"])
        controller.update_signal_state(test_date, test_weights)
        assert mock_state.update_signal_called

        # Test update_position_state delegation
        test_prices = pd.Series([150.0, 120.0], index=["AAPL", "GOOGL"])
        controller.update_position_state(test_date, test_weights, test_prices)
        assert mock_state.update_positions_called

    def test_get_timing_state_returns_interface_type(self):
        """Test that get_timing_state returns ITimingState interface type."""
        mock_state = MockTimingState()
        config = {}
        controller = ConcreteTimingController(config, timing_state=mock_state)

        returned_state = controller.get_timing_state()
        assert returned_state is mock_state
        assert isinstance(returned_state, ITimingState)

    def test_timing_state_factory_creates_compatible_instance(self):
        """Test that the timing state factory creates interface-compatible instances."""
        # Test factory function
        state = create_timing_state()
        assert isinstance(state, ITimingState)

        # Test factory class
        factory = create_timing_state_factory()
        state2 = factory.create_timing_state()
        assert isinstance(state2, ITimingState)

    def test_interface_methods_work_with_real_implementation(self):
        """Test that all interface methods work with the default implementation."""
        config = {}
        controller = ConcreteTimingController(config)
        state = controller.get_timing_state()

        # Test basic interface methods
        test_date = pd.Timestamp("2023-01-01")
        test_weights = pd.Series([0.6, 0.4], index=["AAPL", "MSFT"])
        test_prices = pd.Series([150.0, 250.0], index=["AAPL", "MSFT"])

        # These should not raise exceptions
        state.reset()
        state.update_signal(test_date, test_weights)
        state.update_positions(test_date, test_weights, test_prices)

        # Test properties
        state.last_signal_date = test_date
        assert state.last_signal_date == test_date

        state.last_weights = test_weights
        pd.testing.assert_series_equal(state.last_weights, test_weights)

        # Test query methods
        portfolio_summary = state.get_portfolio_summary()
        assert isinstance(portfolio_summary, dict)

        position_stats = state.get_position_statistics()
        assert isinstance(position_stats, dict)

        state_dict = state.to_dict()
        assert isinstance(state_dict, dict)

    def test_multiple_controllers_can_share_state(self):
        """Test that multiple controllers can share the same timing state instance."""
        shared_state = create_timing_state()
        config1 = {"controller": "first"}
        config2 = {"controller": "second"}

        controller1 = ConcreteTimingController(config1, timing_state=shared_state)
        controller2 = ConcreteTimingController(config2, timing_state=shared_state)

        # Both controllers should reference the same state
        assert controller1.get_timing_state() is shared_state
        assert controller2.get_timing_state() is shared_state
        assert controller1.get_timing_state() is controller2.get_timing_state()

    def test_backward_compatibility_with_existing_code(self):
        """Test that existing code continues to work without modifications."""
        # This simulates existing code that doesn't use dependency injection
        config = {"rebalance_frequency": "M"}
        controller = ConcreteTimingController(config)  # No timing_state parameter

        # All existing functionality should work
        assert controller.config == config
        assert controller.timing_state is not None

        # Test that all existing methods work
        test_date = pd.Timestamp("2023-01-01")
        test_weights = pd.Series([1.0], index=["SPY"])
        test_prices = pd.Series([400.0], index=["SPY"])

        controller.reset_state()
        controller.update_signal_state(test_date, test_weights)
        controller.update_position_state(test_date, test_weights, test_prices)

        state = controller.get_timing_state()
        assert isinstance(state, ITimingState)


if __name__ == "__main__":
    pytest.main([__file__])
