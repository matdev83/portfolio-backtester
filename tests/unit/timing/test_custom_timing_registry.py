"""
Tests for custom timing controller registry and factory.
Split from test_configuration_extensibility.py for better organization.
"""

import pytest
from portfolio_backtester.timing.custom_timing_registry import (
    CustomTimingRegistry,
    TimingControllerFactory,
    register_timing_controller,
)
from portfolio_backtester.timing.timing_controller import TimingController


class TestCustomTimingRegistry:
    """Test custom timing controller registry and factory."""

    def setup_method(self):
        """Set up test environment."""
        # Clear registry before each test
        CustomTimingRegistry.clear()

    def teardown_method(self):
        """Clean up after each test."""
        # Clear registry after each test
        CustomTimingRegistry.clear()

    def test_register_timing_controller(self):
        """Test registering a custom timing controller."""

        class TestController(TimingController):
            def should_generate_signal(self, current_date, strategy):
                return True

            def get_rebalance_dates(self, start_date, end_date, available_dates, strategy):
                return []

        CustomTimingRegistry.register("test_controller", TestController)

        retrieved = CustomTimingRegistry.get("test_controller")
        assert retrieved == TestController

        registered = CustomTimingRegistry.list_registered()
        assert "test_controller" in registered

    def test_register_with_aliases(self):
        """Test registering controller with aliases."""

        class TestController(TimingController):
            def should_generate_signal(self, current_date, strategy):
                return True

            def get_rebalance_dates(self, start_date, end_date, available_dates, strategy):
                return []

        CustomTimingRegistry.register("test_controller", TestController, aliases=["tc", "test"])

        # Test direct access
        assert CustomTimingRegistry.get("test_controller") == TestController

        # Test alias access
        assert CustomTimingRegistry.get("tc") == TestController
        assert CustomTimingRegistry.get("test") == TestController

    def test_register_invalid_controller(self):
        """Test registering invalid controller class."""

        class InvalidController:
            pass

        with pytest.raises(
            ValueError, match="Component must be a class that inherits from TimingController"
        ):
            CustomTimingRegistry.register("invalid", InvalidController)

    def test_unregister_controller(self):
        """Test unregistering a controller."""

        class TestController(TimingController):
            def should_generate_signal(self, current_date, strategy):
                return True

            def get_rebalance_dates(self, start_date, end_date, available_dates, strategy):
                return []

        CustomTimingRegistry.register("test_controller", TestController, aliases=["tc"])

        # Verify registration
        assert CustomTimingRegistry.get("test_controller") == TestController
        assert CustomTimingRegistry.get("tc") == TestController

        # Unregister
        result = CustomTimingRegistry.unregister("test_controller")
        assert result

        # Verify removal
        assert CustomTimingRegistry.get("test_controller") is None
        assert CustomTimingRegistry.get("tc") is None

    def test_decorator_registration(self):
        """Test decorator-based registration."""

        @register_timing_controller("decorated_controller", aliases=["dc"])
        class DecoratedController(TimingController):
            def should_generate_signal(self, current_date, strategy):
                return True

            def get_rebalance_dates(self, start_date, end_date, available_dates, strategy):
                return []

        # Test registration worked
        assert CustomTimingRegistry.get("decorated_controller") == DecoratedController
        assert CustomTimingRegistry.get("dc") == DecoratedController

    def test_timing_controller_factory(self):
        """Test timing controller factory."""
        # Test time-based creation
        time_config = {"mode": "time_based", "rebalance_frequency": "M"}
        controller = TimingControllerFactory.create_controller(time_config)
        assert controller.__class__.__name__ == "TimeBasedTiming"

        # Test signal-based creation
        signal_config = {"mode": "signal_based", "scan_frequency": "D"}
        controller = TimingControllerFactory.create_controller(signal_config)
        assert controller.__class__.__name__ == "SignalBasedTiming"

    def test_custom_controller_creation(self):
        """Test custom controller creation through factory."""

        class CustomController(TimingController):
            def __init__(self, config, custom_param=None):
                super().__init__(config)
                self.custom_param = custom_param

            def should_generate_signal(self, current_date, strategy):
                return True

            def get_rebalance_dates(self, start_date, end_date, available_dates, strategy):
                return []

        CustomTimingRegistry.register("custom_test", CustomController)

        config = {
            "mode": "custom",
            "custom_controller_class": "custom_test",
            "custom_controller_params": {"custom_param": "test_value"},
        }

        controller = TimingControllerFactory.create_controller(config)
        assert isinstance(controller, CustomController)
        assert controller.custom_param == "test_value"

    def test_factory_invalid_mode(self):
        """Test factory with invalid mode."""
        config = {"mode": "invalid_mode"}

        with pytest.raises(ValueError, match="Unknown timing mode"):
            TimingControllerFactory.create_controller(config)

    def test_factory_missing_custom_class(self):
        """Test factory with missing custom controller class."""
        config = {"mode": "custom"}

        with pytest.raises(ValueError, match="custom_controller_class is required"):
            TimingControllerFactory.create_controller(config)

    def test_factory_nonexistent_custom_class(self):
        """Test factory with non-existent custom controller class."""
        config = {"mode": "custom", "custom_controller_class": "nonexistent.Controller"}

        with pytest.raises(ValueError, match="Cannot find timing controller class"):
            TimingControllerFactory.create_controller(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
