"""
Tests for TimeBasedTiming controller.
"""

import pytest
import pandas as pd
from unittest.mock import Mock
from portfolio_backtester.timing.custom_timing_registry import TimingControllerFactory
from portfolio_backtester.timing.time_based_timing import TimeBasedTiming


class TestTimeBasedTiming:
    """Test cases for TimeBasedTiming."""

    def create_time_based_timing(self, config=None):
        """Create TimeBasedTiming using interface-compliant factory pattern."""
        if config is None:
            config = {}
        # Ensure mode is set for factory
        factory_config = config.copy()
        factory_config["mode"] = "time_based"
        controller = TimingControllerFactory.create_controller(factory_config)
        # Verify it's the correct type
        assert isinstance(
            controller, TimeBasedTiming
        ), f"Expected TimeBasedTiming, got {type(controller)}"
        return controller

    def test_initialization_defaults(self):
        """Test TimeBasedTiming initialization with defaults."""
        controller = self.create_time_based_timing({})

        assert controller.frequency == "M"
        assert controller.offset == 0

    def test_initialization_custom_config(self):
        """Test TimeBasedTiming initialization with custom config."""
        config = {"rebalance_frequency": "Q", "rebalance_offset": 5}
        controller = self.create_time_based_timing(config)

        assert controller.frequency == "Q"
        assert controller.offset == 5

    def test_get_rebalance_dates_monthly(self):
        """Test monthly rebalance date generation."""
        controller = self.create_time_based_timing({"rebalance_frequency": "M"})

        start_date = pd.Timestamp("2023-01-01")
        end_date = pd.Timestamp("2023-03-31")
        available_dates = pd.bdate_range("2023-01-01", "2023-03-31")  # Business days
        strategy_context = Mock()

        result = controller.get_rebalance_dates(
            start_date, end_date, available_dates, strategy_context
        )

        # Should get month-end dates (or nearest business day)
        assert len(result) == 3  # Jan, Feb, Mar
        assert result[0] == pd.Timestamp("2023-01-31")  # Jan 31 is a Tuesday
        assert result[1] == pd.Timestamp("2023-02-28")  # Feb 28 is a Tuesday
        assert result[2] == pd.Timestamp("2023-03-31")  # Mar 31 is a Friday

        # Check that scheduled dates are stored in timing state
        assert len(controller.timing_state.scheduled_dates) == 3
        assert pd.Timestamp("2023-01-31") in controller.timing_state.scheduled_dates

    def test_get_rebalance_dates_quarterly(self):
        """Test quarterly rebalance date generation."""
        controller = self.create_time_based_timing({"rebalance_frequency": "Q"})

        start_date = pd.Timestamp("2023-01-01")
        end_date = pd.Timestamp("2023-12-31")
        available_dates = pd.bdate_range("2023-01-01", "2023-12-31")
        strategy_context = Mock()

        result = controller.get_rebalance_dates(
            start_date, end_date, available_dates, strategy_context
        )

        # Should get quarter-end dates (some may be adjusted for weekends)
        assert (
            len(result) >= 3
        )  # At least Q1, Q2, Q3 (Q4 might be beyond range due to weekend adjustment)
        assert result[0] == pd.Timestamp("2023-03-31")  # Q1 end
        assert result[1] == pd.Timestamp("2023-06-30")  # Q2 end
        assert result[2] == pd.Timestamp("2023-09-29")  # Q3 end is a Friday

    def test_get_rebalance_dates_daily(self):
        """Test daily rebalance date generation."""
        controller = self.create_time_based_timing({"rebalance_frequency": "D"})

        start_date = pd.Timestamp("2023-01-02")  # Start on Monday
        end_date = pd.Timestamp("2023-01-06")  # End on Friday
        available_dates = pd.bdate_range("2023-01-02", "2023-01-06")  # Business days only
        strategy_context = Mock()

        result = controller.get_rebalance_dates(
            start_date, end_date, available_dates, strategy_context
        )

        # Should get all business days
        expected_days = 4
        assert len(result) == expected_days
        assert result[0] == pd.Timestamp("2023-01-03")

    def test_get_rebalance_dates_with_offset(self):
        """Test rebalance date generation with offset."""
        controller = self.create_time_based_timing(
            {"rebalance_frequency": "M", "rebalance_offset": -5}  # 5 days before month end
        )

        start_date = pd.Timestamp("2023-01-01")
        end_date = pd.Timestamp("2023-02-28")
        available_dates = pd.bdate_range("2023-01-01", "2023-02-28")
        strategy_context = Mock()

        result = controller.get_rebalance_dates(
            start_date, end_date, available_dates, strategy_context
        )

        # Should get dates 5 days before month end
        assert len(result) == 2
        # Jan 31 - 5 days = Jan 26 (Thursday)
        assert result[0] == pd.Timestamp("2023-01-26")
        # Feb 28 - 5 days = Feb 23 (Thursday)
        assert result[1] == pd.Timestamp("2023-02-23")

    def test_get_rebalance_dates_weekend_adjustment(self):
        """Test that rebalance dates are adjusted for weekends."""
        controller = self.create_time_based_timing({"rebalance_frequency": "M"})

        # Use a month where month-end falls on weekend
        start_date = pd.Timestamp("2023-04-01")
        end_date = pd.Timestamp("2023-05-05")
        available_dates = pd.bdate_range("2023-04-01", "2023-05-05")
        strategy_context = Mock()

        result = controller.get_rebalance_dates(
            start_date, end_date, available_dates, strategy_context
        )

        # Apr 30 is a Sunday, should roll back to Friday, Apr 28
        assert len(result) == 1
        assert result[0] == pd.Timestamp("2023-04-28")

    def test_get_rebalance_dates_no_future_dates(self):
        """Test that no dates beyond end_date are included."""
        controller = self.create_time_based_timing({"rebalance_frequency": "M"})

        start_date = pd.Timestamp("2023-01-01")
        end_date = pd.Timestamp("2023-01-15")  # Mid-month
        available_dates = pd.bdate_range("2023-01-01", "2023-02-28")
        strategy_context = Mock()

        result = controller.get_rebalance_dates(
            start_date, end_date, available_dates, strategy_context
        )

        # Should get no dates since Jan 31 is beyond end_date
        assert len(result) == 0

    def test_should_generate_signal_on_scheduled_date(self):
        """Test signal generation on scheduled dates."""
        controller = self.create_time_based_timing({"rebalance_frequency": "M"})

        # Set up scheduled dates
        scheduled_date = pd.Timestamp("2023-01-31")
        controller.timing_state.scheduled_dates.add(scheduled_date)

        strategy_context = Mock()

        # Should generate signal on scheduled date
        assert controller.should_generate_signal(scheduled_date, strategy_context) is True

        # Should not generate signal on non-scheduled date
        non_scheduled_date = pd.Timestamp("2023-01-15")
        assert controller.should_generate_signal(non_scheduled_date, strategy_context) is False

    def test_should_generate_signal_empty_schedule(self):
        """Test signal generation with empty schedule."""
        controller = self.create_time_based_timing({"rebalance_frequency": "M"})
        strategy_context = Mock()

        # Should not generate signal when no dates are scheduled
        current_date = pd.Timestamp("2023-01-31")
        assert controller.should_generate_signal(current_date, strategy_context) is False

    def test_frequency_conversion_m_to_me(self):
        """Test that 'M' frequency is converted to 'ME' for month-end compatibility."""
        controller = self.create_time_based_timing({"rebalance_frequency": "M"})

        start_date = pd.Timestamp("2023-01-01")
        end_date = pd.Timestamp("2023-01-31")
        available_dates = pd.date_range("2023-01-01", "2023-01-31")
        strategy_context = Mock()

        result = controller.get_rebalance_dates(
            start_date, end_date, available_dates, strategy_context
        )

        # Should get month-end date
        assert len(result) == 1
        assert result[0] == pd.Timestamp("2023-01-31")

    def test_get_rebalance_dates_weekly(self):
        """Test weekly rebalance date generation."""
        controller = self.create_time_based_timing({"rebalance_frequency": "W"})

        start_date = pd.Timestamp("2023-01-02")  # Monday
        end_date = pd.Timestamp("2023-01-29")
        available_dates = pd.bdate_range("2023-01-02", "2023-01-29")
        strategy_context = Mock()

        result = controller.get_rebalance_dates(
            start_date, end_date, available_dates, strategy_context
        )

        # Should get weekly dates (Mondays)
        assert len(result) == 3
        assert result[0] == pd.Timestamp("2023-01-09")
        assert result[1] == pd.Timestamp("2023-01-16")
        assert result[2] == pd.Timestamp("2023-01-23")

    def test_get_rebalance_dates_annual(self):
        """Test annual rebalance date generation."""
        controller = self.create_time_based_timing({"rebalance_frequency": "A"})

        start_date = pd.Timestamp("2022-01-01")
        end_date = pd.Timestamp("2024-12-31")
        available_dates = pd.bdate_range("2022-01-01", "2024-12-31")
        strategy_context = Mock()

        result = controller.get_rebalance_dates(
            start_date, end_date, available_dates, strategy_context
        )

        # Should get year-end dates rolled back to previous business day
        assert len(result) >= 2
        assert result[0] == pd.Timestamp("2022-12-30")  # Dec 31 2022 is a Saturday
        assert result[1] == pd.Timestamp("2023-12-29")  # Dec 31 2023 is a Sunday

    def test_get_rebalance_dates_invalid_frequency(self):
        """Test that invalid frequency raises appropriate error."""
        controller = self.create_time_based_timing({"rebalance_frequency": "INVALID"})

        start_date = pd.Timestamp("2023-01-01")
        end_date = pd.Timestamp("2023-12-31")
        available_dates = pd.bdate_range("2023-01-01", "2023-12-31")
        strategy_context = Mock()

        with pytest.raises(ValueError, match="Invalid frequency 'INVALID'"):
            controller.get_rebalance_dates(start_date, end_date, available_dates, strategy_context)

    def test_get_rebalance_dates_large_positive_offset(self):
        """Test rebalance dates with large positive offset."""
        controller = self.create_time_based_timing(
            {"rebalance_frequency": "M", "rebalance_offset": 10}  # 10 days after month end
        )

        start_date = pd.Timestamp("2023-01-01")
        end_date = pd.Timestamp("2023-03-31")
        available_dates = pd.bdate_range("2023-01-01", "2023-03-31")
        strategy_context = Mock()

        result = controller.get_rebalance_dates(
            start_date, end_date, available_dates, strategy_context
        )

        # Should get dates 10 days after month end
        assert len(result) >= 2
        # Jan 31 + 10 days = Feb 10 (Friday)
        assert result[0] == pd.Timestamp("2023-02-10")
        # Feb 28 + 10 days = Mar 10 (Friday)
        assert result[1] == pd.Timestamp("2023-03-10")

    def test_get_rebalance_dates_large_negative_offset(self):
        """Test rebalance dates with large negative offset."""
        controller = self.create_time_based_timing(
            {"rebalance_frequency": "M", "rebalance_offset": -15}  # 15 days before month end
        )

        start_date = pd.Timestamp("2023-01-01")
        end_date = pd.Timestamp("2023-02-28")
        available_dates = pd.bdate_range("2023-01-01", "2023-02-28")
        strategy_context = Mock()

        result = controller.get_rebalance_dates(
            start_date, end_date, available_dates, strategy_context
        )

        # Should get dates 15 days before month end
        assert len(result) == 2
        # Jan 31 - 15 days = Jan 16 (Monday)
        assert result[0] == pd.Timestamp("2023-01-16")
        # Feb 28 - 15 days = Feb 13 (Monday)
        assert result[1] == pd.Timestamp("2023-02-13")

    def test_get_rebalance_dates_empty_available_dates(self):
        """Test behavior with empty available dates."""
        controller = self.create_time_based_timing({"rebalance_frequency": "M"})

        start_date = pd.Timestamp("2023-01-01")
        end_date = pd.Timestamp("2023-03-31")
        available_dates = pd.DatetimeIndex([])  # Empty
        strategy_context = Mock()

        result = controller.get_rebalance_dates(
            start_date, end_date, available_dates, strategy_context
        )

        # Should return empty result
        assert len(result) == 0

    def test_get_rebalance_dates_single_available_date(self):
        """Test behavior with single available date."""
        controller = self.create_time_based_timing({"rebalance_frequency": "M"})

        start_date = pd.Timestamp("2023-01-01")
        end_date = pd.Timestamp("2023-03-31")
        available_dates = pd.DatetimeIndex([pd.Timestamp("2023-02-15")])  # Single date
        strategy_context = Mock()

        result = controller.get_rebalance_dates(
            start_date, end_date, available_dates, strategy_context
        )

        # Should return the single date if it's a scheduled date
        assert len(result) == 1
        assert result[0] == pd.Timestamp("2023-02-15")

    def test_reset_state_clears_scheduled_dates(self):
        """Test that reset_state clears scheduled dates."""
        controller = self.create_time_based_timing({"rebalance_frequency": "M"})

        # Add some scheduled dates
        controller.timing_state.scheduled_dates.add(pd.Timestamp("2023-01-31"))
        controller.timing_state.scheduled_dates.add(pd.Timestamp("2023-02-28"))

        # Reset state
        controller.reset_state()

        # Scheduled dates should be cleared
        assert len(controller.timing_state.scheduled_dates) == 0

    def test_case_insensitive_frequency(self):
        """Test that frequency is case insensitive."""
        controller_lower = self.create_time_based_timing({"rebalance_frequency": "m"})
        controller_upper = self.create_time_based_timing({"rebalance_frequency": "M"})

        start_date = pd.Timestamp("2023-01-01")
        end_date = pd.Timestamp("2023-02-28")
        available_dates = pd.bdate_range("2023-01-01", "2023-02-28")
        strategy_context = Mock()

        result_lower = controller_lower.get_rebalance_dates(
            start_date, end_date, available_dates, strategy_context
        )
        result_upper = controller_upper.get_rebalance_dates(
            start_date, end_date, available_dates, strategy_context
        )

        # Should produce identical results
        assert len(result_lower) == len(result_upper)
        for i in range(len(result_lower)):
            assert result_lower[i] == result_upper[i]
