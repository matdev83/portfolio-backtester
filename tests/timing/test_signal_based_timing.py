"""
Unit tests for SignalBasedTiming controller.
"""

import pytest
import pandas as pd
from unittest.mock import Mock

from src.portfolio_backtester.timing.signal_based_timing import SignalBasedTiming


class TestSignalBasedTiming:
    """Test cases for SignalBasedTiming controller."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample date range for testing
        self.start_date = pd.Timestamp('2023-01-01')
        self.end_date = pd.Timestamp('2023-01-31')
        
        # Create available trading dates (weekdays only)
        self.available_dates = pd.bdate_range(
            start='2023-01-01', 
            end='2023-01-31', 
            freq='B'  # Business days
        )
        
        # Mock strategy context
        self.mock_strategy = Mock()
    
    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        config = {}
        timing = SignalBasedTiming(config)
        
        assert timing.scan_frequency == 'D'
        assert timing.max_holding_period is None
        assert timing.min_holding_period == 1
        assert timing.timing_state is not None
    
    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = {
            'scan_frequency': 'W',
            'max_holding_period': 30,
            'min_holding_period': 5
        }
        timing = SignalBasedTiming(config)
        
        assert timing.scan_frequency == 'W'
        assert timing.max_holding_period == 30
        assert timing.min_holding_period == 5
    
    def test_config_validation_invalid_scan_frequency(self):
        """Test configuration validation with invalid scan frequency."""
        config = {'scan_frequency': 'X'}
        
        with pytest.raises(ValueError, match="Invalid scan_frequency 'X'"):
            SignalBasedTiming(config)
    
    def test_config_validation_invalid_min_holding_period(self):
        """Test configuration validation with invalid minimum holding period."""
        config = {'min_holding_period': 0}
        
        with pytest.raises(ValueError, match="min_holding_period must be a positive integer"):
            SignalBasedTiming(config)
    
    def test_config_validation_invalid_max_holding_period(self):
        """Test configuration validation with invalid maximum holding period."""
        config = {'max_holding_period': -5}
        
        with pytest.raises(ValueError, match="max_holding_period must be a positive integer"):
            SignalBasedTiming(config)
    
    def test_config_validation_min_greater_than_max(self):
        """Test configuration validation when min > max holding period."""
        config = {
            'min_holding_period': 10,
            'max_holding_period': 5
        }
        
        with pytest.raises(ValueError, match="min_holding_period .* cannot exceed max_holding_period"):
            SignalBasedTiming(config)
    
    def test_get_rebalance_dates_daily_scan(self):
        """Test getting rebalance dates with daily scanning."""
        config = {'scan_frequency': 'D'}
        timing = SignalBasedTiming(config)
        
        rebalance_dates = timing.get_rebalance_dates(
            self.start_date, self.end_date, self.available_dates, self.mock_strategy
        )
        
        # Should return all available trading dates
        expected_dates = self.available_dates
        pd.testing.assert_index_equal(rebalance_dates, expected_dates)
        
        # Check that scheduled dates are stored in timing state
        assert timing.timing_state.scheduled_dates == set(expected_dates)
    
    def test_get_rebalance_dates_weekly_scan(self):
        """Test getting rebalance dates with weekly scanning."""
        config = {'scan_frequency': 'W'}
        timing = SignalBasedTiming(config)
        
        rebalance_dates = timing.get_rebalance_dates(
            self.start_date, self.end_date, self.available_dates, self.mock_strategy
        )
        
        # Should return weekly dates aligned to trading days
        assert len(rebalance_dates) > 0
        assert len(rebalance_dates) < len(self.available_dates)  # Fewer than daily
        
        # All returned dates should be in available dates
        for date in rebalance_dates:
            assert date in self.available_dates
    
    def test_get_rebalance_dates_monthly_scan(self):
        """Test getting rebalance dates with monthly scanning."""
        config = {'scan_frequency': 'M'}
        timing = SignalBasedTiming(config)
        
        rebalance_dates = timing.get_rebalance_dates(
            self.start_date, self.end_date, self.available_dates, self.mock_strategy
        )
        
        # Should return monthly dates aligned to trading days
        assert len(rebalance_dates) > 0
        assert len(rebalance_dates) <= 2  # At most 2 months in January range
        
        # All returned dates should be in available dates
        for date in rebalance_dates:
            assert date in self.available_dates
    
    def test_should_generate_signal_not_scheduled_date(self):
        """Test should_generate_signal returns False for non-scheduled dates."""
        config = {'scan_frequency': 'W'}
        timing = SignalBasedTiming(config)
        
        # Get rebalance dates first to populate scheduled dates
        timing.get_rebalance_dates(
            self.start_date, self.end_date, self.available_dates, self.mock_strategy
        )
        
        # Test with a date not in scheduled dates
        non_scheduled_date = pd.Timestamp('2023-01-03')  # Tuesday, likely not weekly
        if non_scheduled_date not in timing.timing_state.scheduled_dates:
            result = timing.should_generate_signal(non_scheduled_date, self.mock_strategy)
            assert result is False
    
    def test_should_generate_signal_within_min_holding_period(self):
        """Test should_generate_signal respects minimum holding period."""
        config = {
            'scan_frequency': 'D',
            'min_holding_period': 5
        }
        timing = SignalBasedTiming(config)
        
        # Get rebalance dates
        timing.get_rebalance_dates(
            self.start_date, self.end_date, self.available_dates, self.mock_strategy
        )
        
        # Set last signal date
        last_signal_date = pd.Timestamp('2023-01-02')
        timing.timing_state.last_signal_date = last_signal_date
        
        # Test date within minimum holding period
        test_date = pd.Timestamp('2023-01-05')  # 3 days later, less than min 5
        result = timing.should_generate_signal(test_date, self.mock_strategy)
        assert result is False
        
        # Test date after minimum holding period
        test_date = pd.Timestamp('2023-01-09')  # 7 days later, more than min 5
        result = timing.should_generate_signal(test_date, self.mock_strategy)
        assert result is True
    
    def test_should_generate_signal_force_rebalance_max_holding(self):
        """Test should_generate_signal forces rebalance at max holding period."""
        config = {
            'scan_frequency': 'D',
            'min_holding_period': 1,
            'max_holding_period': 5
        }
        timing = SignalBasedTiming(config)
        
        # Get rebalance dates
        timing.get_rebalance_dates(
            self.start_date, self.end_date, self.available_dates, self.mock_strategy
        )
        
        # Set last signal date
        last_signal_date = pd.Timestamp('2023-01-02')
        timing.timing_state.last_signal_date = last_signal_date
        
        # Test date at maximum holding period
        test_date = pd.Timestamp('2023-01-09')  # 7 days later, exceeds max 5
        result = timing.should_generate_signal(test_date, self.mock_strategy)
        assert result is True
    
    def test_get_days_since_last_signal(self):
        """Test getting days since last signal."""
        config = {}
        timing = SignalBasedTiming(config)
        
        # No last signal date
        current_date = pd.Timestamp('2023-01-05')
        days = timing.get_days_since_last_signal(current_date)
        assert days == 0
        
        # With last signal date
        timing.timing_state.last_signal_date = pd.Timestamp('2023-01-02')
        days = timing.get_days_since_last_signal(current_date)
        assert days == 3
    
    def test_position_tracking_methods(self):
        """Test position tracking helper methods."""
        config = {}
        timing = SignalBasedTiming(config)
        
        # Initially no positions
        assert not timing.is_position_held('AAPL')
        assert timing.get_held_assets() == set()
        assert timing.get_position_holding_days('AAPL', pd.Timestamp('2023-01-05')) == 0
        
        # Add position entry
        timing.timing_state.position_entry_dates['AAPL'] = pd.Timestamp('2023-01-02')
        
        # Check position tracking
        assert timing.is_position_held('AAPL')
        assert timing.get_held_assets() == {'AAPL'}
        assert timing.get_position_holding_days('AAPL', pd.Timestamp('2023-01-05')) == 3
    
    def test_can_enter_position(self):
        """Test can_enter_position method."""
        config = {'min_holding_period': 3}
        timing = SignalBasedTiming(config)
        
        current_date = pd.Timestamp('2023-01-05')
        
        # No last signal - can enter
        assert timing.can_enter_position(current_date) is True
        
        # Within min holding period - cannot enter
        timing.timing_state.last_signal_date = pd.Timestamp('2023-01-04')  # 1 day ago
        assert timing.can_enter_position(current_date) is False
        
        # After min holding period - can enter
        timing.timing_state.last_signal_date = pd.Timestamp('2023-01-01')  # 4 days ago
        assert timing.can_enter_position(current_date) is True
    
    def test_can_exit_position(self):
        """Test can_exit_position method."""
        config = {'min_holding_period': 3}
        timing = SignalBasedTiming(config)
        
        current_date = pd.Timestamp('2023-01-05')
        
        # No position held - cannot exit
        assert timing.can_exit_position('AAPL', current_date) is False
        
        # Position held but within min holding period - cannot exit
        timing.timing_state.position_entry_dates['AAPL'] = pd.Timestamp('2023-01-04')
        assert timing.can_exit_position('AAPL', current_date) is False
        
        # Position held and after min holding period - can exit
        timing.timing_state.position_entry_dates['AAPL'] = pd.Timestamp('2023-01-01')
        assert timing.can_exit_position('AAPL', current_date) is True
    
    def test_must_exit_position(self):
        """Test must_exit_position method."""
        config = {'max_holding_period': 5}
        timing = SignalBasedTiming(config)
        
        current_date = pd.Timestamp('2023-01-08')
        
        # No position held - no forced exit
        assert timing.must_exit_position('AAPL', current_date) is False
        
        # Position held but within max holding period - no forced exit
        timing.timing_state.position_entry_dates['AAPL'] = pd.Timestamp('2023-01-05')
        assert timing.must_exit_position('AAPL', current_date) is False
        
        # Position held and at max holding period - must exit
        timing.timing_state.position_entry_dates['AAPL'] = pd.Timestamp('2023-01-02')  # 6 days ago
        assert timing.must_exit_position('AAPL', current_date) is True
        
        # No max holding period configured - no forced exit
        config_no_max = {}
        timing_no_max = SignalBasedTiming(config_no_max)
        timing_no_max.timing_state.position_entry_dates['AAPL'] = pd.Timestamp('2023-01-01')
        assert timing_no_max.must_exit_position('AAPL', current_date) is False
    
    def test_reset_state(self):
        """Test state reset functionality."""
        config = {}
        timing = SignalBasedTiming(config)
        
        # Set some state
        timing.timing_state.last_signal_date = pd.Timestamp('2023-01-02')
        timing.timing_state.position_entry_dates['AAPL'] = pd.Timestamp('2023-01-01')
        timing.timing_state.scheduled_dates.add(pd.Timestamp('2023-01-03'))
        
        # Reset state
        timing.reset_state()
        
        # Verify state is cleared
        assert timing.timing_state.last_signal_date is None
        assert len(timing.timing_state.position_entry_dates) == 0
        assert len(timing.timing_state.scheduled_dates) == 0
    
    def test_integration_with_timing_state_updates(self):
        """Test integration with timing state update methods."""
        config = {}
        timing = SignalBasedTiming(config)
        
        # Test signal update
        date = pd.Timestamp('2023-01-02')
        weights = pd.Series([0.5, 0.5], index=['AAPL', 'GOOGL'])
        timing.update_signal_state(date, weights)
        
        assert timing.timing_state.last_signal_date == date
        pd.testing.assert_series_equal(timing.timing_state.last_weights, weights)
        
        # Test position update - first call with new positions (from zero weights)
        new_weights = pd.Series([0.7, 0.3], index=['AAPL', 'GOOGL'])
        prices = pd.Series([150.0, 2500.0], index=['AAPL', 'GOOGL'])
        
        # Set last_weights to None to simulate initial state (no previous positions)
        timing.timing_state.last_weights = None
        timing.update_position_state(date, new_weights, prices)
        
        # Both positions should be tracked as new entries (from 0.0 to non-zero)
        assert 'AAPL' in timing.timing_state.position_entry_dates
        assert 'GOOGL' in timing.timing_state.position_entry_dates
        assert timing.timing_state.position_entry_prices['AAPL'] == 150.0
        assert timing.timing_state.position_entry_prices['GOOGL'] == 2500.0


class TestSignalBasedTimingEdgeCases:
    """Test edge cases and error conditions for SignalBasedTiming."""
    
    def test_empty_date_range(self):
        """Test behavior with empty date range."""
        config = {}
        timing = SignalBasedTiming(config)
        
        start_date = pd.Timestamp('2023-01-01')
        end_date = pd.Timestamp('2022-12-31')  # End before start
        available_dates = pd.DatetimeIndex([])
        
        rebalance_dates = timing.get_rebalance_dates(
            start_date, end_date, available_dates, Mock()
        )
        
        assert len(rebalance_dates) == 0
    
    def test_single_date_range(self):
        """Test behavior with single date in range."""
        config = {}
        timing = SignalBasedTiming(config)
        
        single_date = pd.Timestamp('2023-01-02')
        available_dates = pd.DatetimeIndex([single_date])
        
        rebalance_dates = timing.get_rebalance_dates(
            single_date, single_date, available_dates, Mock()
        )
        
        assert len(rebalance_dates) == 1
        assert rebalance_dates[0] == single_date
    
    def test_weekend_handling(self):
        """Test handling of weekend dates in configuration."""
        config = {'scan_frequency': 'W'}
        timing = SignalBasedTiming(config)
        
        # Create date range including weekends
        start_date = pd.Timestamp('2023-01-01')  # Sunday
        end_date = pd.Timestamp('2023-01-31')
        available_dates = pd.bdate_range(start_date, end_date)  # Business days only
        
        rebalance_dates = timing.get_rebalance_dates(
            start_date, end_date, available_dates, Mock()
        )
        
        # All returned dates should be business days
        for date in rebalance_dates:
            assert date.weekday() < 5  # Monday=0, Friday=4
            assert date in available_dates
    
    def test_large_date_range_performance(self):
        """Test performance with large date range."""
        config = {'scan_frequency': 'D'}
        timing = SignalBasedTiming(config)
        
        # Create 1 year of business days
        start_date = pd.Timestamp('2023-01-01')
        end_date = pd.Timestamp('2023-12-31')
        available_dates = pd.bdate_range(start_date, end_date)
        
        # This should complete quickly
        rebalance_dates = timing.get_rebalance_dates(
            start_date, end_date, available_dates, Mock()
        )
        
        assert len(rebalance_dates) == len(available_dates)
        pd.testing.assert_index_equal(rebalance_dates, available_dates)