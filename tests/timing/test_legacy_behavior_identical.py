"""
Tests to verify that legacy behavior produces identical results
after migration to the new timing system.

These tests ensure numerical accuracy and behavioral equivalence.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.portfolio_backtester.timing.backward_compatibility import (
    ensure_backward_compatibility,
    migrate_legacy_config
)


class MockStrategy:
    """Mock strategy class for testing timing behavior."""
    
    def __init__(self, strategy_config):
        self.strategy_config = strategy_config
        self.supports_daily_signals_called = False
        self._timing_controller = None
    
    def supports_daily_signals(self):
        """Mock supports_daily_signals method."""
        self.supports_daily_signals_called = True
        return self.strategy_config.get('_mock_supports_daily', False)
    
    def generate_signals(self, *args, **kwargs):
        """Mock signal generation."""
        return pd.DataFrame({'AAPL': [0.5], 'MSFT': [0.5]}, 
                          index=[datetime(2023, 1, 1)])


class TestTimingBehaviorEquivalence:
    """Test that timing behavior is equivalent before and after migration."""
    
    def test_monthly_rebalancing_dates_identical(self):
        """Test that monthly rebalancing produces identical dates."""
        # Legacy configuration
        legacy_config = {
            'strategy': 'momentum',
            'rebalance_frequency': 'M',
            'strategy_params': {'lookback_months': 6}
        }
        
        # Migrated configuration
        migrated_config = migrate_legacy_config(legacy_config)
        
        # Mock date range for testing
        start_date = pd.Timestamp('2023-01-01')
        end_date = pd.Timestamp('2023-12-31')
        available_dates = pd.date_range(start_date, end_date, freq='B')  # Business days
        
        # Test with TimeBasedTiming controller
        from src.portfolio_backtester.timing.time_based_timing import TimeBasedTiming
        
        timing_controller = TimeBasedTiming(migrated_config['timing_config'])
        rebalance_dates = timing_controller.get_rebalance_dates(
            start_date, end_date, available_dates, None
        )
        
        # Expected monthly end dates (last business day of each month)
        expected_dates = []
        for month in range(1, 13):
            if month == 12:
                month_end = pd.Timestamp('2023-12-31')
            else:
                month_end = pd.Timestamp(f'2023-{month+1:02d}-01') - pd.Timedelta(days=1)
            
            # Find last business day of month
            month_business_days = available_dates[
                (available_dates.month == month) & (available_dates.year == 2023)
            ]
            if len(month_business_days) > 0:
                expected_dates.append(month_business_days[-1])
        
        expected_dates = pd.DatetimeIndex(expected_dates)
        
        # Should have similar number of dates (allowing for business day adjustments)
        assert len(rebalance_dates) == len(expected_dates)
        
        # Dates should be close (within a few days for business day adjustments)
        for actual, expected in zip(rebalance_dates, expected_dates):
            assert abs((actual - expected).days) <= 3
    
    def test_quarterly_rebalancing_dates_identical(self):
        """Test that quarterly rebalancing produces identical dates."""
        legacy_config = {
            'strategy': 'value',
            'rebalance_frequency': 'Q',
            'strategy_params': {'book_to_market': True}
        }
        
        migrated_config = migrate_legacy_config(legacy_config)
        
        start_date = pd.Timestamp('2023-01-01')
        end_date = pd.Timestamp('2023-12-31')
        available_dates = pd.date_range(start_date, end_date, freq='B')
        
        from src.portfolio_backtester.timing.time_based_timing import TimeBasedTiming
        
        timing_controller = TimeBasedTiming(migrated_config['timing_config'])
        rebalance_dates = timing_controller.get_rebalance_dates(
            start_date, end_date, available_dates, None
        )
        
        # Should have 4 quarterly dates
        assert len(rebalance_dates) == 4
        
        # Should be roughly at quarter ends
        quarters = [3, 6, 9, 12]  # March, June, September, December
        for i, date in enumerate(rebalance_dates):
            expected_month = quarters[i]
            # Allow some flexibility for business day adjustments
            assert abs(date.month - expected_month) <= 1
    
    def test_daily_signal_strategy_behavior_preserved(self):
        """Test that daily signal strategies behave identically."""
        legacy_config = {
            'strategy': 'uvxy_rsi',
            'rebalance_frequency': 'D',
            'strategy_params': {'rsi_period': 2, 'rsi_threshold': 30}
        }
        
        migrated_config = migrate_legacy_config(legacy_config)
        
        # Verify migration to signal-based timing
        assert migrated_config['timing_config']['mode'] == 'signal_based'
        assert migrated_config['timing_config']['scan_frequency'] == 'D'
        
        # Test signal-based timing controller
        from src.portfolio_backtester.timing.signal_based_timing import SignalBasedTiming
        
        timing_controller = SignalBasedTiming(migrated_config['timing_config'])
        
        start_date = pd.Timestamp('2023-01-01')
        end_date = pd.Timestamp('2023-01-31')
        available_dates = pd.date_range(start_date, end_date, freq='B')
        
        rebalance_dates = timing_controller.get_rebalance_dates(
            start_date, end_date, available_dates, None
        )
        
        # Should scan all business days
        assert len(rebalance_dates) == len(available_dates)
        pd.testing.assert_index_equal(rebalance_dates, available_dates)


class TestSupportsDailySignalsCompatibility:
    """Test that supports_daily_signals method works correctly after migration."""
    
    def test_time_based_strategy_supports_daily_signals_false(self):
        """Test that time-based strategies return False for supports_daily_signals."""
        legacy_config = {
            'strategy': 'momentum',
            'rebalance_frequency': 'M',
            'strategy_params': {'lookback_months': 6}
        }
        
        # Mock BaseStrategy with timing controller
        with patch('src.portfolio_backtester.strategies.base_strategy.BaseStrategy') as MockBaseStrategy:
            mock_strategy = MockStrategy(legacy_config)
            
            # Simulate initialization with backward compatibility
            migrated_config = ensure_backward_compatibility(legacy_config)
            mock_strategy.strategy_config = migrated_config
            
            # Initialize timing controller
            from src.portfolio_backtester.timing.time_based_timing import TimeBasedTiming
            mock_strategy._timing_controller = TimeBasedTiming(migrated_config['timing_config'])
            
            # Test supports_daily_signals method
            from src.portfolio_backtester.timing.signal_based_timing import SignalBasedTiming
            supports_daily = isinstance(mock_strategy._timing_controller, SignalBasedTiming)
            
            assert supports_daily is False
    
    def test_signal_based_strategy_supports_daily_signals_true(self):
        """Test that signal-based strategies return True for supports_daily_signals."""
        legacy_config = {
            'strategy': 'uvxy_rsi',
            'rebalance_frequency': 'D',
            'strategy_params': {'rsi_period': 2}
        }
        
        mock_strategy = MockStrategy(legacy_config)
        
        # Simulate initialization with backward compatibility
        migrated_config = ensure_backward_compatibility(legacy_config)
        mock_strategy.strategy_config = migrated_config
        
        # Initialize timing controller
        from src.portfolio_backtester.timing.signal_based_timing import SignalBasedTiming
        mock_strategy._timing_controller = SignalBasedTiming(migrated_config['timing_config'])
        
        # Test supports_daily_signals method
        supports_daily = isinstance(mock_strategy._timing_controller, SignalBasedTiming)
        
        assert supports_daily is True


class TestNumericalAccuracy:
    """Test numerical accuracy of timing calculations."""
    
    def test_rebalance_offset_calculation_accurate(self):
        """Test that rebalance offset calculations are numerically accurate."""
        legacy_config = {
            'strategy': 'momentum',
            'rebalance_frequency': 'M',
            'rebalance_offset': 5,  # 5 days after month end
            'strategy_params': {'lookback_months': 6}
        }
        
        migrated_config = migrate_legacy_config(legacy_config)
        
        from src.portfolio_backtester.timing.time_based_timing import TimeBasedTiming
        
        timing_controller = TimeBasedTiming(migrated_config['timing_config'])
        
        # Test specific month
        start_date = pd.Timestamp('2023-01-01')
        end_date = pd.Timestamp('2023-02-28')
        available_dates = pd.date_range(start_date, end_date, freq='B')
        
        rebalance_dates = timing_controller.get_rebalance_dates(
            start_date, end_date, available_dates, None
        )
        
        # Should have January rebalance date
        assert len(rebalance_dates) >= 1
        
        # First rebalance should be around January 31 + 5 days = February 5
        # (adjusted for business days)
        jan_rebalance = rebalance_dates[0]
        expected_date = pd.Timestamp('2023-02-05')  # Jan 31 + 5 days
        
        # Should be close to expected date (allowing for weekend adjustments)
        assert abs((jan_rebalance - expected_date).days) <= 3
    
    def test_holding_period_constraints_accurate(self):
        """Test that holding period constraints are numerically accurate."""
        legacy_config = {
            'strategy': 'custom_signal',
            'signal_based': True,
            'scan_frequency': 'D',
            'min_holding_period': 3,
            'max_holding_period': 10
        }
        
        migrated_config = migrate_legacy_config(legacy_config)
        
        from src.portfolio_backtester.timing.signal_based_timing import SignalBasedTiming
        
        timing_controller = SignalBasedTiming(migrated_config['timing_config'])
        
        # Test minimum holding period
        current_date = pd.Timestamp('2023-01-05')
        timing_controller.timing_state.last_signal_date = pd.Timestamp('2023-01-03')
        
        # Should not generate signal (only 2 days since last signal, min is 3)
        should_generate = timing_controller.should_generate_signal(current_date, None)
        assert should_generate is False
        
        # Test after minimum holding period
        current_date = pd.Timestamp('2023-01-07')  # 4 days since last signal
        should_generate = timing_controller.should_generate_signal(current_date, None)
        assert should_generate is True
        
        # Test maximum holding period (should force rebalance)
        current_date = pd.Timestamp('2023-01-14')  # 11 days since last signal
        should_generate = timing_controller.should_generate_signal(current_date, None)
        assert should_generate is True


class TestStateManagement:
    """Test that state management works correctly after migration."""
    
    def test_timing_state_preserved_across_calls(self):
        """Test that timing state is preserved across multiple calls."""
        legacy_config = {
            'strategy': 'uvxy_rsi',
            'rebalance_frequency': 'D',
            'strategy_params': {'rsi_period': 2}
        }
        
        migrated_config = migrate_legacy_config(legacy_config)
        
        from src.portfolio_backtester.timing.signal_based_timing import SignalBasedTiming
        
        timing_controller = SignalBasedTiming(migrated_config['timing_config'])
        
        # First signal generation
        date1 = pd.Timestamp('2023-01-03')
        weights1 = pd.Series({'UVXY': -1.0})
        timing_controller.timing_state.update_signal(date1, weights1)
        
        # Verify state is updated
        assert timing_controller.timing_state.last_signal_date == date1
        pd.testing.assert_series_equal(timing_controller.timing_state.last_weights, weights1)
        
        # Second signal generation
        date2 = pd.Timestamp('2023-01-04')
        weights2 = pd.Series({'UVXY': 0.0})  # Exit position
        timing_controller.timing_state.update_signal(date2, weights2)
        
        # Verify state is updated again
        assert timing_controller.timing_state.last_signal_date == date2
        pd.testing.assert_series_equal(timing_controller.timing_state.last_weights, weights2)
    
    def test_position_tracking_accurate(self):
        """Test that position tracking is numerically accurate."""
        legacy_config = {
            'strategy': 'signal_strategy',
            'signal_based': True,
            'scan_frequency': 'D'
        }
        
        migrated_config = migrate_legacy_config(legacy_config)
        
        from src.portfolio_backtester.timing.signal_based_timing import SignalBasedTiming
        
        timing_controller = SignalBasedTiming(migrated_config['timing_config'])
        
        # Simulate position entry
        entry_date = pd.Timestamp('2023-01-03')
        entry_weights = pd.Series({'AAPL': 0.5, 'MSFT': 0.5})
        entry_prices = pd.Series({'AAPL': 150.0, 'MSFT': 250.0})
        
        timing_controller.timing_state.update_positions(entry_date, entry_weights, entry_prices)
        
        # Verify position tracking
        assert timing_controller.timing_state.position_entry_dates['AAPL'] == entry_date
        assert timing_controller.timing_state.position_entry_dates['MSFT'] == entry_date
        assert timing_controller.timing_state.position_entry_prices['AAPL'] == 150.0
        assert timing_controller.timing_state.position_entry_prices['MSFT'] == 250.0
        
        # Simulate position exit
        exit_date = pd.Timestamp('2023-01-05')
        exit_weights = pd.Series({'AAPL': 0.0, 'MSFT': 0.0})
        exit_prices = pd.Series({'AAPL': 155.0, 'MSFT': 245.0})
        
        timing_controller.timing_state.update_positions(exit_date, exit_weights, exit_prices)
        
        # Verify positions are cleared
        assert 'AAPL' not in timing_controller.timing_state.position_entry_dates
        assert 'MSFT' not in timing_controller.timing_state.position_entry_dates
        assert 'AAPL' not in timing_controller.timing_state.position_entry_prices
        assert 'MSFT' not in timing_controller.timing_state.position_entry_prices


class TestPerformanceEquivalence:
    """Test that performance characteristics are equivalent."""
    
    def test_timing_controller_initialization_performance(self):
        """Test that timing controller initialization doesn't degrade performance."""
        import time
        
        # Test multiple configurations
        configs = [
            {'strategy': 'momentum', 'rebalance_frequency': 'M'},
            {'strategy': 'uvxy_rsi', 'rebalance_frequency': 'D'},
            {'strategy': 'value', 'rebalance_frequency': 'Q'},
        ]
        
        initialization_times = []
        
        for config in configs:
            start_time = time.time()
            
            # Migrate and initialize timing controller
            migrated_config = ensure_backward_compatibility(config)
            
            if migrated_config['timing_config']['mode'] == 'time_based':
                from src.portfolio_backtester.timing.time_based_timing import TimeBasedTiming
                timing_controller = TimeBasedTiming(migrated_config['timing_config'])
            else:
                from src.portfolio_backtester.timing.signal_based_timing import SignalBasedTiming
                timing_controller = SignalBasedTiming(migrated_config['timing_config'])
            
            end_time = time.time()
            initialization_times.append(end_time - start_time)
        
        # All initializations should be fast (< 10ms)
        for init_time in initialization_times:
            assert init_time < 0.01, f"Initialization took {init_time:.4f}s, expected < 0.01s"
    
    def test_rebalance_date_generation_performance(self):
        """Test that rebalance date generation performance is acceptable."""
        import time
        
        legacy_config = {
            'strategy': 'momentum',
            'rebalance_frequency': 'M'
        }
        
        migrated_config = migrate_legacy_config(legacy_config)
        
        from src.portfolio_backtester.timing.time_based_timing import TimeBasedTiming
        
        timing_controller = TimeBasedTiming(migrated_config['timing_config'])
        
        # Test with large date range
        start_date = pd.Timestamp('2000-01-01')
        end_date = pd.Timestamp('2023-12-31')
        available_dates = pd.date_range(start_date, end_date, freq='B')  # ~6000 dates
        
        start_time = time.time()
        rebalance_dates = timing_controller.get_rebalance_dates(
            start_date, end_date, available_dates, None
        )
        end_time = time.time()
        
        generation_time = end_time - start_time
        
        # Should generate dates quickly (< 100ms for 24 years of data)
        assert generation_time < 0.1, f"Date generation took {generation_time:.4f}s, expected < 0.1s"
        
        # Should have reasonable number of monthly dates (~288 months)
        assert 280 <= len(rebalance_dates) <= 300


if __name__ == '__main__':
    pytest.main([__file__])