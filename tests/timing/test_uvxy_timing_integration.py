"""
Test UVXY strategy timing controller integration.
"""

import unittest
import pandas as pd
import numpy as np

from src.portfolio_backtester.strategies.uvxy_rsi_strategy import UvxyRsiStrategy
from src.portfolio_backtester.timing.signal_based_timing import SignalBasedTiming


class TestUvxyTimingIntegration(unittest.TestCase):
    """Test UVXY strategy timing controller integration."""
    
    def setUp(self):
        """Set up test data."""
        self.strategy_config = {
            "strategy_params": {
                "rsi_period": 14,
                "rsi_threshold": 70,
                "lookback_days": 5
            }
        }
    
    def test_uvxy_strategy_uses_signal_based_timing(self):
        """Test that UVXY strategy is automatically configured with signal-based timing."""
        strategy = UvxyRsiStrategy(self.strategy_config)
        
        # Should be detected as signal-based timing due to overridden supports_daily_signals
        timing_controller = strategy.get_timing_controller()
        self.assertIsInstance(timing_controller, SignalBasedTiming)
        
        # Should support daily signals
        self.assertTrue(strategy.supports_daily_signals())
        
        # Check timing controller configuration
        self.assertEqual(timing_controller.scan_frequency, 'D')
        self.assertEqual(timing_controller.min_holding_period, 1)
        self.assertEqual(timing_controller.max_holding_period, 1)  # UVXY has 1-day holding period
    
    def test_uvxy_strategy_with_explicit_timing_config(self):
        """Test UVXY strategy with explicit timing configuration."""
        config_with_timing = {
            "strategy_params": {
                "rsi_period": 14,
                "rsi_threshold": 70,
                "lookback_days": 5
            },
            "timing_config": {
                "mode": "signal_based",
                "scan_frequency": "D",
                "min_holding_period": 3,
                "max_holding_period": 10
            }
        }
        
        strategy = UvxyRsiStrategy(config_with_timing)
        timing_controller = strategy.get_timing_controller()
        
        self.assertIsInstance(timing_controller, SignalBasedTiming)
        self.assertTrue(strategy.supports_daily_signals())
        
        # Should use explicit configuration
        self.assertEqual(timing_controller.scan_frequency, 'D')
        self.assertEqual(timing_controller.min_holding_period, 3)
        self.assertEqual(timing_controller.max_holding_period, 10)
    
    def test_uvxy_strategy_timing_state_management(self):
        """Test that UVXY strategy timing controller manages state correctly."""
        strategy = UvxyRsiStrategy(self.strategy_config)
        timing_controller = strategy.get_timing_controller()
        
        # Test initial state
        self.assertIsNone(timing_controller.timing_state.last_signal_date)
        self.assertEqual(len(timing_controller.timing_state.position_entry_dates), 0)
        
        # Test entering a position from zero
        test_date = pd.Timestamp('2023-01-01')
        test_weights = pd.Series([1.0], index=['UVXY'])
        test_prices = pd.Series([50.0], index=['UVXY'])
        
        # Update position state directly (simulating entering from no position)
        timing_controller.update_position_state(test_date, test_weights, test_prices)
        
        # Update signal state to track the signal
        timing_controller.update_signal_state(test_date, test_weights)
        
        self.assertEqual(timing_controller.timing_state.last_signal_date, test_date)
        pd.testing.assert_series_equal(timing_controller.timing_state.last_weights, test_weights)
        
        # Check position tracking
        self.assertTrue(timing_controller.timing_state.is_position_held('UVXY'))
        self.assertEqual(timing_controller.timing_state.position_entry_dates['UVXY'], test_date)
        self.assertEqual(timing_controller.timing_state.position_entry_prices['UVXY'], 50.0)
    
    def test_uvxy_strategy_rebalance_dates_generation(self):
        """Test that UVXY strategy generates appropriate rebalance dates."""
        strategy = UvxyRsiStrategy(self.strategy_config)
        timing_controller = strategy.get_timing_controller()
        
        start_date = pd.Timestamp('2023-01-01')
        end_date = pd.Timestamp('2023-01-31')
        available_dates = pd.date_range(start_date, end_date, freq='D')
        
        rebalance_dates = timing_controller.get_rebalance_dates(
            start_date, end_date, available_dates, strategy
        )
        
        # Should return all available dates for daily scanning
        self.assertEqual(len(rebalance_dates), len(available_dates))
        self.assertTrue(all(date in available_dates for date in rebalance_dates))
    
    def test_uvxy_strategy_signal_generation_conditions(self):
        """Test UVXY strategy signal generation conditions."""
        strategy = UvxyRsiStrategy(self.strategy_config)
        timing_controller = strategy.get_timing_controller()
        
        # Set up scheduled dates
        test_dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
        timing_controller.timing_state.scheduled_dates = set(test_dates)
        
        # Test signal generation on scheduled date
        test_date = test_dates[0]
        self.assertTrue(timing_controller.should_generate_signal(test_date, strategy))
        
        # Test signal generation on non-scheduled date
        non_scheduled_date = pd.Timestamp('2023-02-01')
        self.assertFalse(timing_controller.should_generate_signal(non_scheduled_date, strategy))
        
        # Test minimum holding period constraint
        timing_controller.timing_state.last_signal_date = test_dates[0]
        next_date = test_dates[1]  # Next day
        
        # With min_holding_period=1, should be able to generate signal next day
        self.assertTrue(timing_controller.should_generate_signal(next_date, strategy))
        
        # Test with higher minimum holding period
        config_with_min_holding = {
            "strategy_params": {
                "rsi_period": 14,
                "rsi_threshold": 70,
                "lookback_days": 5
            },
            "timing_config": {
                "mode": "signal_based",
                "scan_frequency": "D",
                "min_holding_period": 5
            }
        }
        
        strategy_with_min = UvxyRsiStrategy(config_with_min_holding)
        timing_controller_with_min = strategy_with_min.get_timing_controller()
        timing_controller_with_min.timing_state.scheduled_dates = set(test_dates)
        timing_controller_with_min.timing_state.last_signal_date = test_dates[0]
        
        # Should not be able to generate signal within minimum holding period
        self.assertFalse(timing_controller_with_min.should_generate_signal(test_dates[1], strategy_with_min))
        self.assertFalse(timing_controller_with_min.should_generate_signal(test_dates[4], strategy_with_min))
        self.assertTrue(timing_controller_with_min.should_generate_signal(test_dates[5], strategy_with_min))


if __name__ == '__main__':
    unittest.main()