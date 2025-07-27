"""
Integration tests for UVXY strategy migration to new timing framework.

This test suite verifies that the UVXY strategy migration maintains
identical behavior while using the new timing controller system.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.portfolio_backtester.strategies.uvxy_rsi_strategy import UvxyRsiStrategy
from src.portfolio_backtester.timing.signal_based_timing import SignalBasedTiming


class TestUvxyMigrationIntegration(unittest.TestCase):
    """Test UVXY strategy migration integration."""
    
    def setUp(self):
        """Set up test data."""
        # Legacy-style configuration (no timing_config)
        self.legacy_config = {
            "strategy_params": {
                "rsi_period": 2,
                "rsi_threshold": 30.0,
                "price_column_asset": "Close",
                "price_column_benchmark": "Close",
                "long_only": False,
            }
        }
        
        # New-style configuration with explicit timing_config
        self.new_config = {
            "strategy_params": {
                "rsi_period": 2,
                "rsi_threshold": 30.0,
                "price_column_asset": "Close",
                "price_column_benchmark": "Close",
                "long_only": False,
            },
            "timing_config": {
                "mode": "signal_based",
                "scan_frequency": "D",
                "min_holding_period": 1,
                "max_holding_period": 1  # UVXY strategy has 1-day holding period
            }
        }
    
    def test_legacy_config_automatically_migrates_to_signal_based(self):
        """Test that legacy UVXY config automatically migrates to signal-based timing."""
        strategy = UvxyRsiStrategy(self.legacy_config)
        
        # Should be automatically detected as signal-based
        timing_controller = strategy.get_timing_controller()
        self.assertIsInstance(timing_controller, SignalBasedTiming)
        
        # Should have appropriate configuration
        self.assertEqual(timing_controller.scan_frequency, 'D')
        self.assertEqual(timing_controller.min_holding_period, 1)
        self.assertEqual(timing_controller.max_holding_period, 1)  # UVXY-specific default
        
        # Strategy should support daily signals
        self.assertTrue(strategy.supports_daily_signals())
    
    def test_explicit_timing_config_works_correctly(self):
        """Test that explicit timing configuration works correctly."""
        strategy = UvxyRsiStrategy(self.new_config)
        
        timing_controller = strategy.get_timing_controller()
        self.assertIsInstance(timing_controller, SignalBasedTiming)
        
        # Should use explicit configuration
        self.assertEqual(timing_controller.scan_frequency, 'D')
        self.assertEqual(timing_controller.min_holding_period, 1)
        self.assertEqual(timing_controller.max_holding_period, 1)
        
        self.assertTrue(strategy.supports_daily_signals())
    
    def test_timing_controller_integration_with_strategy_methods(self):
        """Test that timing controller integrates correctly with strategy methods."""
        strategy = UvxyRsiStrategy(self.legacy_config)
        timing_controller = strategy.get_timing_controller()
        
        # Test rebalance dates generation
        start_date = pd.Timestamp('2023-01-01')
        end_date = pd.Timestamp('2023-01-10')
        available_dates = pd.date_range(start_date, end_date, freq='D')
        
        rebalance_dates = timing_controller.get_rebalance_dates(
            start_date, end_date, available_dates, strategy
        )
        
        # Should return all available dates for daily scanning
        self.assertEqual(len(rebalance_dates), len(available_dates))
        self.assertTrue(all(date in available_dates for date in rebalance_dates))
    
    def test_signal_generation_with_timing_controller_state(self):
        """Test that signal generation works with timing controller state management."""
        strategy = UvxyRsiStrategy(self.legacy_config)
        timing_controller = strategy.get_timing_controller()
        
        # Create test data
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        current_date = dates[2]
        
        # UVXY universe data
        uvxy_columns = pd.MultiIndex.from_product([['UVXY'], ['Close']], 
                                                names=['Ticker', 'Field'])
        uvxy_data = np.random.randn(5, 1) * 0.02 + 50
        all_historical_data = pd.DataFrame(uvxy_data, index=dates, columns=uvxy_columns)
        
        # SPY data with low RSI to trigger signal
        spy_columns = pd.MultiIndex.from_product([['SPY'], ['Close']], 
                                                names=['Ticker', 'Field'])
        spy_prices = [100, 95, 90, 85, 80]  # Declining trend for low RSI
        spy_data = np.array(spy_prices).reshape(-1, 1)
        non_universe_data = pd.DataFrame(spy_data, index=dates, columns=spy_columns)
        
        benchmark_data = pd.DataFrame(index=dates)
        
        # Test signal generation
        signals = strategy.generate_signals(
            all_historical_data=all_historical_data,
            benchmark_historical_data=benchmark_data,
            current_date=current_date,
            non_universe_historical_data=non_universe_data
        )
        
        # Should generate valid signals
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('UVXY', signals.columns)
        
        # Test timing controller state management
        if signals.loc[current_date, 'UVXY'] != 0:
            # If signal was generated, update timing controller state
            weights = signals.loc[current_date]
            prices = pd.Series([50.0], index=['UVXY'])
            
            timing_controller.update_signal_state(current_date, weights)
            timing_controller.update_position_state(current_date, weights, prices)
            
            # Check state was updated
            self.assertEqual(timing_controller.timing_state.last_signal_date, current_date)
            if weights['UVXY'] != 0:
                self.assertTrue(timing_controller.is_position_held('UVXY'))
    
    def test_backward_compatibility_with_legacy_behavior(self):
        """Test that migrated strategy maintains backward compatibility."""
        # Create two strategies - one with legacy config, one with explicit config
        legacy_strategy = UvxyRsiStrategy(self.legacy_config)
        new_strategy = UvxyRsiStrategy(self.new_config)
        
        # Both should have the same timing controller type and configuration
        legacy_controller = legacy_strategy.get_timing_controller()
        new_controller = new_strategy.get_timing_controller()
        
        self.assertEqual(type(legacy_controller), type(new_controller))
        self.assertEqual(legacy_controller.scan_frequency, new_controller.scan_frequency)
        self.assertEqual(legacy_controller.min_holding_period, new_controller.min_holding_period)
        self.assertEqual(legacy_controller.max_holding_period, new_controller.max_holding_period)
        
        # Both should support daily signals
        self.assertEqual(legacy_strategy.supports_daily_signals(), new_strategy.supports_daily_signals())
        self.assertTrue(legacy_strategy.supports_daily_signals())
        self.assertTrue(new_strategy.supports_daily_signals())
    
    def test_strategy_configuration_preserved_after_migration(self):
        """Test that strategy configuration is preserved after migration."""
        original_config = self.legacy_config.copy()
        strategy = UvxyRsiStrategy(original_config)
        
        # Original strategy params should be preserved
        strategy_params = strategy.strategy_config.get('strategy_params', {})
        self.assertEqual(strategy_params['rsi_period'], 2)
        self.assertEqual(strategy_params['rsi_threshold'], 30.0)
        self.assertFalse(strategy_params['long_only'])
        
        # Timing config should be added
        self.assertIn('timing_config', strategy.strategy_config)
        timing_config = strategy.strategy_config['timing_config']
        self.assertEqual(timing_config['mode'], 'signal_based')
        self.assertEqual(timing_config['scan_frequency'], 'D')
    
    def test_tunable_parameters_unchanged(self):
        """Test that tunable parameters are unchanged after migration."""
        strategy = UvxyRsiStrategy(self.legacy_config)
        
        tunable_params = strategy.tunable_parameters()
        expected_params = {"rsi_period", "rsi_threshold"}
        self.assertEqual(tunable_params, expected_params)
    
    def test_non_universe_data_requirements_unchanged(self):
        """Test that non-universe data requirements are unchanged."""
        strategy = UvxyRsiStrategy(self.legacy_config)
        
        requirements = strategy.get_non_universe_data_requirements()
        self.assertEqual(requirements, ["SPY"])
    
    def test_timing_controller_respects_holding_period_constraints(self):
        """Test that timing controller respects UVXY-specific holding period constraints."""
        strategy = UvxyRsiStrategy(self.legacy_config)
        timing_controller = strategy.get_timing_controller()
        
        # Set up test dates
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        timing_controller.timing_state.scheduled_dates = set(dates)
        
        # Test initial signal generation (should be allowed)
        self.assertTrue(timing_controller.should_generate_signal(dates[0], strategy))
        
        # Simulate signal generation
        test_weights = pd.Series([-1.0], index=['UVXY'])
        timing_controller.update_signal_state(dates[0], test_weights)
        
        # Test next day (should be allowed due to max_holding_period=1)
        self.assertTrue(timing_controller.should_generate_signal(dates[1], strategy))
        
        # Test that position tracking works
        test_prices = pd.Series([50.0], index=['UVXY'])
        timing_controller.update_position_state(dates[0], test_weights, test_prices)
        
        self.assertTrue(timing_controller.is_position_held('UVXY'))
        self.assertEqual(timing_controller.get_position_holding_days('UVXY', dates[1]), 1)


if __name__ == '__main__':
    unittest.main()