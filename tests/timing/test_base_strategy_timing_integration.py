"""
Test timing controller integration with BaseStrategy.
"""

import unittest
import pandas as pd
import numpy as np

from src.portfolio_backtester.strategies.base_strategy import BaseStrategy
from src.portfolio_backtester.timing.time_based_timing import TimeBasedTiming
from src.portfolio_backtester.timing.signal_based_timing import SignalBasedTiming


class MockStrategy(BaseStrategy):
    """Simple test strategy for timing controller integration tests."""
    
    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: pd.DataFrame = None,
        current_date: pd.Timestamp = None,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
    ) -> pd.DataFrame:
        """Simple test implementation that returns equal weights."""
        if all_historical_data.empty:
            return pd.DataFrame()
        
        # Get asset list
        if isinstance(all_historical_data.columns, pd.MultiIndex):
            assets = all_historical_data.columns.get_level_values('Ticker').unique()
        else:
            assets = all_historical_data.columns
        
        # Return equal weights for all assets
        weights = pd.Series(1.0 / len(assets), index=assets, name=current_date)
        return pd.DataFrame([weights])


class TestBaseStrategyTimingIntegration(unittest.TestCase):
    """Test timing controller integration with BaseStrategy."""
    
    def test_default_time_based_timing_initialization(self):
        """Test that BaseStrategy initializes with time-based timing by default."""
        config = {}
        strategy = MockStrategy(config)
        
        timing_controller = strategy.get_timing_controller()
        self.assertIsInstance(timing_controller, TimeBasedTiming)
        self.assertFalse(strategy.supports_daily_signals())
    
    def test_explicit_time_based_timing_configuration(self):
        """Test explicit time-based timing configuration."""
        config = {
            'timing_config': {
                'mode': 'time_based',
                'rebalance_frequency': 'Q',
                'rebalance_offset': 1
            }
        }
        strategy = MockStrategy(config)
        
        timing_controller = strategy.get_timing_controller()
        self.assertIsInstance(timing_controller, TimeBasedTiming)
        self.assertEqual(timing_controller.frequency, 'Q')
        self.assertEqual(timing_controller.offset, 1)
        self.assertFalse(strategy.supports_daily_signals())
    
    def test_signal_based_timing_configuration(self):
        """Test signal-based timing configuration."""
        config = {
            'timing_config': {
                'mode': 'signal_based',
                'scan_frequency': 'D',
                'min_holding_period': 5,
                'max_holding_period': 30
            }
        }
        strategy = MockStrategy(config)
        
        timing_controller = strategy.get_timing_controller()
        self.assertIsInstance(timing_controller, SignalBasedTiming)
        self.assertEqual(timing_controller.scan_frequency, 'D')
        self.assertEqual(timing_controller.min_holding_period, 5)
        self.assertEqual(timing_controller.max_holding_period, 30)
        self.assertTrue(strategy.supports_daily_signals())
    
    def test_legacy_rebalance_frequency_migration(self):
        """Test backward compatibility with legacy rebalance_frequency parameter."""
        config = {
            'rebalance_frequency': 'W'
        }
        strategy = MockStrategy(config)
        
        timing_controller = strategy.get_timing_controller()
        self.assertIsInstance(timing_controller, TimeBasedTiming)
        self.assertEqual(timing_controller.frequency, 'W')
        self.assertFalse(strategy.supports_daily_signals())
    
    def test_invalid_timing_mode_fallback(self):
        """Test fallback to time-based timing for invalid configuration."""
        config = {
            'timing_config': {
                'mode': 'invalid_mode'
            }
        }
        strategy = MockStrategy(config)
        
        timing_controller = strategy.get_timing_controller()
        self.assertIsInstance(timing_controller, TimeBasedTiming)
        self.assertFalse(strategy.supports_daily_signals())
    
    def test_timing_controller_state_management(self):
        """Test that timing controller maintains state correctly."""
        config = {
            'timing_config': {
                'mode': 'signal_based',
                'scan_frequency': 'D'
            }
        }
        strategy = MockStrategy(config)
        timing_controller = strategy.get_timing_controller()
        
        # Test state reset
        timing_controller.reset_state()
        self.assertIsNone(timing_controller.timing_state.last_signal_date)
        self.assertEqual(len(timing_controller.timing_state.position_entry_dates), 0)
        
        # Test state update
        test_date = pd.Timestamp('2023-01-01')
        test_weights = pd.Series([0.5, 0.5], index=['AAPL', 'MSFT'])
        timing_controller.update_signal_state(test_date, test_weights)
        
        self.assertEqual(timing_controller.timing_state.last_signal_date, test_date)
        pd.testing.assert_series_equal(timing_controller.timing_state.last_weights, test_weights)
    
    def test_timing_controller_rebalance_dates_generation(self):
        """Test that timing controller generates rebalance dates correctly."""
        # Test time-based timing
        time_config = {
            'timing_config': {
                'mode': 'time_based',
                'rebalance_frequency': 'M'
            }
        }
        time_strategy = MockStrategy(time_config)
        time_controller = time_strategy.get_timing_controller()
        
        start_date = pd.Timestamp('2023-01-01')
        end_date = pd.Timestamp('2023-03-31')
        available_dates = pd.date_range(start_date, end_date, freq='D')
        
        rebalance_dates = time_controller.get_rebalance_dates(
            start_date, end_date, available_dates, time_strategy
        )
        
        self.assertGreater(len(rebalance_dates), 0)
        self.assertTrue(all(date >= start_date and date <= end_date + pd.Timedelta(days=7) for date in rebalance_dates))
        
        # Test signal-based timing
        signal_config = {
            'timing_config': {
                'mode': 'signal_based',
                'scan_frequency': 'D'
            }
        }
        signal_strategy = MockStrategy(signal_config)
        signal_controller = signal_strategy.get_timing_controller()
        
        signal_dates = signal_controller.get_rebalance_dates(
            start_date, end_date, available_dates, signal_strategy
        )
        
        # Should return all available dates for daily scanning
        self.assertEqual(len(signal_dates), len(available_dates))
    
    def test_backward_compatibility_with_existing_strategies(self):
        """Test that existing strategies without timing_config still work."""
        # Test with empty config (should default to time-based)
        empty_config = {}
        strategy = MockStrategy(empty_config)
        self.assertIsInstance(strategy.get_timing_controller(), TimeBasedTiming)
        
        # Test with legacy config
        legacy_config = {
            'rebalance_frequency': 'Q',
            'some_other_param': 'value'
        }
        legacy_strategy = MockStrategy(legacy_config)
        timing_controller = legacy_strategy.get_timing_controller()
        self.assertIsInstance(timing_controller, TimeBasedTiming)
        self.assertEqual(timing_controller.frequency, 'Q')


class MockStrategyWithCustomSupportsDaily(BaseStrategy):
    """Test strategy that overrides supports_daily_signals for legacy compatibility."""
    
    def supports_daily_signals(self) -> bool:
        """Override to return True for legacy compatibility test."""
        return True
    
    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: pd.DataFrame = None,
        current_date: pd.Timestamp = None,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
    ) -> pd.DataFrame:
        """Simple test implementation."""
        if all_historical_data.empty:
            return pd.DataFrame()
        
        assets = all_historical_data.columns if not isinstance(all_historical_data.columns, pd.MultiIndex) else all_historical_data.columns.get_level_values('Ticker').unique()
        weights = pd.Series(1.0 / len(assets), index=assets, name=current_date)
        return pd.DataFrame([weights])


class TestLegacySupportsDaily(unittest.TestCase):
    """Test legacy supports_daily_signals method compatibility."""
    
    def test_legacy_supports_daily_signals_migration(self):
        """Test that strategies with overridden supports_daily_signals are migrated to signal-based timing."""
        config = {}
        strategy = MockStrategyWithCustomSupportsDaily(config)
        
        # The strategy should be migrated to signal-based timing
        timing_controller = strategy.get_timing_controller()
        self.assertIsInstance(timing_controller, SignalBasedTiming)
        
        # The supports_daily_signals method should still work
        self.assertTrue(strategy.supports_daily_signals())


if __name__ == '__main__':
    unittest.main()