"""
Consolidated timing framework compatibility tests.
Merges functionality from test_timing_framework_compatibility.py and test_state_backward_compatibility.py
"""

import unittest
import pytest
import pandas as pd

from portfolio_backtester.strategies.base.base_strategy import BaseStrategy
from portfolio_backtester.timing.time_based_timing import TimeBasedTiming
from portfolio_backtester.timing.signal_based_timing import SignalBasedTiming
from portfolio_backtester.interfaces.timing_state_interface import create_timing_state


class MockStrategy(BaseStrategy):
    """Mock strategy for testing compatibility."""
    
    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: pd.DataFrame = None,
        current_date: pd.Timestamp = None,
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
    ) -> pd.DataFrame:
        if all_historical_data.empty:
            return pd.DataFrame()
        
        if isinstance(all_historical_data.columns, pd.MultiIndex):
            assets = all_historical_data.columns.get_level_values('Ticker').unique()
        else:
            assets = all_historical_data.columns
        
        weights = pd.Series(1.0 / len(assets), index=assets, name=current_date)
        return pd.DataFrame([weights])




class TestTimingFrameworkCompatibility(unittest.TestCase):
    """Test timing framework compatibility."""
    
    def setUp(self):
        """Set up test environment."""
        self.mock_strategy = MockStrategy({})

    def test_default_time_based_timing_initialization(self):
        """Test default time-based timing initialization."""
        timing_controller = self.mock_strategy.get_timing_controller()
        self.assertIsInstance(timing_controller, TimeBasedTiming)
        self.assertFalse(self.mock_strategy.supports_daily_signals())

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


class TestStateBackwardCompatibility:
    """Test state backward compatibility with legacy methods."""
    
    def setup_method(self):
        """Set up test data."""
        self.state = create_timing_state()
        self.test_date = pd.Timestamp('2023-01-01')
        self.assets = ['AAPL', 'MSFT']
        self.weights = pd.Series([0.6, 0.4], index=self.assets)
        self.prices = pd.Series([150.0, 250.0], index=self.assets)
    
    def test_legacy_methods_still_work(self):
        """Test that legacy methods still work correctly."""
        # Update positions using new method
        self.state.update_positions(self.test_date, self.weights, self.prices)
        
        # Test legacy methods
        assert self.state.is_position_held('AAPL')
        assert self.state.is_position_held('MSFT')
        assert not self.state.is_position_held('GOOGL')
        
        assert self.state.get_position_holding_days('AAPL', self.test_date) == 0
        
        held_assets = self.state.get_held_assets()
        assert 'AAPL' in held_assets
        assert 'MSFT' in held_assets
        
        # Test legacy dictionaries are populated
        assert len(self.state.position_entry_dates) == 2
        assert len(self.state.position_entry_prices) == 2
        assert len(self.state.consecutive_periods) == 2
    
    def test_enhanced_methods_work_with_legacy_data(self):
        """Test enhanced methods work when only legacy data exists."""
        # Manually populate legacy data (simulating old state)
        self.state.position_entry_dates['AAPL'] = self.test_date
        self.state.position_entry_prices['AAPL'] = 150.0
        self.state.consecutive_periods['AAPL'] = 3
        
        # Enhanced methods should still work
        assert self.state.is_position_held('AAPL')
        assert self.state.get_position_holding_days('AAPL', self.test_date + pd.Timedelta(days=5)) == 5
        assert self.state.get_consecutive_periods('AAPL') == 3
        
        held_assets = self.state.get_held_assets()
        assert 'AAPL' in held_assets





if __name__ == '__main__':
    pytest.main([__file__, '-v'])