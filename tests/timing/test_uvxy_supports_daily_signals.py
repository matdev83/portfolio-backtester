"""
Test that UVXY strategy correctly supports daily signals through timing controller.
"""

import unittest
from src.portfolio_backtester.strategies.uvxy_rsi_strategy import UvxyRsiStrategy
from src.portfolio_backtester.timing.signal_based_timing import SignalBasedTiming


class TestUvxySupportsDaily(unittest.TestCase):
    """Test UVXY strategy daily signals support."""
    
    def test_uvxy_supports_daily_signals_through_timing_controller(self):
        """Test that UVXY strategy supports daily signals via timing controller."""
        config = {
            "strategy_params": {
                "rsi_period": 14,
                "rsi_threshold": 70,
                "lookback_days": 5
            }
        }
        
        strategy = UvxyRsiStrategy(config)
        
        # Should support daily signals
        self.assertTrue(strategy.supports_daily_signals())
        
        # Should use signal-based timing controller
        timing_controller = strategy.get_timing_controller()
        self.assertIsInstance(timing_controller, SignalBasedTiming)
        
        # Should have daily scan frequency
        self.assertEqual(timing_controller.scan_frequency, 'D')
    
    def test_uvxy_legacy_method_still_works(self):
        """Test that the legacy supports_daily_signals method still returns True."""
        config = {}
        strategy = UvxyRsiStrategy(config)
        
        # The legacy method should still work
        self.assertTrue(strategy.supports_daily_signals())
        
        # But it should be using the timing controller under the hood
        timing_controller = strategy.get_timing_controller()
        self.assertIsInstance(timing_controller, SignalBasedTiming)


if __name__ == '__main__':
    unittest.main()