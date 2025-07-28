"""
Consolidated tests for UVXY RSI strategy timing and migration.

These tests verify that the UVXYRsiStrategy integrates correctly with the new timing framework,
including legacy configuration migration, signal generation, and state management.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.portfolio_backtester.strategies.uvxy_rsi_strategy import UvxyRsiStrategy
from src.portfolio_backtester.timing.signal_based_timing import SignalBasedTiming


class TestUvxyRsiStrategyTiming(unittest.TestCase):
    """Test UVXY RSI strategy timing, migration, and integration."""

    def setUp(self):
        """Set up test data and configurations."""
        self.legacy_config = {
            "strategy_params": {
                "rsi_period": 2,
                "rsi_threshold": 30.0,
                "price_column_asset": "Close",
                "price_column_benchmark": "Close",
                "long_only": False,
            }
        }
        self.new_config = {
            "strategy_params": {
                "rsi_period": 2,
                "rsi_threshold": 30.0,
            },
            "timing_config": {
                "mode": "signal_based",
                "scan_frequency": "D",
                "min_holding_period": 1,
                "max_holding_period": 1
            }
        }

    def test_legacy_config_migrates_to_signal_based_timing(self):
        """Test that a legacy UVXY config automatically migrates to signal-based timing."""
        strategy = UvxyRsiStrategy(self.legacy_config)
        timing_controller = strategy.get_timing_controller()
        self.assertIsInstance(timing_controller, SignalBasedTiming)
        self.assertEqual(timing_controller.scan_frequency, 'D')
        self.assertEqual(timing_controller.min_holding_period, 1)
        self.assertEqual(timing_controller.max_holding_period, 1)
        self.assertTrue(strategy.supports_daily_signals())

    def test_explicit_timing_config_is_honored(self):
        """Test that an explicit timing configuration is honored by the strategy."""
        config_with_override = {
            "strategy_params": {"rsi_period": 5},
            "timing_config": {
                "mode": "signal_based",
                "scan_frequency": "D",
                "min_holding_period": 3,
                "max_holding_period": 10
            }
        }
        strategy = UvxyRsiStrategy(config_with_override)
        timing_controller = strategy.get_timing_controller()
        self.assertIsInstance(timing_controller, SignalBasedTiming)
        self.assertEqual(timing_controller.min_holding_period, 3)
        self.assertEqual(timing_controller.max_holding_period, 10)

    def test_signal_generation_and_state_management(self):
        """Test the integration of signal generation and timing state updates."""
        strategy = UvxyRsiStrategy(self.legacy_config)
        timing_controller = strategy.get_timing_controller()

        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        current_date = dates[2]

        uvxy_columns = pd.MultiIndex.from_product([['UVXY'], ['Close']], names=['Ticker', 'Field'])
        uvxy_data = pd.DataFrame(np.random.randn(5, 1) * 0.02 + 50, index=dates, columns=uvxy_columns)
    
        spy_columns = pd.MultiIndex.from_product([['SPY'], ['Close']], names=['Ticker', 'Field'])
        spy_prices = [100, 95, 90, 85, 80]  # Declining trend for low RSI
        spy_data = pd.DataFrame(np.array(spy_prices).reshape(-1, 1), index=dates, columns=spy_columns)

        benchmark_data = pd.DataFrame(index=dates)

        signals = strategy.generate_signals(
            all_historical_data=uvxy_data,
            benchmark_historical_data=benchmark_data,
            current_date=current_date,
            non_universe_historical_data=spy_data
        )

        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('UVXY', signals.columns)

        if signals.loc[current_date, 'UVXY'] != 0:
            weights = signals.loc[current_date]
            prices = pd.Series([50.0], index=['UVXY'])
            timing_controller.update_signal_state(current_date, weights)
            timing_controller.update_position_state(current_date, weights, prices)

            self.assertEqual(timing_controller.timing_state.last_signal_date, current_date)
            if weights['UVXY'] != 0:
                self.assertTrue(timing_controller.is_position_held('UVXY'))

    def test_tunable_parameters_are_correct(self):
        """Test that the tunable parameters for the strategy are correctly defined."""
        strategy = UvxyRsiStrategy(self.legacy_config)
        tunable_params = strategy.tunable_parameters()
        expected_params = {"rsi_period", "rsi_threshold"}
        self.assertEqual(tunable_params, expected_params)

    def test_non_universe_data_requirements_are_correct(self):
        """Test that the non-universe data requirements are correctly defined."""
        strategy = UvxyRsiStrategy(self.legacy_config)
        requirements = strategy.get_non_universe_data_requirements()
        self.assertEqual(requirements, ["SPY"])

    def test_timing_controller_respects_holding_period(self):
        """Test that the timing controller correctly respects holding period constraints."""
        strategy = UvxyRsiStrategy(self.new_config)  # Use config with explicit holding periods
        timing_controller = strategy.get_timing_controller()

        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        timing_controller.timing_state.scheduled_dates = set(dates)

        # Initial signal generation should be allowed
        self.assertTrue(timing_controller.should_generate_signal(dates[0], strategy))

        # Simulate signal generation and state update
        timing_controller.update_signal_state(dates[0], pd.Series([-1.0], index=['UVXY']))

        # Next day should also be allowed due to max_holding_period=1
        self.assertTrue(timing_controller.should_generate_signal(dates[1], strategy))


if __name__ == '__main__':
    unittest.main()
