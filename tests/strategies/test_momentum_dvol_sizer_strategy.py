import unittest
import pandas as pd
import numpy as np

from src.portfolio_backtester.strategies.momentum_dvol_sizer_strategy import MomentumDvolSizerStrategy
# Removed precompute_features import
# from src.portfolio_backtester.portfolio.position_sizer import get_position_sizer # For more advanced test

class TestMomentumDvolSizerStrategy(unittest.TestCase):
    def setUp(self):
        # Monthly data for higher-level signal generation (if MomentumStrategy resamples)
        self.monthly_dates = pd.to_datetime(pd.date_range(start='2020-01-01', periods=12, freq='ME'))
        self.monthly_prices = pd.DataFrame({
            'A': np.linspace(100, 200, 12),
            'B': np.linspace(100, 50, 12),
        }, index=self.monthly_dates)

        # Daily data which the sizer needs, and MomentumStrategy might use for calculations
        # Create more data points for daily; e.g. 12 months * 21 days approx
        num_daily_periods = 12 * 21
        self.daily_dates = pd.to_datetime(pd.date_range(start='2020-01-01', periods=num_daily_periods, freq='B')) # Business days

        np.random.seed(42)
        self.daily_historical_data = pd.DataFrame({
            'A': 100 + np.random.randn(num_daily_periods).cumsum() * 0.5 + np.linspace(0, 50, num_daily_periods), # Trend up
            'B': 100 + np.random.randn(num_daily_periods).cumsum() * 0.5 + np.linspace(0, -25, num_daily_periods), # Trend down
        }, index=self.daily_dates)

        self.benchmark_historical_data = pd.DataFrame( # Needs to be DataFrame
            {'SPY': 100 + np.random.randn(num_daily_periods).cumsum() * 0.3 + np.linspace(0, 10, num_daily_periods)},
            index=self.daily_dates
        )

        self.default_strategy_config = {
            'lookback_months': 3, # This would be used by underlying MomentumStrategy
            'sizer_dvol_window': 60, # Example: 60 days for downside vol window
            'target_volatility': 0.15, # Example: 15% target annual vol
            'max_leverage': 1.5,
            'num_holdings': 2 # For the MomentumStrategy part
        }


    def test_resolve_and_defaults(self):
        """Test that strategy initializes with correct position sizer default."""
        # Pass only a minimal config to see if defaults are set
        strategy = MomentumDvolSizerStrategy({'lookback_months': 3})
        self.assertEqual(strategy.strategy_config.get('position_sizer'), 'rolling_downside_volatility')
        self.assertIsNotNone(strategy.strategy_config.get('target_volatility')) # Check if default is set
        self.assertIsNotNone(strategy.strategy_config.get('max_leverage'))

    def test_generate_signals_smoke(self):
        """
        Smoke test for generate_signals (inherited from MomentumStrategy).
        This test primarily ensures the method runs with the new signature and daily data.
        It does NOT test the actual effect of the downside volatility sizer, as that's
        applied by the Backtester after this method.
        """
        strategy = MomentumDvolSizerStrategy(self.default_strategy_config)

        # The generate_signals is inherited from MomentumStrategy.
        # It should produce initial (un-sized) signals.
        # We test for a single date.
        current_date = self.daily_historical_data.index[self.default_strategy_config['lookback_months'] * 21 + 5] # Ensure enough lookback

        try:
            # MomentumStrategy's generate_signals is called here
            initial_signals = strategy.generate_signals(
                self.daily_historical_data, # Pass daily data as all_historical_data
                self.benchmark_historical_data,
                current_date
            )
            self.assertIsInstance(initial_signals, pd.DataFrame)
            self.assertEqual(initial_signals.shape, (1, len(self.daily_historical_data.columns)))
            self.assertTrue(np.isfinite(initial_signals.values).all())
        except Exception as e:
            self.fail(f"generate_signals (from MomentumStrategy) raised an exception: {e}")

    # To truly test the MomentumDvolSizerStrategy's effect, one would need to:
    # 1. Get initial signals (as above).
    # 2. Get the sizer: `sizer = get_position_sizer(strategy.strategy_config['position_sizer'])`
    # 3. Prepare all necessary inputs for the sizer including daily_prices_for_vol, monthly_prices, etc.
    #    The `prices` argument to the sizer is monthly prices aligned with signals.
    #    The `daily_prices_for_vol` is the daily price data.
    #    This implies that `MomentumStrategy` must produce monthly signals or that
    #    the sizer or backtester handles the alignment.
    # 4. Call `sized_weights = sizer(signals=initial_signals_monthly, prices=monthly_prices, ..., daily_prices_for_vol=self.daily_historical_data, ...)`
    # 5. Assert properties of `sized_weights`.
    # This is more of an integration test or requires careful mocking of Backtester behavior.

if __name__ == '__main__':
    unittest.main()

