import unittest
import pandas as pd
import numpy as np
from src.portfolio_backtester.strategies.calmar_momentum_strategy import CalmarMomentumStrategy
# Removed CalmarRatio and precompute_features imports


class TestCalmarMomentumStrategy(unittest.TestCase):

    def setUp(self):
        """Set up test data and strategy configuration."""
        # Using a rolling window that makes sense for monthly data, e.g., 6 months.
        # The Calmar calculation itself involves pct_change (1 period) + rolling (window-1 periods).
        # So, first `rolling_window` periods of Calmar ratio will be NaN.
        self.rolling_window_periods = 6
        self.strategy_config = {
            'rolling_window': self.rolling_window_periods, # Number of periods for Calmar calculation
            'num_holdings': 2, # Select top 2 assets
            # 'top_decile_fraction': 0.1, # Alternatively, use num_holdings
            'smoothing_lambda': 0.5,
            'leverage': 1.0,
            'long_only': True,
            # 'sma_filter_window': None # SMA filter is not part of this strategy now
            'apply_trading_lag': False # Test without lag first
        }
        self.strategy = CalmarMomentumStrategy(self.strategy_config)

        # Create sample data: Monthly prices for 2 years
        dates = pd.date_range('2020-01-01', periods=24, freq='ME') # freq='M' in pandas > 2.0, 'ME' for month-end
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        np.random.seed(42)
        data = {}
        for ticker in tickers:
            # Create data with some positive trend for Calmar to be meaningful
            base_trend = np.linspace(0, 0.1 * (1 if ticker in ['AAPL', 'MSFT'] else -0.05) , len(dates))
            returns = np.random.normal(0.01, 0.05, len(dates)) + base_trend
            prices = 100 * (1 + returns).cumprod()
            data[ticker] = prices
        
        # self.data is now expected to be 'Close' prices
        self.all_historical_data = pd.DataFrame(data, index=dates)
        
        # Benchmark data (not directly used by Calmar calc but needed for generate_signals signature)
        benchmark_returns = np.random.normal(0.008, 0.04, len(dates))
        self.benchmark_historical_data = pd.DataFrame({'SPY': 100 * (1 + benchmark_returns).cumprod()}, index=dates)


    def test_internal_calculate_calmar_ratio(self):
        """Test the internal _calculate_calmar_ratio method."""
        # The method is called _calculate_calmar_ratio and then _calculate_single_asset_calmar
        # We test the public interface _calculate_calmar_ratio

        # Provide data that has 'Close' prices (self.all_historical_data is already this)
        rolling_calmar_df = self.strategy._calculate_calmar_ratio(self.all_historical_data)

        self.assertEqual(rolling_calmar_df.shape, self.all_historical_data.shape)

        # Calmar ratio calculation involves pct_change() (1 NaN) and then rolling(window)
        # So, the first 'rolling_window_periods' will have NaNs.
        # Example: window=6. prices[0..5]. rets[0] is NaN. rets[1..5]. roll_mean(rets[1..5]) needs 6 valid rets.
        # So, data up to index 5 (6 periods) will result in NaN for Calmar at index 5.
        # The first valid Calmar will be at index `rolling_window_periods`.
        self.assertTrue(rolling_calmar_df.iloc[:self.rolling_window_periods-1].isnull().all().all())

        # Values from rolling_window_periods onwards should be finite (unless all returns are zero, etc.)
        # Given the data generation, they should be finite.
        self.assertTrue(np.isfinite(rolling_calmar_df.iloc[self.rolling_window_periods:]).all().all())

    def test_calculate_candidate_weights(self):
        """Test _calculate_candidate_weights helper."""
        # This tests a BaseStrategy helper, but through the lens of this strategy's config
        self.strategy.num_holdings = 2 # Explicitly set for this test clarity
        self.strategy.strategy_config['num_holdings'] = 2
        self.strategy.strategy_config['top_decile_fraction'] = None # Ensure num_holdings is used

        look = pd.Series([0.5, 1.2, 0.8, 1.5, 0.3], 
                        index=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
        
        weights = self.strategy._calculate_candidate_weights(look)
        
        self.assertAlmostEqual(weights.sum(), 1.0, places=6)
        self.assertTrue((weights >= 0).all())
        
        expected_holdings = 2 # Based on self.strategy.num_holdings
        actual_holdings = (weights > 1e-6).sum() # Use tolerance for float comparison
        self.assertEqual(actual_holdings, expected_holdings)
        self.assertEqual(weights['AMZN'], 0.5) # AMZN (1.5) and MSFT (1.2) should be picked
        self.assertEqual(weights['MSFT'], 0.5)


    def test_apply_leverage_and_smoothing(self):
        """Test _apply_leverage_and_smoothing helper."""
        # This also tests a BaseStrategy helper
        self.strategy.strategy_config['smoothing_lambda'] = 0.5 # ensure it's set
        self.strategy.strategy_config['leverage'] = 1.0

        cand = pd.Series([0.5, 0.5, 0.0, 0.0, 0.0], 
                        index=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
        w_prev = pd.Series([0.3, 0.3, 0.4, 0.0, 0.0], 
                          index=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
        
        w_new = self.strategy._apply_leverage_and_smoothing(cand, w_prev)
        
        # Expected: 0.5 * 0.3 + 0.5 * 0.5 = 0.15 + 0.25 = 0.4 for AAPL and MSFT
        # Expected: 0.5 * 0.4 + 0.5 * 0.0 = 0.2 for GOOGL
        self.assertAlmostEqual(w_new['AAPL'], 0.4)
        self.assertAlmostEqual(w_new['MSFT'], 0.4)
        self.assertAlmostEqual(w_new['GOOGL'], 0.2)
        self.assertTrue(np.isfinite(w_new).all())

    def test_generate_signals_single_date(self):
        """Test signal generation for a single specific date."""
        # Pick a date where Calmar ratio should be computable
        # rolling_window_periods = 6. First calmar at index 5 (6th month end).
        # So data up to index 5 is needed. current_date can be index 5.
        current_date = self.all_historical_data.index[self.rolling_window_periods]

        # Reset weights_history for consistent test results if smoothing is involved
        self.strategy.weights_history = pd.DataFrame()

        signals_for_date = self.strategy.generate_signals(
            self.all_historical_data, # Full history for lookbacks
            self.benchmark_historical_data, # Full benchmark history
            current_date
        )

        self.assertEqual(signals_for_date.shape, (1, len(self.all_historical_data.columns)))
        self.assertTrue(np.isfinite(signals_for_date).all().all())
        if self.strategy.long_only:
            self.assertTrue((signals_for_date.values >= 0).all()) # .values to avoid future warnings

        # Check sum of weights (should be <= leverage, here 1.0 for long_only)
        self.assertAlmostEqual(signals_for_date.sum().sum(), 1.0, places=5) # num_holdings > 0

        # Test for an early date where Calmar cannot be computed fully
        early_date = self.all_historical_data.index[self.rolling_window_periods - 2] # e.g. index 4
        self.strategy.weights_history = pd.DataFrame() # Reset
        signals_early_date = self.strategy.generate_signals(
            self.all_historical_data, self.benchmark_historical_data, early_date
        )
        # Expect zero weights as Calmar ratio would be NaN or based on insufficient data
        self.assertTrue((signals_early_date.values == 0).all())

    def test_generate_signals_iteratively_with_smoothing(self):
        """Test signal generation iteratively to check smoothing."""
        all_generated_signals = []
        self.strategy.weights_history = pd.DataFrame() # Ensure clean slate for this test

        # Iterate from a point where Calmar can be calculated
        start_idx = self.rolling_window_periods

        # Let's check weights for two consecutive dates
        date1 = self.all_historical_data.index[start_idx]
        date2 = self.all_historical_data.index[start_idx + 1]

        # Signals for date1 (populates self.strategy.weights_history)
        signals_date1 = self.strategy.generate_signals(
            self.all_historical_data, self.benchmark_historical_data, date1
        )

        # Now generate for date2, which should use date1's weights for smoothing
        signals_date2 = self.strategy.generate_signals(
            self.all_historical_data, self.benchmark_historical_data, date2
        )

        self.assertEqual(signals_date1.shape, (1, len(self.all_historical_data.columns)))
        self.assertEqual(signals_date2.shape, (1, len(self.all_historical_data.columns)))

        # Crude check: if smoothing is 0.5, and candidate weights change,
        # date2 weights should be different from date1 if candidates were different.
        # And if candidates were same, date2 weights should be closer to candidates than date1's were (if date1 was smoothed from zero)
        # This requires knowing the candidate weights, which is complex to assert directly here.
        # A simpler check: ensure weights are not always zero (unless underlying data makes them so)
        # and that they sum to 1 (for long_only, num_holdings > 0).
        self.assertGreater(signals_date1.abs().sum().sum(), 0) # Assuming some assets are picked
        self.assertGreater(signals_date2.abs().sum().sum(), 0)
        self.assertAlmostEqual(signals_date1.sum().sum(), 1.0, places=5)
        self.assertAlmostEqual(signals_date2.sum().sum(), 1.0, places=5)

        # A more specific check would be needed if we knew the exact Calmar values and candidates.
        # For now, this verifies the mechanism runs and produces valid shaped outputs.

    # Removed test_generate_signals_with_sma_filter as SMA filter is not part of this strategy.

    def test_edge_cases_generate_signals(self):
        """Test edge cases for generate_signals."""
        self.strategy.weights_history = pd.DataFrame() # Reset

        # Test with very small dataset (not enough for full Calmar)
        # Data for self.rolling_window_periods (e.g., 6 months) allows 1 Calmar value at the end.
        small_data_len = self.rolling_window_periods
        small_data = self.all_historical_data.iloc[:small_data_len]
        small_benchmark = self.benchmark_historical_data.iloc[:small_data_len]
        current_date_small = small_data.index[-1] # Last date of this small set

        signals = self.strategy.generate_signals(small_data, small_benchmark, current_date_small)
        self.assertEqual(signals.shape, (1, len(small_data.columns)))
        # With just enough data for one Calmar calculation, it should produce valid signals.
        # (The _calculate_calmar_ratio handles NaNs from pct_change and rolling)
        # If it does produce signals, sum should be 1. Otherwise, sum is 0.
        # self.assertTrue((signals.values == 0).all()) # if not enough data for any Calmar
        self.assertTrue(np.isfinite(signals.values).all())


        # Test with all zero returns (flat prices)
        self.strategy.weights_history = pd.DataFrame() # Reset
        zero_data = pd.DataFrame(100.0, index=self.all_historical_data.index,
                                 columns=self.all_historical_data.columns)
        current_date_zero = zero_data.index[self.rolling_window_periods + 2] # Some date after initial NaNs

        signals_zero = self.strategy.generate_signals(zero_data, self.benchmark_historical_data, current_date_zero)
        self.assertEqual(signals_zero.shape, (1, len(zero_data.columns)))
        self.assertTrue(np.isfinite(signals_zero.values).all())
        # For flat prices, Calmar ratio is typically 0 for all assets.
        # _calculate_candidate_weights might pick num_holdings assets due to tie-breaking in nlargest.
        # So, weights might not be all zero. They should sum to 1.
        if self.strategy.num_holdings is not None and self.strategy.num_holdings > 0:
             self.assertAlmostEqual(signals_zero.sum().sum(), 1.0, places=5)
        else: # if num_holdings is 0 or None and top_decile_fraction is 0
             self.assertAlmostEqual(signals_zero.sum().sum(), 0.0, places=5)


if __name__ == '__main__':
    unittest.main()