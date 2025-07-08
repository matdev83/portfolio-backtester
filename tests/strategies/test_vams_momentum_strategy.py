import unittest
import pandas as pd
import numpy as np
from src.portfolio_backtester.strategies.vams_momentum_strategy import VAMSMomentumStrategy
# Removed VAMS feature import and precompute_features


class TestVAMSMomentumStrategy(unittest.TestCase): # Changed class name

    def setUp(self):
        """Set up test data and strategy configuration for VAMSMomentumStrategy (DPVAMS)."""
        self.lookback_periods = 6  # Number of periods (months in this test data)
        self.strategy_config = {
            'lookback_months': self.lookback_periods, # VAMSMomentumStrategy uses this name
            'alpha': 0.5,  # Key parameter for DPVAMS
            'num_holdings': 2,
            'smoothing_lambda': 0.5,
            'leverage': 1.0,
            'long_only': True,
            'apply_trading_lag': False
        }
        self.strategy = VAMSMomentumStrategy(self.strategy_config)

        # Create sample data: Monthly prices for 2 years
        dates = pd.date_range('2020-01-01', periods=24, freq='ME')
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

        np.random.seed(42)
        data = {}
        for ticker in tickers:
            # Create data with some trend and volatility
            base_trend = np.linspace(0, 0.1 * (1 if ticker in ['AAPL', 'GOOGL'] else -0.05) , len(dates))
            returns = np.random.normal(0.01, 0.05, len(dates)) + base_trend
            prices = 100 * (1 + returns).cumprod()
            data[ticker] = prices

        self.all_historical_data = pd.DataFrame(data, index=dates) # Renamed self.data

        benchmark_returns = np.random.normal(0.008, 0.04, len(dates))
        self.benchmark_historical_data = pd.DataFrame(
            {'SPY': 100 * (1 + benchmark_returns).cumprod()},
            index=dates
        ) # Made it a DataFrame

    def test_internal_calculate_dp_vams(self): # Renamed method
        """Test the internal DPVAMS calculation."""
        # Call the internal method from the strategy instance
        dp_vams_scores = self.strategy._calculate_dp_vams(self.all_historical_data)

        self.assertEqual(dp_vams_scores.shape, self.all_historical_data.shape)

        # DPVAMS calculation involves pct_change() and rolling operations.
        # The first `self.lookback_periods - 1` rows of pct_change().rolling() will be NaN.
        # The implementation fills these NaNs with 0.
        # So, we expect 0s for initial periods where full lookback is not available.
        # The number of initial periods that are zero depends on exact rolling mechanics.
        # `pct_change` makes 1st row NaN. `rolling` needs `lookback_periods` values.
        # So, `lookback_periods` initial values in dp_vams_scores should be 0.
        self.assertTrue((dp_vams_scores.iloc[:self.lookback_periods-1] == 0).all().all())

        # Values after initial period should be finite (given the fillna(0) in the method)
        self.assertTrue(np.isfinite(dp_vams_scores.iloc[self.lookback_periods-1:]).all().all())

        # Optional: Add a check for sensitivity to alpha if easily testable,
        # e.g., by creating another strategy instance with different alpha.
        # For now, focusing on basic functionality.

    def test_calculate_candidate_weights(self):
        """Test _calculate_candidate_weights helper (from BaseStrategy)."""
        self.strategy.num_holdings = 2
        self.strategy.strategy_config['num_holdings'] = 2
        self.strategy.strategy_config['top_decile_fraction'] = None # Ensure num_holdings is used

        look = pd.Series([0.5, 1.2, 0.8, 1.5, 0.3],
                        index=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
        weights = self.strategy._calculate_candidate_weights(look)

        self.assertAlmostEqual(weights.sum(), 1.0, places=6)
        self.assertTrue((weights >= 0).all())
        self.assertEqual((weights > 1e-6).sum(), 2) # Expect 2 holdings
        self.assertEqual(weights['AMZN'], 0.5)
        self.assertEqual(weights['MSFT'], 0.5)

    def test_apply_leverage_and_smoothing(self):
        """Test _apply_leverage_and_smoothing helper (from BaseStrategy)."""
        self.strategy.strategy_config['smoothing_lambda'] = 0.5
        self.strategy.strategy_config['leverage'] = 1.0

        cand = pd.Series([0.5, 0.5, 0.0, 0.0, 0.0],
                        index=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
        w_prev = pd.Series([0.3, 0.3, 0.4, 0.0, 0.0],
                          index=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
        w_new = self.strategy._apply_leverage_and_smoothing(cand, w_prev)

        self.assertAlmostEqual(w_new['AAPL'], 0.4)
        self.assertAlmostEqual(w_new['MSFT'], 0.4)
        self.assertAlmostEqual(w_new['GOOGL'], 0.2)
        self.assertTrue(np.isfinite(w_new).all())

    def test_generate_signals_single_date(self): # Renamed from test_generate_signals
        """Test signal generation for a single date using DPVAMS."""
        self.strategy.weights_history = pd.DataFrame() # Reset history

        # Pick a date where DPVAMS should be computable
        current_date = self.all_historical_data.index[self.lookback_periods + 1]

        signals_for_date = self.strategy.generate_signals(
            self.all_historical_data,
            self.benchmark_historical_data,
            current_date
        )

        self.assertEqual(signals_for_date.shape, (1, len(self.all_historical_data.columns)))
        self.assertTrue(np.isfinite(signals_for_date.values).all())
        if self.strategy.long_only:
            self.assertTrue((signals_for_date.values >= 0).all())
        self.assertAlmostEqual(signals_for_date.sum().sum(), 1.0, places=5)

        # Test for an early date (insufficient history for DPVAMS)
        early_date = self.all_historical_data.index[self.lookback_periods - 2]
        self.strategy.weights_history = pd.DataFrame() # Reset
        signals_early_date = self.strategy.generate_signals(
            self.all_historical_data, self.benchmark_historical_data, early_date
        )
        self.assertTrue((signals_early_date.values == 0).all())

    # test_generate_signals_with_sma_filter removed

    def test_edge_cases_generate_signals(self): # Renamed from test_edge_cases
        """Test edge cases for generate_signals with DPVAMS."""
        self.strategy.weights_history = pd.DataFrame()

        # Test with dataset just large enough for one DPVAMS calculation
        small_data_len = self.lookback_periods
        small_data = self.all_historical_data.iloc[:small_data_len]
        small_benchmark = self.benchmark_historical_data.iloc[:small_data_len]
        current_date_small = small_data.index[-1]

        signals = self.strategy.generate_signals(small_data, small_benchmark, current_date_small)
        self.assertEqual(signals.shape, (1, len(small_data.columns)))
        # DPVAMS internal fillna(0) should lead to finite signals or zeros.
        self.assertTrue(np.isfinite(signals.values).all())
        # If any DPVAMS scores are non-zero and different, it might pick assets.
        # If all are zero (e.g. due to lookback), then sum of weights = 0.
        # Or if num_holdings > 0 and some are picked, sum = 1.
        # This depends on the exact values from _calculate_dp_vams with minimal data.
        # For now, finite check is key.

        # Test with all zero returns (flat prices)
        self.strategy.weights_history = pd.DataFrame()
        zero_data = pd.DataFrame(100.0, index=self.all_historical_data.index,
                                 columns=self.all_historical_data.columns)
        current_date_zero = zero_data.index[self.lookback_periods + 2]

        signals_zero = self.strategy.generate_signals(zero_data, self.benchmark_historical_data, current_date_zero)
        self.assertEqual(signals_zero.shape, (1, len(zero_data.columns)))
        self.assertTrue(np.isfinite(signals_zero.values).all())
        # For flat prices, DPVAMS (momentum part is 0) should be 0 for all assets.
        # _calculate_candidate_weights with all zero scores might pick num_holdings assets (tie-breaking).
        if self.strategy.num_holdings is not None and self.strategy.num_holdings > 0:
             self.assertAlmostEqual(signals_zero.sum().sum(), 1.0, places=5)
        else:
             self.assertAlmostEqual(signals_zero.sum().sum(), 0.0, places=5)


if __name__ == '__main__':
    unittest.main()