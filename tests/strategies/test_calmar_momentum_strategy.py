import unittest
import pandas as pd
import numpy as np
from src.portfolio_backtester.strategies.calmar_momentum_strategy import CalmarMomentumStrategy
from src.portfolio_backtester.features.calmar_ratio import CalmarRatio


class TestCalmarMomentumStrategy(unittest.TestCase):

    def setUp(self):
        """Set up test data and strategy configuration."""
        self.strategy_config = {
            'rolling_window': 6,
            'top_decile_fraction': 0.1,
            'smoothing_lambda': 0.5,
            'leverage': 1.0,
            'long_only': True,
            'sma_filter_window': None
        }
        self.strategy = CalmarMomentumStrategy(self.strategy_config)

        # Create sample data
        dates = pd.date_range('2020-01-01', periods=24, freq='ME')
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        # Generate sample price data with different trends
        np.random.seed(42)
        data = {}
        for ticker in tickers:
            # Create trending data with some volatility
            returns = np.random.normal(0.01, 0.05, len(dates))
            prices = 100 * (1 + returns).cumprod()
            data[ticker] = prices
        
        self.data = pd.DataFrame(data, index=dates)
        self.benchmark_data = pd.Series(100 * (1 + np.random.normal(0.008, 0.04, len(dates))).cumprod(), 
                                       index=dates, name='SPY')

    def test_calculate_rolling_calmar(self):
        """Test the rolling Calmar ratio calculation."""
        calmar_feature = CalmarRatio(rolling_window=6)
        rolling_calmar = calmar_feature.compute(self.data)
        
        # Check that the result has the correct shape
        self.assertEqual(rolling_calmar.shape, self.data.shape)
        
        # Check that initial periods have NaN values (due to rolling window and pct_change)
        rolling_window = self.strategy_config['rolling_window']
        self.assertTrue(rolling_calmar.iloc[:rolling_window-1].isna().all().all())

        # Check that values after the rolling window are finite
        self.assertTrue(np.isfinite(rolling_calmar.iloc[rolling_window:]).all().all())

    def test_calculate_candidate_weights(self):
        """Test candidate weight calculation."""
        # Create a sample look series
        look = pd.Series([0.5, 1.2, 0.8, 1.5, 0.3], 
                        index=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
        
        weights = self.strategy._calculate_candidate_weights(look)
        
        # Check that weights sum to approximately 1 for long-only strategy
        self.assertAlmostEqual(weights.sum(), 1.0, places=6)
        
        # Check that only top assets have positive weights
        self.assertTrue((weights >= 0).all())
        
        # Check that the number of non-zero weights matches expected holdings
        expected_holdings = max(int(np.ceil(0.1 * len(look))), 1)
        actual_holdings = (weights > 0).sum()
        self.assertEqual(actual_holdings, expected_holdings)

    def test_apply_leverage_and_smoothing(self):
        """Test leverage and smoothing application."""
        cand = pd.Series([0.5, 0.5, 0.0, 0.0, 0.0], 
                        index=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
        w_prev = pd.Series([0.3, 0.3, 0.4, 0.0, 0.0], 
                          index=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
        
        w_new = self.strategy._apply_leverage_and_smoothing(cand, w_prev)
        
        # Check that smoothing was applied (new weights should be between previous and candidate)
        self.assertTrue(((w_new - w_prev) * (cand - w_prev) >= 0).all())
        
        # Check that weights are finite
        self.assertTrue(np.isfinite(w_new).all())

    def test_generate_signals(self):
        """Test signal generation."""
        current_date = self.data.index[-1]
        # Create proper benchmark DataFrame with 'Close' column
        benchmark_df = self.benchmark_data.to_frame()
        benchmark_df.columns = ['Close']  # Rename to match expected column name
        signals = self.strategy.generate_signals(self.data, benchmark_df, current_date)
        
        # Check that signals have the correct shape (1 row for current_date, columns for each asset)
        expected_shape = (1, len(self.data.columns))
        self.assertEqual(signals.shape, expected_shape)
        
        # Check that all values are finite
        self.assertTrue(np.isfinite(signals).all().all())
        
        # Check that weights are non-negative for long-only strategy
        self.assertTrue((signals >= 0).all().all())
        
        # Check that the index contains the current_date
        self.assertIn(current_date, signals.index)

    def test_generate_signals_with_sma_filter(self):
        """Test signal generation with SMA filter."""
        # Create strategy with SMA filter
        config_with_sma = self.strategy_config.copy()
        config_with_sma['sma_filter_window'] = 3
        strategy_with_sma = CalmarMomentumStrategy(config_with_sma)
        current_date = self.data.index[-1]
        # Create proper benchmark DataFrame with 'Close' column
        benchmark_df = self.benchmark_data.to_frame()
        benchmark_df.columns = ['Close']  # Rename to match expected column name
        signals = strategy_with_sma.generate_signals(self.data, benchmark_df, current_date)
        
        # Check that signals have the correct shape (1 row for current_date, columns for each asset)
        expected_shape = (1, len(self.data.columns))
        self.assertEqual(signals.shape, expected_shape)
        
        # Check that all values are finite
        self.assertTrue(np.isfinite(signals).all().all())

    def test_edge_cases(self):
        """Test edge cases."""
        # Test with very small dataset
        small_data = self.data.iloc[:3]
        small_benchmark = self.benchmark_data.iloc[:3]
        current_date = small_data.index[-1]
        # Create proper benchmark DataFrame with 'Close' column
        small_benchmark_df = small_benchmark.to_frame()
        small_benchmark_df.columns = ['Close']  # Rename to match expected column name
        signals = self.strategy.generate_signals(small_data, small_benchmark_df, current_date)
        expected_shape = (1, len(small_data.columns))
        self.assertEqual(signals.shape, expected_shape)

        # Test with all zero returns
        zero_data = pd.DataFrame(100, index=self.data.index, columns=self.data.columns)
        current_date = zero_data.index[-1]
        # Create proper benchmark DataFrame with 'Close' column
        benchmark_df = self.benchmark_data.to_frame()
        benchmark_df.columns = ['Close']  # Rename to match expected column name
        signals = self.strategy.generate_signals(zero_data, benchmark_df, current_date)
        expected_shape = (1, len(zero_data.columns))
        self.assertEqual(signals.shape, expected_shape)


if __name__ == '__main__':
    unittest.main()