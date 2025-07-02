
import unittest
import pandas as pd
import numpy as np
from src.portfolio_backtester.portfolio.position_sizer import (
    equal_weight_sizer,
    rolling_sharpe_sizer,
    rolling_sortino_sizer,
    rolling_beta_sizer,
    rolling_corr_sizer,
)

class TestPositionSizer(unittest.TestCase):

    def test_equal_weight_sizer(self):
        signals = pd.DataFrame({
            'StockA': [1, 1, 0],
            'StockB': [0, 1, 1],
            'StockC': [1, 0, 1]
        })
        expected_weights = pd.DataFrame({
            'StockA': [0.5, 1/2, 0],
            'StockB': [0, 1/2, 0.5],
            'StockC': [0.5, 0, 0.5]
        })
        
        result_weights = equal_weight_sizer(signals)
        pd.testing.assert_frame_equal(result_weights, expected_weights)

    def test_rolling_sharpe_sizer(self):
        signals = pd.DataFrame({'A': [1, 1, 1], 'B': [1, 1, 1]})
        returns = pd.DataFrame({'A': [0.1, 0.0, 0.1], 'B': [0.2, 0.2, 0.2]})
        result = rolling_sharpe_sizer(signals, returns, window=2)
        self.assertAlmostEqual(result.iloc[-1]['A'], 1.0)
        self.assertAlmostEqual(result.iloc[-1]['B'], 0.0)

    def test_rolling_sortino_sizer(self):
        signals = pd.DataFrame({'A': [1, 1, 1], 'B': [1, 1, 1]})
        returns = pd.DataFrame({'A': [0.1, -0.1, 0.2], 'B': [-0.1, -0.1, -0.1]})
        result = rolling_sortino_sizer(signals, returns, window=2, target_return=0)
        self.assertGreater(result.iloc[-1]['A'], result.iloc[-1]['B'])

    def test_rolling_beta_sizer(self):
        signals = pd.DataFrame({'A': [1, 1, 1], 'B': [1, 1, 1]})
        returns = pd.DataFrame({'A': [0.1, 0.2, 0.3], 'B': [0.0, 0.0, 0.0]})
        benchmark = pd.Series([0.1, 0.2, 0.3])
        result = rolling_beta_sizer(signals, returns, benchmark, window=2)
        self.assertGreater(result.iloc[-1]['B'], result.iloc[-1]['A'])

    def test_rolling_corr_sizer(self):
        signals = pd.DataFrame({'A': [1, 1, 1], 'B': [1, 1, 1]})
        returns = pd.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]})
        benchmark = pd.Series([1, 2, 3])
        result = rolling_corr_sizer(signals, returns, benchmark, window=2)
        self.assertAlmostEqual(result.iloc[-1]['A'], 0.0)
        self.assertAlmostEqual(result.iloc[-1]['B'], 1.0)

    def test_equal_weight_sizer_empty_signals(self):
        signals = pd.DataFrame()
        expected_weights = pd.DataFrame()
        result_weights = equal_weight_sizer(signals)
        pd.testing.assert_frame_equal(result_weights, expected_weights)

    def test_equal_weight_sizer_all_zeros(self):
        signals = pd.DataFrame({
            'StockA': [0, 0],
            'StockB': [0, 0]
        })
        expected_weights = pd.DataFrame({
            'StockA': [np.nan, np.nan],
            'StockB': [np.nan, np.nan]
        })
        result_weights = equal_weight_sizer(signals)
        pd.testing.assert_frame_equal(result_weights, expected_weights)

if __name__ == '__main__':
    unittest.main()
