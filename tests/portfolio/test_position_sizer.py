
import unittest
import pandas as pd
import numpy as np
from src.portfolio_backtester.portfolio.position_sizer import equal_weight_sizer

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
