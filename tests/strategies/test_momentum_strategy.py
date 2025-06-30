
import unittest
import pandas as pd
import numpy as np
from src.portfolio_backtester.strategies.momentum_strategy import MomentumStrategy

class TestMomentumStrategy(unittest.TestCase):

    def setUp(self):
        # Create a sample price dataframe
        dates = pd.to_datetime(pd.date_range(start='2020-01-01', periods=12, freq='ME'))
        self.data = pd.DataFrame({
            'StockA': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210],
            'StockB': [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 1],
            'StockC': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'StockD': [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        }, index=dates)
        self.benchmark_data = pd.Series([100] * 12, index=dates)

    def test_generate_signals_smoke(self):
        # Smoke test to ensure the function runs without errors
        strategy_config = {
            'lookback_months': 3,
            'top_decile_fraction': 0.5,
            'smoothing_lambda': 0.5,
            'leverage': 1.0,
            'long_only': True
        }
        strategy = MomentumStrategy(strategy_config)
        try:
            strategy.generate_signals(self.data, self.benchmark_data)
        except Exception as e:
            self.fail(f"generate_signals raised an exception: {e}")

    def test_top_performer_selection(self):
        strategy_config = {
            'lookback_months': 3,
            'num_holdings': 1, # Explicitly select only the top stock
            'smoothing_lambda': 0.0, # No smoothing
            'leverage': 1.0,
            'long_only': True
        }
        strategy = MomentumStrategy(strategy_config)
        weights = strategy.generate_signals(self.data, self.benchmark_data)

        # At the end of the period, StockA should be the top performer
        final_weights = weights.iloc[-1]
        self.assertEqual(final_weights['StockA'], 1.0)
        self.assertEqual(final_weights['StockB'], 0.0)
        self.assertEqual(final_weights['StockC'], 0.0)

if __name__ == '__main__':
    unittest.main()
