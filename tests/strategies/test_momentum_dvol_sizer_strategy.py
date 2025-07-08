import unittest
import pandas as pd
import numpy as np

from src.portfolio_backtester.strategies.momentum_dvol_sizer_strategy import MomentumDvolSizerStrategy


class TestMomentumDvolSizerStrategy(unittest.TestCase):
    def setUp(self):
        dates = pd.to_datetime(pd.date_range(start='2020-01-01', periods=12, freq='ME'))
        self.data = pd.DataFrame({
            'A': np.linspace(100, 200, 12),
            'B': np.linspace(100, 50, 12),
        }, index=dates)
        self.benchmark = pd.Series(np.linspace(100, 120, 12), index=dates, name='SPY')

    def test_resolve_and_defaults(self):
        strategy = MomentumDvolSizerStrategy({'lookback_months': 3})
        self.assertEqual(strategy.strategy_config['position_sizer'], 'rolling_downside_volatility')

    def test_generate_signals_smoke(self):
        strategy_config = {'lookback_months': 3}
        strategy = MomentumDvolSizerStrategy(strategy_config)
        # Create proper benchmark DataFrame with 'Close' column
        benchmark_df = self.benchmark.to_frame()
        benchmark_df.columns = ['Close']  # Rename to match expected column name
        current_date = self.data.index[-1]
        signals = strategy.generate_signals(self.data, benchmark_df, current_date)
        # Check that signals have the correct shape (1 row for current_date, columns for each asset)
        expected_shape = (1, len(self.data.columns))
        self.assertEqual(signals.shape, expected_shape)


if __name__ == '__main__':
    unittest.main()

