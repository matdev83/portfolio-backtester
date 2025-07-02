import unittest
import pandas as pd
import numpy as np

from src.portfolio_backtester.strategies.momentum_dvol_sizer_strategy import MomentumDvolSizerStrategy
from src.portfolio_backtester.feature_engineering import precompute_features


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
        required = strategy.get_required_features({'strategy_params': strategy_config})
        features = precompute_features(self.data, required, self.benchmark)
        signals = strategy.generate_signals(self.data, features, self.benchmark)
        self.assertEqual(signals.shape, self.data.shape)


if __name__ == '__main__':
    unittest.main()

