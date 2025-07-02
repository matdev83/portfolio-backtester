import unittest
import pandas as pd
import numpy as np

from src.portfolio_backtester.portfolio.volatility_targeting import (
    NoVolatilityTargeting,
    AnnualizedVolatilityTargeting,
)


class TestVolatilityTargeting(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        np.random.seed(0)
        self.returns = pd.Series(np.random.normal(0, 0.01, len(dates)), index=dates)

    def test_no_volatility_targeting(self):
        vt = NoVolatilityTargeting()
        pd.testing.assert_series_equal(vt.adjust_returns(self.returns), self.returns)

    def test_annualized_targeting_scales_returns(self):
        vt = AnnualizedVolatilityTargeting(target_vol=0.1, window=5, ann_factor=252)
        adjusted = vt.adjust_returns(self.returns)
        # After enough observations, volatility should be near target
        realized_vol = adjusted.std() * np.sqrt(252)
        self.assertAlmostEqual(realized_vol, 0.1, delta=0.07)


if __name__ == "__main__":
    unittest.main()
