import unittest
import pandas as pd

from src.portfolio_backtester.portfolio.volatility_targeting import (
    NoVolatilityTargeting,
    AnnualizedVolatilityTargeting,
)


class TestVolatilityTargeting(unittest.TestCase):
    def test_no_volatility_targeting_pass_through(self):
        rets = pd.Series([0.01, -0.02, 0.03])
        vt = NoVolatilityTargeting()
        pd.testing.assert_series_equal(vt.adjust_returns(rets), rets)

    def test_annualized_volatility_targeting_scales_returns(self):
        dates = pd.date_range("2020-01-01", periods=6, freq="D")
        rets = pd.Series(0.01, index=dates)
        vt = AnnualizedVolatilityTargeting(target_vol=0.2, window=3, max_leverage=2.0)
        scaled = vt.adjust_returns(rets)
        self.assertTrue((scaled.iloc[:3] == 0.01).all())
        self.assertTrue((scaled.iloc[3:] == 0.02).all())


if __name__ == "__main__":
    unittest.main()
