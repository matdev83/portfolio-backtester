import pytest
import pandas as pd
import numpy as np
from src.portfolio_backtester.feature import Momentum

class TestMomentumFeature:
    @pytest.fixture
    def sample_monthly_prices(self) -> pd.DataFrame:
        dates = pd.date_range(start='2020-01-31', periods=24, freq='M') # Use month-end dates
        data = {
            'AssetA': np.linspace(100, 100 + 23 * 2, 24),
            'AssetB': np.concatenate([np.linspace(100, 120, 12), np.linspace(119, 100, 12)]),
            'AssetC': np.concatenate([np.full(12, 100.0), np.linspace(100.0, 100.0 + 11 * 2, 12)]) # Start flat part at 100
        }
        # Ensure float type for prices
        df = pd.DataFrame(data, index=dates).astype(float)
        return df

    def test_momentum_no_skip(self, sample_monthly_prices):
        feature = Momentum(lookback_months=3, skip_months=0)
        mom_values = feature.compute(sample_monthly_prices)

        # Expected for AssetA at 2020-03-31 (index 2): P(Mar)/P(Jan)-1, since ret_Jan is 0 due to fillna(0)
        # P_Jan = 100, P_Mar = 104 for AssetA
        expected_asset_a_mar = (sample_monthly_prices.loc['2020-03-31', 'AssetA'] /
                                sample_monthly_prices.loc['2020-01-31', 'AssetA']) - 1
        assert abs(mom_values.loc['2020-03-31', 'AssetA'] - expected_asset_a_mar) < 1e-6

        # Expected for AssetA at 2020-04-30 (index 3): P(Apr)/P(Feb)-1
        # P_Feb = 102, P_Apr = 106 for AssetA
        expected_asset_a_apr = (sample_monthly_prices.loc['2020-04-30', 'AssetA'] /
                                sample_monthly_prices.loc['2020-02-29', 'AssetA']) - 1
        assert abs(mom_values.loc['2020-04-30', 'AssetA'] - expected_asset_a_apr) < 1e-6


    def test_momentum_with_skip(self, sample_monthly_prices):
        feature = Momentum(lookback_months=3, skip_months=1)
        mom_values = feature.compute(sample_monthly_prices)

        # For L=3, S=1, value at date 't' is based on P(t-1)/P(t-1-3) = P(t-1)/P(t-4)
        # First valid point for mom_values will be 2020-04-30 (index 3)
        # This uses momentum_lookback_period.loc['2020-03-31']
        # momentum_lookback_period.loc['2020-03-31'] is P(Mar)/P(Jan)-1 for AssetA = 104/100-1 = 0.04
        expected_val_apr = (sample_monthly_prices.loc['2020-03-31', 'AssetA'] /
                            sample_monthly_prices.loc['2020-01-31', 'AssetA']) - 1
        assert abs(mom_values.loc['2020-04-30', 'AssetA'] - expected_val_apr) < 1e-6

        # Check AssetA at 2020-05-31 (index 4)
        # This uses momentum_lookback_period.loc['2020-04-30']
        # momentum_lookback_period.loc['2020-04-30'] is P(Apr)/P(Feb)-1 for AssetA = 106/102-1
        expected_val_may = (sample_monthly_prices.loc['2020-04-30', 'AssetA'] /
                            sample_monthly_prices.loc['2020-02-29', 'AssetA']) - 1
        assert abs(mom_values.loc['2020-05-31', 'AssetA'] - expected_val_may) < 1e-6


    def test_momentum_name_generation(self):
        f1 = Momentum(12, 0)
        assert f1.name == "momentum_12m_skip0m"
        f2 = Momentum(6, 1, name_suffix="std")
        assert f2.name == "momentum_6m_skip1m_std"
        f3 = Momentum(3, name_suffix="pred_mom") # skip default 0
        assert f3.name == "momentum_3m_skip0m_pred_mom"
        f4 = Momentum(lookback_months=5) # skip=0, suffix=""
        assert f4.name == "momentum_5m_skip0m"
