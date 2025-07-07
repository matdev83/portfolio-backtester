import pytest
import pandas as pd
import numpy as np
from src.portfolio_backtester.features.momentum import Momentum

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
        lookback = 3
        skip = 0
        feature = Momentum(lookback_months=lookback, skip_months=skip)
        mom_values = feature.compute(sample_monthly_prices)

        # First `lookback` rows will be NaN due to data.shift(lookback) when skip is 0
        assert mom_values.iloc[:lookback].isna().all().all()

        # Test first calculable point: index `lookback` (which is the (lookback+1)-th month)
        # P(t) / P(t - lookback) - 1
        # For date sample_monthly_prices.index[lookback] ('2020-04-30')
        # Uses P('2020-04-30') / P('2020-01-31') - 1
        date_to_check = sample_monthly_prices.index[lookback] # This is the L-th index (0-based), so (L+1)th date
        price_at_date_to_check = sample_monthly_prices.loc[date_to_check, 'AssetA']

        # Denominator price is P(t - lookback_months)
        # For index `lookback`, the denominator index is `lookback - lookback = 0`
        price_at_denom_date = sample_monthly_prices.iloc[lookback - lookback].loc['AssetA']

        expected_val = (price_at_date_to_check / price_at_denom_date) - 1
        assert abs(mom_values.loc[date_to_check, 'AssetA'] - expected_val) < 1e-6


    def test_momentum_with_skip(self, sample_monthly_prices):
        lookback = 3
        skip = 1
        feature = Momentum(lookback_months=lookback, skip_months=skip)
        mom_values = feature.compute(sample_monthly_prices)

        # First (lookback + skip) rows will be NaN
        # (data.shift(skip) / data.shift(skip + lookback)) - 1
        # NaNs produced by data.shift(skip + lookback)
        assert mom_values.iloc[:(lookback + skip)].isna().all().all()

        # Test first calculable point: index `lookback + skip`
        # P(t-skip) / P(t - skip - lookback) - 1
        date_to_check_idx = lookback + skip # e.g., index 4 for L=3, S=1 ('2020-05-31')
        date_to_check = sample_monthly_prices.index[date_to_check_idx]

        numerator_price_date_idx = date_to_check_idx - skip # index 3 ('2020-04-30')
        numerator_price = sample_monthly_prices.iloc[numerator_price_date_idx].loc['AssetA']

        denominator_price_date_idx = date_to_check_idx - skip - lookback # index 0 ('2020-01-31')
        denominator_price = sample_monthly_prices.iloc[denominator_price_date_idx].loc['AssetA']

        expected_val = (numerator_price / denominator_price) - 1
        assert abs(mom_values.loc[date_to_check, 'AssetA'] - expected_val) < 1e-6


    def test_momentum_name_generation(self):
        f1 = Momentum(12, 0)
        assert f1.name == "momentum_12m" # Corrected: skip=0 and no suffix uses old format
        f2 = Momentum(6, 1, name_suffix="std")
        assert f2.name == "momentum_6m_skip1m_std"
        f3 = Momentum(3, name_suffix="pred_mom") # skip default 0
        assert f3.name == "momentum_3m_skip0m_pred_mom" # Corrected: skip=0 but has suffix
        f4 = Momentum(lookback_months=5) # skip=0, suffix=""
        assert f4.name == "momentum_5m" # Corrected: skip=0 and no suffix
