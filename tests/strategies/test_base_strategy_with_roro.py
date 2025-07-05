import unittest
import pandas as pd
import numpy as np

from src.portfolio_backtester.strategies.base_strategy import BaseStrategy
from src.portfolio_backtester.signal_generators import MomentumSignalGenerator
from src.portfolio_backtester.roro_signals import DummyRoRoSignal
from src.portfolio_backtester.feature import Momentum

# --- Mock Components ---
class TestSignalGenerator(MomentumSignalGenerator):
    """A simple signal generator for testing."""
    def required_features(self):
        return {Momentum(lookback_months=1)}

    def scores(self, features: dict) -> pd.DataFrame:
        # Return a dummy score DataFrame, e.g., based on the 'momentum_1m' feature
        # For simplicity, let's assume 'momentum_1m' exists and has positive scores
        if "momentum_1m" not in features or features["momentum_1m"].empty:
             # Provide a default DataFrame with NaNs if features are missing or empty
            num_dates = len(features.get('dates_for_scores', pd.date_range(start="2020-01-01", periods=3, freq="ME")))
            num_assets = 2
            return pd.DataFrame(np.nan,
                                index=features.get('dates_for_scores', pd.date_range(start="2020-01-01", periods=3, freq="ME")),
                                columns=[f'Asset{i+1}' for i in range(num_assets)])

        # If feature exists, generate scores. Let's make them constant positive for simplicity.
        scores_df = pd.DataFrame(1.0, index=features["momentum_1m"].index, columns=features["momentum_1m"].columns)
        return scores_df


class StrategyWithDummyRoRo(BaseStrategy):
    """A test strategy that uses the DummyRoRoSignal."""
    signal_generator_class = TestSignalGenerator
    roro_signal_class = DummyRoRoSignal

    def __init__(self, strategy_config: dict):
        super().__init__(strategy_config)
        # Ensure roro_signal_class is set on the instance if BaseStrategy relies on it being there
        # This might be redundant if BaseStrategy's __init__ or get_roro_signal handles it correctly
        self.roro_signal_class = DummyRoRoSignal


class TestBaseStrategyWithRoRo(unittest.TestCase):

    def setUp(self):
        self.strategy_config = {
            "strategy_params": {"lookback_months": 1}, # For TestSignalGenerator
            "num_holdings": 1, # Simplified strategy logic
            "long_only": True,
            "smoothing_lambda": 0.0, # No smoothing for simpler weight checking
            "leverage": 1.0
            # No "roro_signal_params" needed for DummyRoRoSignal
        }
        self.test_strategy = StrategyWithDummyRoRo(self.strategy_config)

        # Create sample data
        self.dates = pd.to_datetime([
            "2005-12-31", "2006-01-31", "2006-02-28", # Before, Start of RoRo, During RoRo
            "2009-12-31", "2010-01-31",             # End of RoRo, After RoRo
            "2019-12-31", "2020-01-31", "2020-02-29", # Before, Start of RoRo, During RoRo
            "2020-04-30", "2020-05-31",             # End of RoRo (Window ends Apr 1, so Apr 30 is out), After RoRo
            "2021-12-31", "2022-01-31", "2022-10-31", # Before, Start of RoRo, During RoRo
            "2022-11-30", "2022-12-31"              # End of RoRo (Window ends Nov 5, so Nov 30 is out), After RoRo
        ])
        # Ensure dates are month-end for typical strategy signal generation
        self.dates = pd.DatetimeIndex([d + pd.offsets.MonthEnd(0) for d in self.dates]).unique()


        self.assets = ["Asset1", "Asset2"]
        self.prices = pd.DataFrame(
            np.random.rand(len(self.dates), len(self.assets)) + 10,
            index=self.dates,
            columns=self.assets
        )
        # Mock features: A simple momentum feature that TestSignalGenerator expects
        # The actual values of momentum don't matter as much as its presence and structure
        # TestSignalGenerator will return constant positive scores anyway.
        mock_momentum_data = pd.DataFrame(
            0.05, # Constant positive momentum
            index=self.dates,
            columns=self.assets
        )
        self.features = {"momentum_1m": mock_momentum_data, 'dates_for_scores': self.dates}

        # Mock benchmark data (not strictly needed if sma_filter_window is not set, but good practice)
        self.benchmark_data = pd.Series(np.random.rand(len(self.dates)) + 10, index=self.dates)


    def test_weights_with_dummy_roro_signal(self):
        """
        Tests that the strategy weights are zero when DummyRoRoSignal is off (0)
        and non-zero when DummyRoRoSignal is on (1).
        """
        generated_weights = self.test_strategy.generate_signals(
            prices=self.prices,
            features=self.features,
            benchmark_data=self.benchmark_data
        )

        # Expected RoRo "on" periods for the *monthly* dates in self.dates
        # DummyRoRoSignal windows:
        # 2006-01-01 to 2009-12-31
        # 2020-01-01 to 2020-04-01
        # 2022-01-01 to 2022-11-05

        expected_risk_on_dates = pd.to_datetime([
            # First window
            "2006-01-31", "2006-02-28", "2009-12-31",
            # Second window
            "2020-01-31", "2020-02-29", # Note: 2020-04-01 is the end, so 2020-03-31 would be in, 2020-04-30 is out
            # Third window
            "2022-01-31", "2022-10-31" # Note: 2022-11-05 is the end, so 2022-10-31 is in, 2022-11-30 is out
        ])
        # Ensure these are also month-end
        expected_risk_on_dates = pd.DatetimeIndex([d + pd.offsets.MonthEnd(0) for d in expected_risk_on_dates]).unique()

        self.assertEqual(generated_weights.shape, self.prices.shape)

        for date in self.dates:
            weights_at_date = generated_weights.loc[date]
            if date in expected_risk_on_dates:
                # Weights should be non-zero (or positive for long_only=True)
                # Based on num_holdings=1 and TestSignalGenerator returning positive scores for all,
                # one asset should have weight 1.0 (or 1.0 / num_assets if TestSignalGenerator was different)
                # and others 0.
                self.assertTrue(weights_at_date.sum() > 0, f"Weights should be positive on {date} (RoRo ON)")
                self.assertAlmostEqual(weights_at_date.sum(), self.strategy_config.get("leverage",1.0), places=5, msg=f"Total weight should be close to leverage on {date}")
            else:
                # Weights should be zero
                self.assertTrue(np.allclose(weights_at_date, 0), f"Weights should be zero on {date} (RoRo OFF)")
                self.assertAlmostEqual(weights_at_date.sum(), 0.0, places=5, msg=f"Total weight should be zero on {date} (RoRo OFF)")

if __name__ == '__main__':
    unittest.main()
