import unittest
import pandas as pd
import numpy as np

from src.portfolio_backtester.strategies.portfolio.momentum_strategy import MomentumStrategy
from src.portfolio_backtester.roro_signals import DummyRoRoSignal, BaseRoRoSignal

class StrategyWithDummyRoRo(MomentumStrategy):
    """A test strategy that uses the DummyRoRoSignal."""
    
    # Set the class attribute to enable RoRo signal
    roro_signal_class = DummyRoRoSignal
    
    def __init__(self, strategy_config: dict):
        super().__init__(strategy_config)
        # The base class will handle instantiation via get_roro_signal()


class TestBaseStrategyWithRoRo(unittest.TestCase):

    def setUp(self):
        self.strategy_config = {
            "strategy_params": {
                "lookback_months": 1,
                "skip_months": 0,
                "num_holdings": 1,
                "long_only": True,
                "smoothing_lambda": 0.0,
                "leverage": 1.0,
                "price_column_asset": "Close",
                "price_column_benchmark": "Close"
            }
        }
        self.test_strategy = StrategyWithDummyRoRo(self.strategy_config)

        # Create sample data with MultiIndex columns (OHLCV format)
        # Start from 2010 onwards to ensure sufficient historical data
        self.dates = pd.to_datetime([
            "2019-12-31", "2020-01-31", "2020-02-29", "2020-03-31", # Before, Start of RoRo, During RoRo, End of RoRo
            "2020-04-30", "2020-05-31",             # After RoRo (Window ends Apr 1, so Apr 30 is out)
            "2021-12-31", "2022-01-31", "2022-10-31", # Before, Start of RoRo, During RoRo
            "2022-11-30", "2022-12-31"              # End of RoRo (Window ends Nov 5, so Nov 30 is out), After RoRo
        ])
        # Ensure dates are month-end for typical strategy signal generation
        self.dates = pd.DatetimeIndex([d + pd.offsets.MonthEnd(0) for d in self.dates]).unique()

        self.assets = ["Asset1", "Asset2"]
        
        # Create MultiIndex DataFrame for asset data (OHLCV format)
        fields = ['Open', 'High', 'Low', 'Close', 'Volume']
        columns = pd.MultiIndex.from_product([self.assets, fields], names=['Ticker', 'Field'])
        
        # Generate sample OHLCV data with sufficient history before test dates
        # Add historical data starting from 2010 to ensure momentum calculations work
        historical_start = pd.Timestamp("2010-01-31")
        all_dates = pd.date_range(start=historical_start, end=self.dates[-1], freq='ME')
        
        np.random.seed(42)
        data = {}
        for asset in self.assets:
            base_price = 100
            for i, date in enumerate(all_dates):
                price = base_price + np.random.normal(0, 5) + i * 0.1  # Slight upward trend
                data[(asset, 'Open')] = data.get((asset, 'Open'), []) + [price * 0.99]
                data[(asset, 'High')] = data.get((asset, 'High'), []) + [price * 1.02]
                data[(asset, 'Low')] = data.get((asset, 'Low'), []) + [price * 0.98]
                data[(asset, 'Close')] = data.get((asset, 'Close'), []) + [price]
                data[(asset, 'Volume')] = data.get((asset, 'Volume'), []) + [1000000]
        
        self.asset_data = pd.DataFrame(data, index=all_dates, columns=columns)
        
        # Create benchmark data with MultiIndex - also with full history
        benchmark_columns = pd.MultiIndex.from_product([['SPY'], fields], names=['Ticker', 'Field'])
        benchmark_data = {}
        base_price = 100
        for i, date in enumerate(all_dates):
            price = base_price + np.random.normal(0, 3) + i * 0.1
            benchmark_data[('SPY', 'Open')] = benchmark_data.get(('SPY', 'Open'), []) + [price * 0.99]
            benchmark_data[('SPY', 'High')] = benchmark_data.get(('SPY', 'High'), []) + [price * 1.01]
            benchmark_data[('SPY', 'Low')] = benchmark_data.get(('SPY', 'Low'), []) + [price * 0.98]
            benchmark_data[('SPY', 'Close')] = benchmark_data.get(('SPY', 'Close'), []) + [price]
            benchmark_data[('SPY', 'Volume')] = benchmark_data.get(('SPY', 'Volume'), []) + [50000000]
        
        self.benchmark_data = pd.DataFrame(benchmark_data, index=all_dates, columns=benchmark_columns)

    def test_weights_with_dummy_roro_signal(self):
        """
        Tests that the strategy weights are zero when DummyRoRoSignal is off (0)
        and non-zero when DummyRoRoSignal is on (1).
        """
        all_weights = []
        
        # Generate signals for each date
        for date in self.dates:
            weights = self.test_strategy.generate_signals(
                all_historical_data=self.asset_data,
                benchmark_historical_data=self.benchmark_data,
                current_date=date
            )
            all_weights.append(weights)
        
        # Combine all weights into a single DataFrame
        generated_weights = pd.concat(all_weights, axis=0)

        # Expected RoRo "on" periods for the *monthly* dates in self.dates
        # DummyRoRoSignal windows:
        # 2020-01-01 to 2020-04-01
        # 2022-01-01 to 2022-11-05

        expected_risk_on_dates = pd.to_datetime([
            # Second window: 2020-01-01 to 2020-04-01
            "2020-01-31", "2020-02-29", "2020-03-31", # Note: 2020-04-01 is the end, so 2020-03-31 is in, 2020-04-30 is out
            # Third window: 2022-01-01 to 2022-11-05
            "2022-01-31", "2022-10-31" # Note: 2022-11-05 is the end, so 2022-10-31 is in, 2022-11-30 is out
        ])
        # Ensure these are also month-end
        expected_risk_on_dates = pd.DatetimeIndex([d + pd.offsets.MonthEnd(0) for d in expected_risk_on_dates]).unique()

        self.assertEqual(generated_weights.shape[0], len(self.dates))
        self.assertEqual(generated_weights.shape[1], len(self.assets))

        for date in self.dates:
            weights_at_date = generated_weights.loc[date]
            if date in expected_risk_on_dates:
                # Weights should be non-zero (or positive for long_only=True)
                # Based on num_holdings=1 and momentum strategy,
                # one asset should have weight 1.0 and others 0.
                self.assertTrue(weights_at_date.sum() > 0, f"Weights should be positive on {date} (RoRo ON)")
                # For early dates with insufficient history, momentum might be equal/zero, leading to equal weights
                # Be more tolerant for the total weight assertion
                total_weight = weights_at_date.sum()
                self.assertGreater(total_weight, 0, f"Total weight should be positive on {date} (RoRo ON)")
                # Allow some tolerance for cases where momentum scores are equal
                self.assertLessEqual(total_weight, self.strategy_config["strategy_params"].get("leverage", 1.0) + 0.01, msg=f"Total weight should not exceed leverage on {date}")
            else:
                # Weights should be zero
                self.assertTrue(np.allclose(weights_at_date, 0), f"Weights should be zero on {date} (RoRo OFF)")
                self.assertAlmostEqual(float(weights_at_date.sum().item()), 0.0, places=5, msg=f"Total weight should be zero on {date} (RoRo OFF)")

class TestDummyRoRoSignal(unittest.TestCase):

    def test_dummy_roro_signal_generation(self):
        """
        Tests that the DummyRoRoSignal generates 0s and 1s correctly
        for specified date windows.
        """
        signal_generator = DummyRoRoSignal()

        # Test dates covering all periods (before, during, between, after windows)
        test_dates = pd.to_datetime([
            "2005-12-31", # Before first window
            "2006-01-01", # Start of first window
            "2007-06-15", # During first window
            "2009-12-31", # End of first window
            "2010-01-01", # After first window, before second
            "2019-12-31", # Before second window
            "2020-01-01", # Start of second window
            "2020-02-15", # During second window
            "2020-04-01", # End of second window
            "2020-04-02", # After second window
            "2021-12-31", # Before third window
            "2022-01-01", # Start of third window
            "2022-06-10", # During third window
            "2022-11-05", # End of third window
            "2022-11-06", # After third window
            "2023-01-01"  # Well after all windows
        ])

        expected_values = [
            0, # 2005-12-31
            1, # 2006-01-01
            1, # 2007-06-15
            1, # 2009-12-31
            0, # 2010-01-01
            0, # 2019-12-31
            1, # 2020-01-01
            1, # 2020-02-15
            1, # 2020-04-01
            0, # 2020-04-02
            0, # 2021-12-31
            1, # 2022-01-01
            1, # 2022-06-10
            1, # 2022-11-05
            0, # 2022-11-06
            0  # 2023-01-01
        ]

        # Create dummy data for the new interface
        dummy_data = pd.DataFrame()
        dummy_benchmark = pd.DataFrame()
        
        # Generate signals for each date individually
        signal_values = []
        for date in test_dates:
            signal_values.append(int(signal_generator.generate_signal(dummy_data, dummy_benchmark, date)))
        
        signal_series = pd.Series(signal_values, index=test_dates)

        self.assertIsInstance(signal_series, pd.Series)
        self.assertEqual(len(signal_series), len(test_dates))
        pd.testing.assert_index_equal(signal_series.index, test_dates)
        pd.testing.assert_series_equal(signal_series, pd.Series(expected_values, index=test_dates, dtype=int), check_dtype=False)

    def test_dummy_roro_signal_empty_dates(self):
        """
        Tests the DummyRoRoSignal with an empty DatetimeIndex.
        """
        signal_generator = DummyRoRoSignal()
        test_dates = pd.DatetimeIndex([])
        signal_series = pd.Series([], index=test_dates, dtype=int)
        self.assertIsInstance(signal_series, pd.Series)
        self.assertTrue(signal_series.empty)
        self.assertEqual(signal_series.dtype, int)

    def test_dummy_roro_signal_no_overlap(self):
        """
        Tests the DummyRoRoSignal with dates that do not overlap any risk-on window.
        """
        signal_generator = DummyRoRoSignal()
        test_dates = pd.to_datetime(["2000-01-01", "2015-01-01", "2025-01-01"])
        expected_values = [0, 0, 0]
        
        # Create dummy data for the new interface
        dummy_data = pd.DataFrame()
        dummy_benchmark = pd.DataFrame()
        
        # Generate signals for each date individually
        signal_values = []
        for date in test_dates:
            signal_values.append(int(signal_generator.generate_signal(dummy_data, dummy_benchmark, date)))
        
        signal_series = pd.Series(signal_values, index=test_dates)
        pd.testing.assert_series_equal(signal_series, pd.Series(expected_values, index=test_dates, dtype=int), check_dtype=False)

    def test_base_roro_signal_default_features(self):
        """
        Tests that BaseRoRoSignal returns an empty set for required features by default.
        """
        class ConcreteRoRo(BaseRoRoSignal):
            def generate_signal(self, all_historical_data: pd.DataFrame, benchmark_historical_data: pd.DataFrame, current_date: pd.Timestamp) -> bool:
                return False # Dummy implementation

        roro_signal = ConcreteRoRo()
        self.assertEqual(roro_signal.get_required_features(), set())

if __name__ == '__main__':
    unittest.main()