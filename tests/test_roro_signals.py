import unittest
import pandas as pd

from src.portfolio_backtester.roro_signals import DummyRoRoSignal, BaseRoRoSignal

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
