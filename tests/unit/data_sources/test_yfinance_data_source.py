import unittest
import pandas as pd
import os
import shutil
from unittest.mock import patch
from portfolio_backtester.data_sources.yfinance_data_source import YFinanceDataSource
from pathlib import Path
import time
import pytest

pytestmark = pytest.mark.network
pytest.skip("network test", allow_module_level=True)


class TestYFinanceDataSource(unittest.TestCase):

    def setUp(self):
        self.test_data_dir = Path("./test_data")
        self.test_data_dir.mkdir(exist_ok=True)
        self.data_source = YFinanceDataSource(cache_expiry_hours=0.01)  # Short expiry for testing
        self.data_source.data_dir = self.test_data_dir  # Redirect to test directory

    def tearDown(self):
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)

    @patch("yfinance.download")
    def test_get_data_download(self, mock_yfinance_download):
        # Mock yfinance.download to return a predictable DataFrame
        mock_df = pd.DataFrame(
            {"Close": [100, 101, 102], "Adj Close": [99, 100, 101]},
            index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
        )
        mock_yfinance_download.return_value = mock_df

        tickers = ["TEST"]
        start_date = "2023-01-01"
        end_date = "2023-01-03"

        result_df = self.data_source.get_data(tickers, start_date, end_date)

        mock_yfinance_download.assert_called_once_with(
            "TEST", start=start_date, end=end_date, auto_adjust=True, progress=False
        )
        self.assertFalse(result_df.empty)
        self.assertIn("TEST", result_df.columns)
        self.assertEqual(result_df["TEST"].iloc[0], 100)  # Should pick 'Close'

    @patch("yfinance.download")
    def test_get_data_cache(self, mock_yfinance_download):
        # Create a dummy cached file
        ticker = "CACHED"
        file_path = self.test_data_dir / f"{ticker}.csv"
        cached_df = pd.DataFrame(
            {"Close": [200, 201, 202]},
            index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
        )
        cached_df.index.name = "Date"
        cached_df.to_csv(file_path)

        # Ensure the file is recent enough to be cached
        os.utime(file_path, (time.time(), time.time()))

        tickers = ["CACHED"]
        start_date = "2023-01-01"
        end_date = "2023-01-03"

        result_df = self.data_source.get_data(tickers, start_date, end_date)

        mock_yfinance_download.assert_not_called()  # Should not download if cached
        self.assertFalse(result_df.empty)
        self.assertIn("CACHED", result_df.columns)
        self.assertEqual(result_df["CACHED"].iloc[0], 200)


if __name__ == "__main__":
    unittest.main()
