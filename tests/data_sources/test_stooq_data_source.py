import os
import shutil
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.portfolio_backtester.data_sources.stooq_data_source import StooqDataSource

pytestmark = pytest.mark.network
pytest.skip("network test", allow_module_level=True)


class TestStooqDataSource(unittest.TestCase):
    """Unit tests for the StooqDataSource class."""

    def setUp(self):
        # Use a temporary directory so tests do not interfere with real cache.
        self.test_data_dir = Path("./test_data_stooq")
        self.test_data_dir.mkdir(exist_ok=True)

        # Short cache expiry so that cache tests are deterministic.
        self.data_source = StooqDataSource(cache_expiry_hours=0.01)
        # Redirect cache to temp folder.
        self.data_source.data_dir = self.test_data_dir

    def tearDown(self):
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)

    @patch("pandas_datareader.data.DataReader")
    def test_get_data_download(self, mock_datareader):
        """Verify that data is downloaded when not present in cache."""
        mock_df = pd.DataFrame(
            {
                "Close": [100, 101, 102],
                "Adj Close": [99, 100, 101],
            },
            index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
        )
        mock_datareader.return_value = mock_df

        tickers = ["^GSPC"]
        start_date = "2023-01-01"
        end_date = "2023-01-03"

        result_df = self.data_source.get_data(tickers, start_date, end_date)

        mock_datareader.assert_called_once_with("^SPX", "stooq", start=start_date, end=end_date)
        self.assertFalse(result_df.empty)
        self.assertIn("^GSPC", result_df.columns)
        self.assertEqual(result_df["^GSPC"].iloc[0], 100)  # Series renamed to original ticker

    @patch("pandas_datareader.data.DataReader")
    def test_get_data_cache(self, mock_datareader):
        """Verify that cached data is used when fresh."""
        ticker = "^GSPC"
        file_path = self.test_data_dir / f"{ticker}.csv"
        cached_df = pd.DataFrame(
            {"Close": [200, 201, 202]},
            index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
        )
        cached_df.index.name = "Date"
        cached_df.to_csv(file_path)

        # Update mtime to now to ensure cache freshness.
        os.utime(file_path, (time.time(), time.time()))

        tickers = [ticker]
        start_date = "2023-01-01"
        end_date = "2023-01-03"

        result_df = self.data_source.get_data(tickers, start_date, end_date)

        mock_datareader.assert_not_called()  # Should not hit network
        self.assertFalse(result_df.empty)
        self.assertIn(ticker, result_df.columns)
        self.assertEqual(result_df[ticker].iloc[0], 200)


if __name__ == "__main__":
    unittest.main() 