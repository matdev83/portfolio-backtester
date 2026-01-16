"""
Data sanity check tests for the MDMP data source.

These tests ensure we have up-to-date market data available via the
market-data-multi-provider package.
"""

import unittest
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import pytest
import pandas_market_calendars as mcal

from portfolio_backtester.interfaces.data_source_interface import create_data_source


class TestDataSanityCheck(unittest.TestCase):
    """
    Sanity check tests to ensure we have up-to-date market data available.
    These tests verify that data sources can fetch recent data for major tickers like AAPL.
    """

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Initialize NYSE calendar for proper trading day calculation
        self.nyse = mcal.get_calendar("NYSE")

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        pass

    def _get_expected_last_trading_day(
        self, reference_date: datetime | None = None
    ) -> pd.Timestamp:
        """
        Calculate the expected last trading day using NYSE market calendar.

        Args:
            reference_date: Date to calculate from (default: today)

        Returns:
            Expected last trading day as pandas Timestamp (timezone-naive)
        """
        if reference_date is None:
            reference_date = datetime.now()

        # Convert to pandas Timestamp for easier manipulation
        current_date = pd.Timestamp(reference_date).normalize()

        # Include future dates to handle timezone differences and current data
        start_date = current_date - timedelta(days=30)
        end_date = current_date + timedelta(days=5)

        # Get valid trading days from NYSE calendar
        valid_trading_days = self.nyse.valid_days(start_date=start_date, end_date=end_date)

        if len(valid_trading_days) == 0:
            start_date = current_date - timedelta(days=60)
            valid_trading_days = self.nyse.valid_days(start_date=start_date, end_date=end_date)

        # Convert timezone-aware NYSE calendar dates to timezone-naive
        valid_days_naive = [
            pd.Timestamp(day).tz_convert(None).normalize() for day in valid_trading_days
        ]

        # Filter to only include days up to current date
        current_or_past_days = [day for day in valid_days_naive if day <= current_date]

        if current_or_past_days:
            return current_or_past_days[-1]
        elif valid_days_naive:
            return valid_days_naive[-1]
        else:
            return current_date - timedelta(days=5)

    @pytest.mark.network
    def test_mdmp_data_source_aapl_recent_data(self) -> None:
        """
        Test that MDMP data source can fetch recent AAPL data.
        This is the main sanity check for data availability.
        """
        print("\n" + "=" * 60)
        print("SANITY CHECK: Testing MDMP data source for recent AAPL data")
        print("=" * 60)

        # Create MDMP data source
        mdmp_ds = create_data_source({"data_source": "mdmp"})

        # Calculate expected last trading day using NYSE calendar
        expected_last_trading_day = self._get_expected_last_trading_day()
        print(
            f"Expected last trading day (NYSE): {expected_last_trading_day.strftime('%Y-%m-%d')}"
        )

        # Fetch data for the last 30 days to ensure we get recent data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        print(f"Fetching AAPL data from {start_date} to {end_date}")

        # Fetch the data
        data = mdmp_ds.get_data(["AAPL"], start_date, end_date)

        # Basic checks
        self.assertFalse(data.empty, "No data was fetched for AAPL")
        self.assertIn(
            "AAPL", data.columns.get_level_values("Ticker"), "AAPL not found in fetched data"
        )

        # Get AAPL close prices
        aapl_close = data[("AAPL", "Close")].dropna()
        self.assertGreater(len(aapl_close), 0, "No valid AAPL close prices found")

        # Check the last available date
        last_data_date = aapl_close.index.max()
        print(f"Last available data date: {last_data_date.strftime('%Y-%m-%d')}")

        # Date validation
        current_date = pd.Timestamp(datetime.now()).normalize()
        range_start = current_date - timedelta(days=30)
        range_end = current_date + timedelta(days=5)

        recent_trading_days = self.nyse.valid_days(start_date=range_start, end_date=range_end)

        # Timezone normalization
        last_data_date_naive = pd.Timestamp(last_data_date).tz_localize(None).normalize()
        recent_trading_days_normalized = [
            pd.Timestamp(day).tz_convert(None).normalize() for day in recent_trading_days
        ]

        # Main validation: data should be from a recent trading day
        self.assertIn(
            last_data_date_naive,
            recent_trading_days_normalized,
            f"Last data date {last_data_date.strftime('%Y-%m-%d')} is not a recent NYSE trading day. "
            f"Expected range: {range_start.strftime('%Y-%m-%d')} to {range_end.strftime('%Y-%m-%d')}. "
            f"Recent trading days: {[d.strftime('%Y-%m-%d') for d in recent_trading_days_normalized[-10:]]}",
        )

        # Secondary validation: ensure data is not too stale
        days_behind_expected = len(
            [
                d
                for d in recent_trading_days_normalized
                if expected_last_trading_day >= d > last_data_date_naive
            ]
        )
        self.assertLessEqual(
            days_behind_expected,
            10,
            f"Data is too old: {days_behind_expected} trading days behind expected last trading day "
            f"({expected_last_trading_day.strftime('%Y-%m-%d')})",
        )

        # Check data quality
        last_close_price = aapl_close.iloc[-1]
        self.assertGreater(last_close_price, 0, "Last AAPL close price should be positive")
        self.assertLess(last_close_price, 1000, "Last AAPL close price seems unreasonably high")
        self.assertGreater(last_close_price, 50, "Last AAPL close price seems unreasonably low")

        print(f"Last AAPL close price: ${last_close_price:.2f}")
        print("✅ AAPL sanity check PASSED")

    @pytest.mark.network
    def test_mdmp_multiple_tickers(self) -> None:
        """
        Test that MDMP data source can fetch data for multiple tickers.
        """
        print("\n" + "=" * 60)
        print("SANITY CHECK: Testing MDMP data source for multiple tickers")
        print("=" * 60)

        # Create MDMP data source
        mdmp_ds = create_data_source({"data_source": "mdmp"})

        # Test period
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        tickers = ["SPY", "AAPL", "MSFT"]
        print(f"Fetching data for {tickers} from {start_date} to {end_date}")

        # Fetch the data
        data = mdmp_ds.get_data(tickers, start_date, end_date)

        # Basic checks
        self.assertFalse(data.empty, "No data was fetched")

        # Check that we got data for at least some tickers
        fetched_tickers = data.columns.get_level_values("Ticker").unique().tolist()
        print(f"Fetched tickers: {fetched_tickers}")
        self.assertGreater(len(fetched_tickers), 0, "No tickers found in fetched data")

        print("✅ Multiple tickers sanity check PASSED")


if __name__ == "__main__":
    unittest.main()
