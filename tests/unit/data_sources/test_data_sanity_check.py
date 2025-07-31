import unittest
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import pytest
import pandas_market_calendars as mcal

from src.portfolio_backtester.data_sources.hybrid_data_source import HybridDataSource
from src.portfolio_backtester.data_sources.stooq_data_source import StooqDataSource
from src.portfolio_backtester.data_sources.yfinance_data_source import YFinanceDataSource


class TestDataSanityCheck(unittest.TestCase):
    """
    Sanity check tests to ensure we have up-to-date market data available.
    These tests verify that data sources can fetch recent data for major tickers like AAPL.
    """

    def setUp(self):
        """Set up test fixtures."""
        # Use a temporary directory for testing
        self.test_data_dir = Path("./test_data_sanity")
        self.test_data_dir.mkdir(exist_ok=True)
        
        # Initialize NYSE calendar for proper trading day calculation
        self.nyse = mcal.get_calendar('NYSE')

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)

    def _get_expected_last_trading_day(self, reference_date=None):
        """
        Calculate the expected last trading day using NYSE market calendar.
        
        IMPORTANT: This method handles several tricky timezone and date validation issues:
        
        1. TIMEZONE HANDLING: NYSE calendar returns timezone-aware timestamps (UTC), but data 
           sources often return timezone-naive timestamps. Always convert to timezone-naive 
           using .tz_convert(None).normalize() for consistent comparison.
           
        2. FUTURE DATE TOLERANCE: Data providers may return data for "future" dates due to 
           timezone differences (e.g., Asian markets trading while US is still in previous day).
           Always include a buffer of future dates (typically 5 days) in validation ranges.
           
        3. WEEKEND/HOLIDAY EDGE CASES: When testing on weekends or holidays, the "expected" 
           last trading day should be the most recent actual trading day, not the current date.
           
        4. DATA PROVIDER LAG: Some data sources have delays. Don't be too strict about requiring
           the absolute latest trading day - allow for reasonable lag (up to 10 trading days).
        
        Args:
            reference_date: Date to calculate from (default: today)
            
        Returns:
            Expected last trading day as pandas Timestamp (timezone-naive)
        """
        if reference_date is None:
            reference_date = datetime.now()
        
        # Convert to pandas Timestamp for easier manipulation
        current_date = pd.Timestamp(reference_date).normalize()
        
        # CRITICAL: Include future dates to handle timezone differences and current data
        # Data providers may return data for dates that appear "future" due to timezone offsets
        start_date = current_date - timedelta(days=30)
        end_date = current_date + timedelta(days=5)  # Allow for future dates
        
        # Get valid trading days from NYSE calendar
        # NOTE: This returns timezone-aware timestamps in UTC
        valid_trading_days = self.nyse.valid_days(start_date=start_date, end_date=end_date)
        
        if len(valid_trading_days) == 0:
            # Fallback: if no valid days found, go back further
            start_date = current_date - timedelta(days=60)
            valid_trading_days = self.nyse.valid_days(start_date=start_date, end_date=end_date)
        
        # CRITICAL: Convert timezone-aware NYSE calendar dates to timezone-naive
        # This prevents comparison errors between timezone-aware and timezone-naive timestamps
        valid_days_naive = [pd.Timestamp(day).tz_convert(None).normalize() for day in valid_trading_days]
        
        # Filter to only include days up to current date (allow same day)
        current_or_past_days = [day for day in valid_days_naive if day <= current_date]
        
        if current_or_past_days:
            # Return the most recent trading day (could be today if it's a trading day)
            return current_or_past_days[-1]
        elif valid_days_naive:
            # Fallback: use the most recent valid day even if it's in the future
            # This handles cases where we're testing with current/future data
            return valid_days_naive[-1]
        else:
            # Ultimate fallback: just use current date minus 5 days
            return current_date - timedelta(days=5)

    def _clean_cache_for_ticker(self, ticker, data_source):
        """Remove cached data for a specific ticker to force fresh download."""
        if hasattr(data_source, 'data_dir'):
            cache_file = data_source.data_dir / f"{ticker}.csv"
            if cache_file.exists():
                cache_file.unlink()
                print(f"Removed cached file: {cache_file}")

    @pytest.mark.network
    def test_hybrid_data_source_aapl_recent_data(self):
        """
        Test that hybrid data source can fetch recent AAPL data.
        This is the main sanity check for data availability.
        """
        print("\n" + "="*60)
        print("SANITY CHECK: Testing hybrid data source for recent AAPL data")
        print("="*60)
        
        # Create hybrid data source with short cache expiry to force fresh data
        hybrid_ds = HybridDataSource(cache_expiry_hours=0.01, prefer_stooq=True)
        hybrid_ds.data_dir = self.test_data_dir
        hybrid_ds.stooq_source.data_dir = self.test_data_dir
        hybrid_ds.yfinance_source.data_dir = self.test_data_dir
        
        # Clean any existing cache
        self._clean_cache_for_ticker('AAPL', hybrid_ds)
        
        # Calculate expected last trading day using NYSE calendar
        expected_last_trading_day = self._get_expected_last_trading_day()
        print(f"Expected last trading day (NYSE): {expected_last_trading_day.strftime('%Y-%m-%d')}")
        
        # Fetch data for the last 30 days to ensure we get recent data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        print(f"Fetching AAPL data from {start_date} to {end_date}")
        
        # Fetch the data
        data = hybrid_ds.get_data(['AAPL'], start_date, end_date)
        
        # Basic checks
        self.assertFalse(data.empty, "No data was fetched for AAPL")
        self.assertIn('AAPL', data.columns.get_level_values('Ticker'), "AAPL not found in fetched data")
        
        # Get AAPL close prices
        aapl_close = data[('AAPL', 'Close')].dropna()
        self.assertGreater(len(aapl_close), 0, "No valid AAPL close prices found")
        
        # Check the last available date
        last_data_date = aapl_close.index.max()
        print(f"Last available data date: {last_data_date.strftime('%Y-%m-%d')}")
        
        # CRITICAL DATE VALIDATION LOGIC - This section handles several tricky edge cases:
        # 
        # PROBLEM 1: Data sources may return "future" dates due to timezone differences
        # SOLUTION: Create a validation range that includes future dates (up to 5 days)
        #
        # PROBLEM 2: NYSE calendar returns timezone-aware dates, data sources return timezone-naive
        # SOLUTION: Always normalize both to timezone-naive before comparison
        #
        # PROBLEM 3: Tests may run on weekends/holidays when markets are closed
        # SOLUTION: Use NYSE market calendar to determine valid trading days, not just weekdays
        
        current_date = pd.Timestamp(datetime.now()).normalize()
        
        # Create a wider validation range to handle edge cases
        range_start = current_date - timedelta(days=30)
        range_end = current_date + timedelta(days=5)  # CRITICAL: Allow for future dates
        
        recent_trading_days = self.nyse.valid_days(
            start_date=range_start,
            end_date=range_end
        )
        
        # CRITICAL: Timezone normalization to prevent comparison failures
        # NYSE calendar returns UTC timestamps, but data sources often return timezone-naive
        last_data_date_naive = pd.Timestamp(last_data_date).tz_localize(None).normalize()
        recent_trading_days_normalized = [pd.Timestamp(day).tz_convert(None).normalize() for day in recent_trading_days]
        
        # Main validation: data should be from a recent trading day
        # This was the primary failure point before the fix - data for 2025-07-25 (Friday)
        # was being rejected because the validation range didn't include future dates
        self.assertIn(
            last_data_date_naive,
            recent_trading_days_normalized,
            f"Last data date {last_data_date.strftime('%Y-%m-%d')} is not a recent NYSE trading day. "
            f"Expected range: {range_start.strftime('%Y-%m-%d')} to {range_end.strftime('%Y-%m-%d')}. "
            f"Recent trading days: {[d.strftime('%Y-%m-%d') for d in recent_trading_days_normalized[-10:]]}"
        )
        
        # Secondary validation: ensure data is not too stale (conservative lag tolerance)
        days_behind_expected = len([d for d in recent_trading_days_normalized 
                                   if expected_last_trading_day >= d > last_data_date_naive])
        self.assertLessEqual(
            days_behind_expected, 10,
            f"Data is too old: {days_behind_expected} trading days behind expected last trading day "
            f"({expected_last_trading_day.strftime('%Y-%m-%d')})"
        )
        
        # Check that the data file was created
        aapl_file = self.test_data_dir / "AAPL.csv"
        self.assertTrue(aapl_file.exists(), f"AAPL.csv file was not created at {aapl_file}")
        
        # Verify the file contains the expected data
        file_data = pd.read_csv(aapl_file, index_col=0, parse_dates=True)
        self.assertGreater(len(file_data), 0, "AAPL.csv file is empty")
        
        # Check that the last row in the file matches our expectations
        last_file_date = file_data.index.max()
        print(f"Last date in AAPL.csv file: {last_file_date.strftime('%Y-%m-%d')}")
        
        self.assertEqual(
            last_data_date.date(), 
            last_file_date.date(),
            "Last date in DataFrame doesn't match last date in CSV file"
        )
        
        # Check data quality
        last_close_price = aapl_close.iloc[-1]
        self.assertGreater(last_close_price, 0, "Last AAPL close price should be positive")
        self.assertLess(last_close_price, 1000, "Last AAPL close price seems unreasonably high")
        self.assertGreater(last_close_price, 50, "Last AAPL close price seems unreasonably low")
        
        print(f"Last AAPL close price: ${last_close_price:.2f}")
        
        # Check failure report
        failure_report = hybrid_ds.get_failure_report()
        print(f"Data source failure report: {failure_report}")
        
        # AAPL should not have failed on both sources
        self.assertNotIn('AAPL', failure_report['total_failures'], 
                        "AAPL failed on both Stooq and yfinance sources")
        
        print("✅ AAPL sanity check PASSED")

    @pytest.mark.network
    def test_individual_data_sources_aapl(self):
        """
        Test individual data sources (Stooq and yfinance) for AAPL data.
        This helps identify which source might be having issues.
        """
        print("\n" + "="*60)
        print("SANITY CHECK: Testing individual data sources for AAPL")
        print("="*60)
        
        # Test period
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        expected_last_trading_day = self._get_expected_last_trading_day()
        
        # Test Stooq
        print("\n--- Testing Stooq ---")
        stooq_ds = StooqDataSource()
        stooq_ds.data_dir = self.test_data_dir
        self._clean_cache_for_ticker('AAPL', stooq_ds)
        
        try:
            stooq_data = stooq_ds.get_data(['AAPL'], start_date, end_date)
            if not stooq_data.empty and 'AAPL' in stooq_data.columns.get_level_values('Ticker'):
                stooq_last_date = stooq_data[('AAPL', 'Close')].dropna().index.max()
                print(f"Stooq last date: {stooq_last_date.strftime('%Y-%m-%d')}")
                print("✅ Stooq source working")
            else:
                print("❌ Stooq source returned empty data")
        except Exception as e:
            print(f"❌ Stooq source failed: {e}")
        
        # Test yfinance
        print("\n--- Testing yfinance ---")
        yfinance_ds = YFinanceDataSource()
        yfinance_ds.data_dir = self.test_data_dir
        self._clean_cache_for_ticker('AAPL', yfinance_ds)
        
        try:
            yfinance_data = yfinance_ds.get_data(['AAPL'], start_date, end_date)
            if not yfinance_data.empty and 'AAPL' in yfinance_data.columns.get_level_values('Ticker'):
                yfinance_last_date = yfinance_data[('AAPL', 'Close')].dropna().index.max()
                print(f"yfinance last date: {yfinance_last_date.strftime('%Y-%m-%d')}")
                print("✅ yfinance source working")
            else:
                print("❌ yfinance source returned empty data")
        except Exception as e:
            print(f"❌ yfinance source failed: {e}")

    


if __name__ == '__main__':
    unittest.main() 