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
        
        Args:
            reference_date: Date to calculate from (default: today)
            
        Returns:
            Expected last trading day as pandas Timestamp (timezone-naive)
        """
        if reference_date is None:
            reference_date = datetime.now()
        
        # Convert to pandas Timestamp for easier manipulation
        current_date = pd.Timestamp(reference_date).normalize()
        
        # Get the last 30 days of valid trading days to find the most recent one
        # This accounts for weekends, holidays, and potential data provider delays
        end_date = current_date
        start_date = current_date - timedelta(days=30)
        
        # Get valid trading days from NYSE calendar
        valid_trading_days = self.nyse.valid_days(start_date=start_date, end_date=end_date)
        
        if len(valid_trading_days) == 0:
            # Fallback: if no valid days found, go back further
            start_date = current_date - timedelta(days=60)
            valid_trading_days = self.nyse.valid_days(start_date=start_date, end_date=end_date)
        
        # Return the most recent valid trading day
        # Account for potential data provider lag by allowing up to 3 trading days back
        if len(valid_trading_days) >= 3:
            # Use the 3rd most recent trading day to account for data lag
            # Convert to timezone-naive for consistency
            return valid_trading_days[-3].tz_convert(None).normalize()
        elif len(valid_trading_days) > 0:
            # Use the most recent if we have fewer than 3 days
            return valid_trading_days[-1].tz_convert(None).normalize()
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
        
        # The last data date should be within a reasonable range of the expected trading day
        # Allow for up to 5 trading days lag (very conservative for data provider delays)
        # Get the last 10 trading days to create acceptable range
        recent_trading_days = self.nyse.valid_days(
            start_date=expected_last_trading_day - timedelta(days=20),
            end_date=expected_last_trading_day
        )
        
        # Check if last_data_date is within the recent trading days
        # Convert both to timezone-naive for comparison
        last_data_date_naive = pd.Timestamp(last_data_date).tz_localize(None).normalize()
        recent_trading_days_normalized = [pd.Timestamp(day).tz_convert(None).normalize() for day in recent_trading_days]
        
        self.assertIn(
            last_data_date_naive,
            recent_trading_days_normalized,
            f"Last data date {last_data_date.strftime('%Y-%m-%d')} is not a recent NYSE trading day. "
            f"Recent trading days: {[d.strftime('%Y-%m-%d') for d in recent_trading_days_normalized[-5:]]}"
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

    def test_market_calendar_logic(self):
        """
        Test the market calendar logic to ensure it's working correctly.
        This test runs offline and validates the NYSE calendar functionality.
        """
        print("\n" + "="*60)
        print("SANITY CHECK: Testing NYSE market calendar logic")
        print("="*60)
        
        # Test known trading days and holidays
        known_trading_day = pd.Timestamp('2023-01-03', tz='UTC')  # Tuesday, should be trading day
        known_holiday = pd.Timestamp('2023-01-01', tz='UTC')      # New Year's Day, should be holiday
        known_weekend = pd.Timestamp('2023-01-01', tz='UTC')      # Sunday, should be non-trading
        
        # Get valid trading days for January 2023
        jan_2023_trading_days = self.nyse.valid_days(
            start_date='2023-01-01', 
            end_date='2023-01-31'
        )
        
        # NYSE calendar returns timezone-aware timestamps
        # Convert our test timestamps to match
        jan_2023_trading_days_normalized = [pd.Timestamp(day).normalize() for day in jan_2023_trading_days]
        
        # Test assertions
        self.assertIn(known_trading_day.normalize(), jan_2023_trading_days_normalized, 
                     "January 3, 2023 should be a trading day")
        self.assertNotIn(known_holiday.normalize(), jan_2023_trading_days_normalized, 
                        "January 1, 2023 (New Year's Day) should not be a trading day")
        
        # Test current date functionality
        current_expected = self._get_expected_last_trading_day()
        self.assertIsInstance(current_expected, pd.Timestamp, 
                            "Expected last trading day should be a pandas Timestamp")
        
        # Test that we get reasonable number of trading days in a month (around 20-23)
        self.assertGreater(len(jan_2023_trading_days), 15, 
                          "Should have more than 15 trading days in January 2023")
        self.assertLess(len(jan_2023_trading_days), 25, 
                       "Should have fewer than 25 trading days in January 2023")
        
        print(f"January 2023 had {len(jan_2023_trading_days)} trading days")
        print(f"First trading day: {jan_2023_trading_days[0].strftime('%Y-%m-%d')}")
        print(f"Last trading day: {jan_2023_trading_days[-1].strftime('%Y-%m-%d')}")
        print("✅ NYSE market calendar logic working correctly")

    def test_data_lag_tolerance(self):
        """
        Test that our data lag tolerance logic works correctly.
        This test runs offline and validates the acceptable data lag calculation.
        """
        print("\n" + "="*60)
        print("SANITY CHECK: Testing data lag tolerance logic")
        print("="*60)
        
        # Test with different reference dates
        test_dates = [
            datetime(2023, 1, 3),   # Tuesday
            datetime(2023, 1, 6),   # Friday
            datetime(2023, 1, 7),   # Saturday (weekend)
            datetime(2023, 1, 8),   # Sunday (weekend)
            datetime(2023, 1, 9),   # Monday
        ]
        
        for test_date in test_dates:
            expected_trading_day = self._get_expected_last_trading_day(test_date)
            
            # The expected trading day should be a valid trading day
            # Get trading days around the expected date
            start_check = expected_trading_day - timedelta(days=10)
            end_check = expected_trading_day + timedelta(days=10)
            
            valid_days = self.nyse.valid_days(start_date=start_check, end_date=end_check)
            valid_days_normalized = [pd.Timestamp(day).tz_convert(None).normalize() for day in valid_days]
            
            # expected_trading_day is already timezone-naive from helper function
            self.assertIn(expected_trading_day, valid_days_normalized,
                         f"Expected trading day {expected_trading_day.strftime('%Y-%m-%d')} "
                         f"for reference date {test_date.strftime('%Y-%m-%d')} should be a valid trading day")
            
            print(f"Reference: {test_date.strftime('%Y-%m-%d')} -> Expected: {expected_trading_day.strftime('%Y-%m-%d')}")
        
        print("✅ Data lag tolerance logic working correctly")


if __name__ == '__main__':
    unittest.main() 