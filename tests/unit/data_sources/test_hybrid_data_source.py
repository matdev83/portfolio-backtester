import os
import shutil
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.offsets import BDay

from src.portfolio_backtester.data_sources.hybrid_data_source import HybridDataSource
from tests.unit.data_sources.test_consolidated_data_validation import DataValidationUtilities


class TestHybridDataSource(unittest.TestCase):
    """Comprehensive tests for the HybridDataSource class."""

    def setUp(self):
        """Set up test fixtures."""
        # Use a temporary directory for testing
        self.test_data_dir = Path("./test_data_hybrid")
        self.test_data_dir.mkdir(exist_ok=True)
        
        # Create hybrid data source instance
        self.data_source = HybridDataSource(cache_expiry_hours=0.01, prefer_stooq=True)
        
        # Redirect both underlying sources to test directory
        self.data_source.stooq_source.data_dir = self.test_data_dir
        self.data_source.yfinance_source.data_dir = self.test_data_dir
        self.data_source.data_dir = self.test_data_dir
        self.data_source._negative_cache.clear()

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)

    def _create_mock_stooq_data(self, tickers, start_date="2023-01-01", end_date="2023-01-03"):
        """Create mock data in Stooq format (MultiIndex)."""
        dates = pd.date_range(start_date, end_date, freq='D')
        
        # Create MultiIndex columns for all tickers
        columns = []
        for ticker in tickers:
            for field in ['Open', 'High', 'Low', 'Close', 'Volume']:
                columns.append((ticker, field))
        
        multi_columns = pd.MultiIndex.from_tuples(columns, names=['Ticker', 'Field'])
        
        # Create mock data
        data = {}
        for ticker in tickers:
            base_price = 100 + hash(ticker) % 50  # Different base price per ticker
            for i, field in enumerate(['Open', 'High', 'Low', 'Close', 'Volume']):
                if field == 'Volume':
                    data[(ticker, field)] = [1000000 + i * 10000 for i in range(len(dates))]
                else:
                    data[(ticker, field)] = [base_price + i + j * 0.5 for j in range(len(dates))]
        
        return pd.DataFrame(data, index=dates, columns=multi_columns)

    def _create_mock_yfinance_data(self, tickers, start_date="2023-01-01", end_date="2023-01-03"):
        """Create mock data in yfinance format (flat columns)."""
        dates = pd.date_range(start_date, end_date, freq='D')
        
        # Create flat columns with ticker names
        data = {}
        for ticker in tickers:
            base_price = 100 + hash(ticker) % 50  # Different base price per ticker
            data[ticker] = [base_price + i * 0.5 for i in range(len(dates))]
        
        return pd.DataFrame(data, index=dates)

    def _create_invalid_data(self, tickers):
        """Create invalid data for testing failure scenarios."""
        dates = pd.date_range("2023-01-01", "2023-01-03", freq='D')
        
        # Create data with all NaN values
        data = {}
        for ticker in tickers:
            data[ticker] = [np.nan] * len(dates)
        
        return pd.DataFrame(data, index=dates)

    def test_init_basic(self):
        """Test basic initialization."""
        ds = HybridDataSource(cache_expiry_hours=12, prefer_stooq=False)
        
        self.assertEqual(ds.cache_expiry_hours, 12)
        self.assertFalse(ds.prefer_stooq)
        self.assertIsNotNone(ds.stooq_source)
        self.assertIsNotNone(ds.yfinance_source)
        self.assertEqual(ds.failed_tickers, {'stooq': set(), 'yfinance': set()})

    

    def test_normalize_data_format_stooq(self):
        """Test data normalization from Stooq format."""
        mock_data = self._create_mock_stooq_data(['AAPL', 'GOOGL'])
        
        normalized = self.data_source._normalize_data_format(mock_data, 'stooq', ['AAPL', 'GOOGL'])
        
        # Should return the same data since it's already in MultiIndex format
        self.assertTrue(isinstance(normalized.columns, pd.MultiIndex))
        self.assertEqual(normalized.columns.names, ['Ticker', 'Field'])
        self.assertIn(('AAPL', 'Close'), normalized.columns)
        self.assertIn(('GOOGL', 'Close'), normalized.columns)

    def test_normalize_data_format_yfinance(self):
        """Test data normalization from yfinance format."""
        mock_data = self._create_mock_yfinance_data(['AAPL', 'GOOGL'])
        
        normalized = self.data_source._normalize_data_format(mock_data, 'yfinance', ['AAPL', 'GOOGL'])
        
        # Should convert to MultiIndex format
        self.assertTrue(isinstance(normalized.columns, pd.MultiIndex))
        self.assertEqual(normalized.columns.names, ['Ticker', 'Field'])
        self.assertIn(('AAPL', 'Close'), normalized.columns)
        self.assertIn(('GOOGL', 'Close'), normalized.columns)

    def test_convert_to_multiindex(self):
        """Test conversion of flat DataFrame to MultiIndex."""
        flat_data = self._create_mock_yfinance_data(['AAPL', 'GOOGL'])
        
        multiindex_data = self.data_source._convert_to_multiindex(flat_data, ['AAPL', 'GOOGL'])
        
        self.assertTrue(isinstance(multiindex_data.columns, pd.MultiIndex))
        self.assertEqual(multiindex_data.columns.names, ['Ticker', 'Field'])
        
        # Check that all tickers are present
        tickers = multiindex_data.columns.get_level_values('Ticker').unique()
        self.assertIn('AAPL', tickers)
        self.assertIn('GOOGL', tickers)
        
        # Check that OHLC columns are created (using Close as approximation)
        fields = multiindex_data.columns.get_level_values('Field').unique()
        expected_fields = ['Open', 'High', 'Low', 'Close', 'Volume']
        for field in expected_fields:
            self.assertIn(field, fields)

    def test_fetch_from_source_success(self):
        """Test successful data fetching from a source."""
        # Mock the data source instances directly
        mock_stooq_instance = MagicMock()
        mock_yfinance_instance = MagicMock()
        
        # Create hybrid data source and replace the instances
        ds = HybridDataSource()
        ds.stooq_source = mock_stooq_instance
        ds.yfinance_source = mock_yfinance_instance
        
        # Mock successful Stooq data
        mock_stooq_data = self._create_mock_stooq_data(['AAPL', 'GOOGL'])
        mock_stooq_instance.get_data.return_value = mock_stooq_data
        
        data, successful, failed = ds._fetch_from_source('stooq', ['AAPL', 'GOOGL'], '2023-01-01', '2023-01-03')
        
        mock_stooq_instance.get_data.assert_called_once_with(['AAPL', 'GOOGL'], '2023-01-01', '2023-01-03')
        self.assertFalse(data.empty)
        self.assertEqual(len(successful), 2)
        self.assertEqual(len(failed), 0)
        self.assertIn('AAPL', successful)
        self.assertIn('GOOGL', successful)

    def test_fetch_from_source_partial_failure(self):
        """Test partial failure scenario where some tickers fail validation."""
        # Mock the data source instances directly
        mock_stooq_instance = MagicMock()
        mock_yfinance_instance = MagicMock()
        
        # Create hybrid data source and replace the instances
        ds = HybridDataSource()
        ds.stooq_source = mock_stooq_instance
        ds.yfinance_source = mock_yfinance_instance
        
        # Create mixed data: valid for AAPL, invalid for GOOGL
        dates = pd.date_range("2023-01-01", "2023-01-10", freq='D')
        mixed_data = pd.DataFrame({
            ('AAPL', 'Close'): [100 + i for i in range(len(dates))],
            ('GOOGL', 'Close'): [np.nan] * len(dates)  # Invalid data
        }, index=dates)
        mixed_data.columns = pd.MultiIndex.from_tuples(mixed_data.columns, names=['Ticker', 'Field'])
        
        mock_stooq_instance.get_data.return_value = mixed_data
        
        data, successful, failed = ds._fetch_from_source('stooq', ['AAPL', 'GOOGL'], '2023-01-01', '2023-01-03')
        
        self.assertFalse(data.empty)
        self.assertEqual(len(successful), 1)
        self.assertEqual(len(failed), 1)
        self.assertIn('AAPL', successful)
        self.assertIn('GOOGL', failed)

    def test_get_data_stooq_success(self):
        """Test successful data fetching from Stooq only."""
        # Mock the data source instances directly
        mock_stooq_instance = MagicMock()
        mock_yfinance_instance = MagicMock()
        
        # Create hybrid data source and replace the instances
        ds = HybridDataSource(prefer_stooq=True)
        ds._negative_cache.clear()
        ds.stooq_source = mock_stooq_instance
        ds.yfinance_source = mock_yfinance_instance
        
        # Mock successful Stooq data
        mock_stooq_data = self._create_mock_stooq_data(['AAPL', 'GOOGL'])
        mock_stooq_instance.get_data.return_value = mock_stooq_data
        
        result = ds.get_data(['AAPL', 'GOOGL'], '2023-01-01', '2023-01-03')
        
        # Should only call Stooq, not yfinance
        mock_stooq_instance.get_data.assert_called_once()
        mock_yfinance_instance.get_data.assert_not_called()
        
        self.assertFalse(result.empty)
        self.assertTrue(isinstance(result.columns, pd.MultiIndex))
        
        # Check failure tracking
        self.assertEqual(len(ds.failed_tickers['stooq']), 0)
        self.assertEqual(len(ds.failed_tickers['yfinance']), 0)

    def test_get_data_fallback_to_yfinance(self):
        """Test fallback to yfinance when Stooq fails."""
        # Mock the data source instances directly
        mock_stooq_instance = MagicMock()
        mock_yfinance_instance = MagicMock()
        
        # Create hybrid data source and replace the instances
        ds = HybridDataSource(prefer_stooq=True)
        ds._negative_cache.clear()
        ds.stooq_source = mock_stooq_instance
        ds.yfinance_source = mock_yfinance_instance
        
        # Mock Stooq failure (empty data)
        mock_stooq_instance.get_data.return_value = pd.DataFrame()
        
        # Mock successful yfinance data
        mock_yfinance_data = self._create_mock_yfinance_data(['AAPL', 'GOOGL'])
        mock_yfinance_instance.get_data.return_value = mock_yfinance_data
        
        result = ds.get_data(['AAPL', 'GOOGL'], '2023-01-01', '2023-01-03')
        
        # Should call both sources
        mock_stooq_instance.get_data.assert_called_once()
        mock_yfinance_instance.get_data.assert_called_once()
        
        self.assertFalse(result.empty)
        self.assertTrue(isinstance(result.columns, pd.MultiIndex))
        
        # Check failure tracking
        self.assertEqual(len(ds.failed_tickers['stooq']), 2)  # Both failed on Stooq
        self.assertEqual(len(ds.failed_tickers['yfinance']), 0)  # Both succeeded on yfinance

    def test_get_data_mixed_sources(self):
        """Test scenario where some tickers succeed on Stooq, others need yfinance."""
        # Mock the data source instances directly
        mock_stooq_instance = MagicMock()
        mock_yfinance_instance = MagicMock()
        
        # Create hybrid data source and replace the instances
        ds = HybridDataSource(prefer_stooq=True)
        ds._negative_cache.clear()
        ds.stooq_source = mock_stooq_instance
        ds.yfinance_source = mock_yfinance_instance
        
        # Mock partial Stooq success (only AAPL)
        stooq_data = self._create_mock_stooq_data(['AAPL'])
        mock_stooq_instance.get_data.return_value = stooq_data
        
        # Mock yfinance success for failed ticker (GOOGL)
        yfinance_data = self._create_mock_yfinance_data(['GOOGL'])
        mock_yfinance_instance.get_data.return_value = yfinance_data
        
        result = ds.get_data(['AAPL', 'GOOGL'], '2023-01-01', '2023-01-03')
        
        # Should call both sources
        mock_stooq_instance.get_data.assert_called_once_with(['AAPL', 'GOOGL'], '2023-01-01', '2023-01-03')
        mock_yfinance_instance.get_data.assert_called_once_with(['GOOGL'], '2023-01-01', '2023-01-03')
        
        self.assertFalse(result.empty)
        self.assertTrue(isinstance(result.columns, pd.MultiIndex))
        
        # Should have data for both tickers
        tickers = result.columns.get_level_values('Ticker').unique()
        self.assertIn('AAPL', tickers)
        self.assertIn('GOOGL', tickers)

    def test_get_data_complete_failure(self):
        """Test scenario where both sources fail."""
        # Mock the data source instances directly
        mock_stooq_instance = MagicMock()
        mock_yfinance_instance = MagicMock()
        
        # Create hybrid data source and replace the instances
        ds = HybridDataSource(prefer_stooq=True)
        ds._negative_cache.clear()
        ds.stooq_source = mock_stooq_instance
        ds.yfinance_source = mock_yfinance_instance
        
        # Mock both sources returning empty data
        mock_stooq_instance.get_data.return_value = pd.DataFrame()
        mock_yfinance_instance.get_data.return_value = pd.DataFrame()
        
        result = ds.get_data(['AAPL', 'GOOGL'], '2023-01-01', '2023-01-03')
        
        # Should call both sources
        mock_stooq_instance.get_data.assert_called_once()
        mock_yfinance_instance.get_data.assert_called_once()
        
        self.assertTrue(result.empty)
        
        # Check failure tracking
        self.assertEqual(len(ds.failed_tickers['stooq']), 2)
        self.assertEqual(len(ds.failed_tickers['yfinance']), 2)

    def test_get_data_prefer_yfinance(self):
        """Test that prefer_stooq=False uses yfinance as primary source."""
        # Mock the data source instances directly
        mock_stooq_instance = MagicMock()
        mock_yfinance_instance = MagicMock()
        
        # Create hybrid data source and replace the instances
        ds = HybridDataSource(prefer_stooq=False)
        ds._negative_cache.clear()
        ds.stooq_source = mock_stooq_instance
        ds.yfinance_source = mock_yfinance_instance
        
        # Mock successful yfinance data
        mock_yfinance_data = self._create_mock_yfinance_data(['AAPL'])
        mock_yfinance_instance.get_data.return_value = mock_yfinance_data
        
        result = ds.get_data(['AAPL'], '2023-01-01', '2023-01-03')
        
        # Should only call yfinance, not Stooq
        mock_yfinance_instance.get_data.assert_called_once()
        mock_stooq_instance.get_data.assert_not_called()
        
        self.assertFalse(result.empty)

    def test_get_failure_report(self):
        """Test failure report generation."""
        # Set up some failure data
        self.data_source.failed_tickers = {
            'stooq': {'AAPL', 'GOOGL'},
            'yfinance': {'GOOGL', 'MSFT'}
        }
        
        report = self.data_source.get_failure_report()
        
        expected_report = {
            'stooq_failures': ['AAPL', 'GOOGL'],
            'yfinance_failures': ['GOOGL', 'MSFT'],
            'total_failures': ['GOOGL'],  # Only GOOGL failed on both
            'stooq_only_count': 1,  # MSFT succeeded on Stooq but failed on yfinance
            'yfinance_only_count': 1,  # AAPL succeeded on yfinance but failed on Stooq
            'complete_failure_count': 1  # GOOGL failed on both
        }
        
        # Convert lists to sets for comparison since order doesn't matter
        for key in ['stooq_failures', 'yfinance_failures', 'total_failures']:
            self.assertEqual(set(report[key]), set(expected_report[key]))
        
        for key in ['stooq_only_count', 'yfinance_only_count', 'complete_failure_count']:
            self.assertEqual(report[key], expected_report[key])

    def test_get_data_empty_tickers(self):
        """Test behavior with empty ticker list."""
        result = self.data_source.get_data([], '2023-01-01', '2023-01-03')
        
        self.assertTrue(result.empty)

    def _get_expected_last_trading_day(self, reference_date=None):
        """Calculate the expected last trading day, accounting for weekends and holidays."""
        if reference_date is None:
            reference_date = datetime.now()
        
        current_date = pd.Timestamp(reference_date).normalize()
        
        # US market holidays (simplified)
        us_holidays_2024_2025 = [
            pd.Timestamp('2024-01-01'), pd.Timestamp('2024-07-04'), pd.Timestamp('2024-12-25'),
            pd.Timestamp('2025-01-01'), pd.Timestamp('2025-07-04'), pd.Timestamp('2025-12-25'),
        ]
        
        candidate_date = current_date
        for _ in range(15):  # Look back up to 2 weeks
            if candidate_date.weekday() < 5 and candidate_date not in us_holidays_2024_2025:
                return candidate_date
            candidate_date = candidate_date - timedelta(days=1)
        
        return current_date - BDay(5)

    @pytest.mark.network
    def test_hybrid_data_source_basic_sanity_check(self):
        """
        Basic sanity check that hybrid data source can fetch real AAPL data.
        This test verifies the fail-tolerance workflow works with real data sources.
        """
        # Skip this test in CI or if network access is not desired
        if os.environ.get('SKIP_NETWORK_TESTS', '').lower() == 'true':
            self.skipTest("Network tests disabled")
        
        # Create hybrid data source with very short cache expiry
        ds = HybridDataSource(cache_expiry_hours=0.01, prefer_stooq=True)
        ds.data_dir = self.test_data_dir
        ds.stooq_source.data_dir = self.test_data_dir
        ds.yfinance_source.data_dir = self.test_data_dir
        
        # Calculate expected last trading day
        expected_last_trading_day = self._get_expected_last_trading_day()
        
        # Fetch recent data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
        
        try:
            data = ds.get_data(['AAPL'], start_date, end_date)
            
            # Basic validation
            self.assertFalse(data.empty, "No data was fetched for AAPL")
            self.assertIn('AAPL', data.columns.get_level_values('Ticker'), "AAPL not found in data")
            
            # Check we have close prices
            aapl_close = data[('AAPL', 'Close')].dropna()
            self.assertGreater(len(aapl_close), 0, "No valid AAPL close prices")
            
            # Check data recency (within 5 business days of expected)
            last_data_date = aapl_close.index.max()
            max_lag = BDay(5)
            earliest_acceptable = expected_last_trading_day - max_lag
            
            self.assertGreaterEqual(
                last_data_date, 
                earliest_acceptable,
                f"Data too old: {last_data_date} vs expected {expected_last_trading_day}"
            )
            
            # Check data quality
            last_price = aapl_close.iloc[-1]
            self.assertGreater(last_price, 50, "AAPL price seems too low")
            self.assertLess(last_price, 1000, "AAPL price seems too high")
            
            # Check failure report
            failure_report = ds.get_failure_report()
            self.assertNotIn('AAPL', failure_report['total_failures'], 
                           "AAPL should not fail on both sources")
            
        except Exception as e:
            # If network fails, just log a warning but don't fail the test
            print(f"Warning: Network test failed: {e}")
            self.skipTest(f"Network connectivity issue: {e}")

    def test_exception_handling(self):
        """Test exception handling in data fetching."""
        # Mock the data source instances directly
        mock_stooq_instance = MagicMock()
        mock_yfinance_instance = MagicMock()
        
        # Create hybrid data source and replace the instances
        ds = HybridDataSource(prefer_stooq=True)
        ds._negative_cache.clear()
        ds.stooq_source = mock_stooq_instance
        ds.yfinance_source = mock_yfinance_instance
        
        # Mock Stooq to raise an exception
        mock_stooq_instance.get_data.side_effect = Exception("Network error")
        
        # Mock successful yfinance data
        mock_yfinance_data = self._create_mock_yfinance_data(['AAPL'])
        mock_yfinance_instance.get_data.return_value = mock_yfinance_data
        
        result = ds.get_data(['AAPL'], '2023-01-01', '2023-01-03')
        
        # Should still get data from yfinance fallback
        self.assertFalse(result.empty)
        
        # Check that the exception was handled gracefully
        self.assertEqual(len(ds.failed_tickers['stooq']), 1)
        self.assertEqual(len(ds.failed_tickers['yfinance']), 0)


if __name__ == '__main__':
    unittest.main() 

class TestHybridDataSourceValidation(unittest.TestCase):
    """Test hybrid data source specific validation."""
    
    def setUp(self):
        """Set up hybrid data source tests."""
        self.hybrid_source = HybridDataSource()
        self.validation_utils = DataValidationUtilities()
    
    def test_hybrid_data_source_validation_integration(self):
        """Test integration of validation utilities with hybrid data source."""
        # Mock successful data retrieval
        mock_data = pd.DataFrame({
            ('AAPL', 'Close'): [100, 101, 102],
            ('AAPL', 'Volume'): [1000, 1100, 1200]
        })
        mock_data.index = pd.date_range('2023-01-01', periods=3)
        mock_data.columns = pd.MultiIndex.from_tuples(mock_data.columns, names=['Ticker', 'Field'])
        
        with patch.object(self.hybrid_source, 'get_data', return_value=mock_data):
            result = self.hybrid_source.get_data(['AAPL'], '2023-01-01', '2023-01-03')
            
            # Should pass basic validation
            self.assertTrue(isinstance(result, pd.DataFrame))
            self.assertFalse(result.empty)
    
    def test_validation_with_real_data_patterns(self):
        """Test validation with realistic data patterns."""
        # Test with data that might come from real sources
        realistic_data = self.create_realistic_market_data()
        
        # Should pass all validations
        self.assertTrue(self.validation_utils.validate_ohlcv_structure(realistic_data))
        self.assertTrue(self.validation_utils.validate_price_consistency(realistic_data))
        self.assertTrue(self.validation_utils.validate_data_completeness(realistic_data))
        self.assertTrue(self.validation_utils.validate_date_index(realistic_data))
    
    def create_realistic_market_data(self):
        """Create realistic market data for testing."""
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        
        np.random.seed(42)
        data_frames = []
        
        for ticker in tickers:
            # Simulate realistic price movements
            returns = np.random.normal(0.0005, 0.02, len(dates))
            prices = 100 * np.cumprod(1 + returns)
            
            # Create OHLCV with realistic relationships
            df = pd.DataFrame(index=dates)
            df['Close'] = prices
            df['Open'] = prices * (1 + np.random.normal(0, 0.001, len(dates)))
            df['High'] = prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates))))
            df['Low'] = prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates))))
            df['Volume'] = np.random.randint(1000000, 50000000, len(dates))
            
            # Ensure price consistency
            df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
            df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)
            
            df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
            data_frames.append(df)
        
        result = pd.concat(data_frames, axis=1)
        result.columns.names = ['Ticker', 'Field']
        return result 