"""
Consolidated data source validation tests.
Merges overlapping validation logic from test_data_sanity_check.py and test_hybrid_data_source.py
"""

import unittest
import pandas as pd
import numpy as np


class DataValidationUtilities:
    """Shared data validation utilities for reuse across different data source tests."""
    
    @staticmethod
    def validate_ohlcv_structure(data: pd.DataFrame) -> bool:
        """Validate OHLCV data structure."""
        if data.empty:
            return False
        
        # Check for MultiIndex columns
        if not isinstance(data.columns, pd.MultiIndex):
            return False
        
        # Check for required fields
        required_fields = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_fields = data.columns.get_level_values(1).unique()
        
        return all(field in available_fields for field in required_fields)
    
    @staticmethod
    def validate_price_consistency(data: pd.DataFrame) -> bool:
        """Validate price consistency (High >= Low, etc.)."""
        if data.empty:
            return True
        
        try:
            for ticker in data.columns.get_level_values(0).unique():
                ticker_data = data[ticker]
                
                # High should be >= Low
                if not (ticker_data['High'] >= ticker_data['Low']).all():
                    return False
                
                # Close should be between High and Low
                if not ((ticker_data['Close'] >= ticker_data['Low']) & 
                       (ticker_data['Close'] <= ticker_data['High'])).all():
                    return False
                
                # Open should be between High and Low
                if not ((ticker_data['Open'] >= ticker_data['Low']) & 
                       (ticker_data['Open'] <= ticker_data['High'])).all():
                    return False
            
            return True
        except Exception:
            return False
    
    @staticmethod
    def validate_data_completeness(data: pd.DataFrame, min_periods: int = 100) -> bool:
        """Validate data completeness."""
        if data.empty:
            return False
        
        # Check minimum number of periods
        if len(data) < min_periods:
            return False
        
        # Check for excessive NaN values
        nan_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if nan_ratio > 0.1:  # More than 10% NaN values
            return False
        
        return True
    
    @staticmethod
    def validate_date_index(data: pd.DataFrame) -> bool:
        """Validate date index properties."""
        if data.empty:
            return False
        
        # Check if index is DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            return False
        
        # Check for duplicates
        if data.index.duplicated().any():
            return False
        
        # Check if sorted
        if not data.index.is_monotonic_increasing:
            return False
        
        return True


class TestConsolidatedDataValidation(unittest.TestCase):
    """Consolidated data validation tests."""
    
    def setUp(self):
        """Set up test environment."""
        self.validation_utils = DataValidationUtilities()
        
        # Create sample valid data
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        tickers = ['AAPL', 'MSFT']
        
        # Generate realistic OHLCV data
        np.random.seed(42)
        data_dict = {}
        
        for ticker in tickers:
            base_price = 100
            prices = []
            
            for i in range(len(dates)):
                # Generate realistic OHLCV
                prev_close = base_price if i == 0 else prices[i-1]['Close']
                
                # Random walk for close price
                close = prev_close * (1 + np.random.normal(0, 0.02))
                
                # Generate other prices relative to close
                high = close * (1 + abs(np.random.normal(0, 0.01)))
                low = close * (1 - abs(np.random.normal(0, 0.01)))
                open_price = prev_close * (1 + np.random.normal(0, 0.005))
                
                # Ensure price consistency
                high = max(high, close, open_price, low)
                low = min(low, close, open_price)
                
                volume = np.random.randint(1000000, 10000000)
                
                prices.append({
                    'Open': open_price,
                    'High': high,
                    'Low': low,
                    'Close': close,
                    'Volume': volume
                })
            
            data_dict[ticker] = pd.DataFrame(prices, index=dates)
        
        # Create MultiIndex DataFrame
        self.valid_data = pd.concat(data_dict, axis=1)
        self.valid_data.columns.names = ['Ticker', 'Field']
    
    def test_ohlcv_structure_validation(self):
        """Test OHLCV structure validation."""
        # Valid data should pass
        self.assertTrue(self.validation_utils.validate_ohlcv_structure(self.valid_data))
        
        # Empty data should fail
        empty_data = pd.DataFrame()
        self.assertFalse(self.validation_utils.validate_ohlcv_structure(empty_data))
        
        # Data without MultiIndex should fail
        simple_data = pd.DataFrame({'A': [1, 2, 3]})
        self.assertFalse(self.validation_utils.validate_ohlcv_structure(simple_data))
        
        # Data missing required fields should fail
        incomplete_data = self.valid_data[['AAPL']].copy()
        incomplete_data = incomplete_data.drop('Volume', axis=1, level=1)
        self.assertFalse(self.validation_utils.validate_ohlcv_structure(incomplete_data))
    
    def test_price_consistency_validation(self):
        """Test price consistency validation."""
        # Valid data should pass
        self.assertTrue(self.validation_utils.validate_price_consistency(self.valid_data))
        
        # Create inconsistent data
        inconsistent_data = self.valid_data.copy()
        # Make High < Low (invalid)
        inconsistent_data.loc[inconsistent_data.index[0], ('AAPL', 'High')] = 50
        inconsistent_data.loc[inconsistent_data.index[0], ('AAPL', 'Low')] = 100
        
        self.assertFalse(self.validation_utils.validate_price_consistency(inconsistent_data))
    
    def test_data_completeness_validation(self):
        """Test data completeness validation."""
        # Valid data should pass
        self.assertTrue(self.validation_utils.validate_data_completeness(self.valid_data))
        
        # Data with too few periods should fail
        short_data = self.valid_data.head(50)
        self.assertFalse(self.validation_utils.validate_data_completeness(short_data, min_periods=100))
        
        # Data with too many NaN values should fail
        nan_data = self.valid_data.copy()
        nan_data.iloc[:50] = np.nan  # 25% NaN values
        self.assertFalse(self.validation_utils.validate_data_completeness(nan_data))
    
    def test_date_index_validation(self):
        """Test date index validation."""
        # Valid data should pass
        self.assertTrue(self.validation_utils.validate_date_index(self.valid_data))
        
        # Data with non-datetime index should fail
        non_datetime_data = self.valid_data.copy()
        non_datetime_data.index = range(len(non_datetime_data))
        self.assertFalse(self.validation_utils.validate_date_index(non_datetime_data))
        
        # Data with duplicate dates should fail
        duplicate_data = self.valid_data.copy()
        duplicate_data.index = [duplicate_data.index[0]] * len(duplicate_data)
        self.assertFalse(self.validation_utils.validate_date_index(duplicate_data))





if __name__ == '__main__':
    unittest.main()