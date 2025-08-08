"""
Test suite for OptimizedDataGenerator class.

This module tests the optimized data generation utilities including:
- Cached OHLCV data generation
- Standardized date range generation
- Lazy loading mechanisms
- Data validation utilities
"""

import pytest
import pandas as pd
import numpy as np

from tests.fixtures.optimized_data_generator import OptimizedDataGenerator


class TestOptimizedDataGenerator:
    """Test cases for OptimizedDataGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Clear cache before each test
        OptimizedDataGenerator.clear_cache()
        
        # Standard test parameters
        self.test_tickers = ('AAPL', 'MSFT', 'GOOGL')
        self.test_start = '2020-01-01'
        self.test_end = '2020-12-31'
        self.test_freq = 'B'
    
    def test_generate_cached_ohlcv_data_basic(self):
        """Test basic OHLCV data generation with caching."""
        data = OptimizedDataGenerator.generate_cached_ohlcv_data(
            tickers=self.test_tickers,
            start_date=self.test_start,
            end_date=self.test_end,
            freq=self.test_freq,
            pattern='random'
        )
        
        # Check basic structure
        assert isinstance(data, pd.DataFrame)
        assert isinstance(data.columns, pd.MultiIndex)
        assert list(data.columns.names) == ['Ticker', 'Field']
        assert isinstance(data.index, pd.DatetimeIndex)
        
        # Check tickers and fields
        tickers = data.columns.get_level_values('Ticker').unique()
        fields = data.columns.get_level_values('Field').unique()
        
        assert set(tickers) == set(self.test_tickers)
        assert set(fields) == {'Open', 'High', 'Low', 'Close', 'Volume'}
        
        # Check data is not empty
        assert not data.empty
        assert len(data) > 0
    
    def test_caching_functionality(self):
        """Test that caching works correctly."""
        # Clear cache and get initial cache info
        OptimizedDataGenerator.clear_cache()
        initial_info = OptimizedDataGenerator.get_cache_info()
        assert initial_info['lru_cache_hits'] == 0
        assert initial_info['lru_cache_misses'] == 0
        
        # Generate data first time (should be cache miss)
        data1 = OptimizedDataGenerator.generate_cached_ohlcv_data(
            tickers=self.test_tickers,
            start_date=self.test_start,
            end_date=self.test_end,
            freq=self.test_freq,
            pattern='random',
            seed=42
        )
        
        cache_info_after_first = OptimizedDataGenerator.get_cache_info()
        assert cache_info_after_first['lru_cache_misses'] == 1
        
        # Generate same data again (should be cache hit)
        data2 = OptimizedDataGenerator.generate_cached_ohlcv_data(
            tickers=self.test_tickers,
            start_date=self.test_start,
            end_date=self.test_end,
            freq=self.test_freq,
            pattern='random',
            seed=42
        )
        
        cache_info_after_second = OptimizedDataGenerator.get_cache_info()
        assert cache_info_after_second['lru_cache_hits'] == 1
        
        # Data should be identical
        pd.testing.assert_frame_equal(data1, data2)
    
    def test_different_patterns(self):
        """Test different data generation patterns."""
        patterns = ['random', 'trending_up', 'trending_down', 'volatile', 'stable']
        
        for pattern in patterns:
            data = OptimizedDataGenerator.generate_cached_ohlcv_data(
                tickers=('TEST',),
                start_date='2020-01-01',
                end_date='2020-03-31',
                freq='B',
                pattern=pattern,
                seed=42
            )
            
            assert isinstance(data, pd.DataFrame)
            assert not data.empty
            
            # Check that different patterns produce different results
            close_prices = data[('TEST', 'Close')]
            assert len(close_prices) > 0
            assert not close_prices.isna().all()
    
    def test_create_standardized_date_range(self):
        """Test standardized date range generation."""
        # Test different frequencies
        test_cases = [
            ('2020-01-01', '2020-12-31', 'D'),
            ('2020-01-01', '2020-12-31', 'B'),
            ('2020-01-01', '2020-12-31', 'ME'),
            ('2020-01-01', '2020-12-31', 'M'),  # Should be converted to ME
        ]
        
        for start, end, freq in test_cases:
            date_range = OptimizedDataGenerator.create_standardized_date_range(
                start, end, freq
            )
            
            assert isinstance(date_range, pd.DatetimeIndex)
            assert len(date_range) > 0
            assert date_range[0] >= pd.to_datetime(start)
            assert date_range[-1] <= pd.to_datetime(end)
    
    def test_create_business_date_range(self):
        """Test business date range generation."""
        # Test without holidays
        bdate_range = OptimizedDataGenerator.create_business_date_range(
            '2020-01-01', '2020-01-31'
        )
        
        assert isinstance(bdate_range, pd.DatetimeIndex)
        assert len(bdate_range) > 0
        
        # Check that weekends are excluded
        for date in bdate_range:
            assert date.weekday() < 5  # Monday=0, Friday=4
        
        # Test with holidays
        holidays = ['2020-01-15', '2020-01-20']
        bdate_range_with_holidays = OptimizedDataGenerator.create_business_date_range(
            '2020-01-01', '2020-01-31', holidays=holidays
        )
        
        # Should be shorter than without holidays
        assert len(bdate_range_with_holidays) < len(bdate_range)
        
        # Check that holidays are excluded
        holiday_dates = pd.to_datetime(holidays)
        for holiday in holiday_dates:
            assert holiday not in bdate_range_with_holidays
    
    def test_lazy_data_loader_decorator(self):
        """Test lazy loading decorator functionality."""
        call_count = 0
        
        @OptimizedDataGenerator.lazy_data_loader('test_key')
        def expensive_computation():
            nonlocal call_count
            call_count += 1
            return pd.DataFrame({'A': [1, 2, 3]})
        
        # First call should execute function
        result1 = expensive_computation()
        assert call_count == 1
        assert isinstance(result1, pd.DataFrame)
        
        # Second call should use cached result
        result2 = expensive_computation()
        assert call_count == 1  # Should not increment
        pd.testing.assert_frame_equal(result1, result2)
        
        # Clear cache and call again
        OptimizedDataGenerator.clear_cache()
        expensive_computation()
        assert call_count == 2  # Should increment again
    
    def test_validate_ohlcv_data_structure_valid(self):
        """Test validation of valid OHLCV data structure."""
        # Generate valid data
        data = OptimizedDataGenerator.generate_cached_ohlcv_data(
            tickers=('AAPL', 'MSFT'),
            start_date='2020-01-01',
            end_date='2020-01-31',
            freq='B'
        )
        
        # Should not raise any exceptions
        result = OptimizedDataGenerator.validate_ohlcv_data_structure(data)
        assert result is True
    
    def test_validate_ohlcv_data_structure_invalid(self):
        """Test validation with invalid data structures."""
        # Test with non-DataFrame
        with pytest.raises(ValueError, match="Data must be a pandas DataFrame"):
            OptimizedDataGenerator.validate_ohlcv_data_structure("not a dataframe")
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="Data cannot be empty"):
            OptimizedDataGenerator.validate_ohlcv_data_structure(empty_df)
        
        # Test with non-MultiIndex columns
        simple_df = pd.DataFrame({'A': [1, 2, 3]})
        with pytest.raises(ValueError, match="Data must have MultiIndex columns"):
            OptimizedDataGenerator.validate_ohlcv_data_structure(simple_df)
        
        # Test with wrong column names
        wrong_columns = pd.MultiIndex.from_product([['AAPL'], ['Price']], names=['Symbol', 'Type'])
        wrong_df = pd.DataFrame([[100]], columns=wrong_columns)
        with pytest.raises(ValueError, match="Column names must be"):
            OptimizedDataGenerator.validate_ohlcv_data_structure(wrong_df)
        
        # Test with missing fields
        incomplete_columns = pd.MultiIndex.from_product([['AAPL'], ['Open']], names=['Ticker', 'Field'])
        incomplete_df = pd.DataFrame([[100]], columns=incomplete_columns)
        with pytest.raises(ValueError, match="Missing required fields"):
            OptimizedDataGenerator.validate_ohlcv_data_structure(incomplete_df)
    
    def test_validate_date_range_consistency(self):
        """Test date range consistency validation."""
        # Generate test data
        data = OptimizedDataGenerator.generate_cached_ohlcv_data(
            tickers=('AAPL',),
            start_date='2020-01-01',
            end_date='2020-01-31',
            freq='B'
        )
        
        # Should validate successfully
        result = OptimizedDataGenerator.validate_date_range_consistency(
            data, '2020-01-01', '2020-01-31', 'B'
        )
        assert result is True
        
        # Test with wrong expected range
        with pytest.raises(ValueError, match="Date range length mismatch"):
            OptimizedDataGenerator.validate_date_range_consistency(
                data, '2020-01-01', '2020-12-31', 'B'  # Much longer range
            )
    
    def test_validate_data_types(self):
        """Test data type validation."""
        # Generate valid data
        data = OptimizedDataGenerator.generate_cached_ohlcv_data(
            tickers=('AAPL',),
            start_date='2020-01-01',
            end_date='2020-01-31',
            freq='B'
        )
        
        # Should validate successfully
        result = OptimizedDataGenerator.validate_data_types(data)
        assert result is True
        
        # Test with invalid data types
        invalid_data = data.copy()
        string_values = ['not_numeric'] * len(invalid_data)
        invalid_data[('AAPL', 'Close')] = string_values
        
        with pytest.raises(ValueError, match="must be numeric"):
            OptimizedDataGenerator.validate_data_types(invalid_data)
        
        # Test with negative prices
        negative_data = data.copy()
        negative_data[('AAPL', 'Close')] = -100
        
        with pytest.raises(ValueError, match="contains negative values"):
            OptimizedDataGenerator.validate_data_types(negative_data)
    
    def test_create_test_data_with_validation(self):
        """Test comprehensive data creation with validation."""
        data = OptimizedDataGenerator.create_test_data_with_validation(
            tickers=['AAPL', 'MSFT'],
            start_date='2020-01-01',
            end_date='2020-03-31',
            freq='B',
            pattern='random'
        )
        
        # Should return valid, validated data
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        
        # All validations should pass (no exceptions raised)
        OptimizedDataGenerator.validate_ohlcv_data_structure(data)
        OptimizedDataGenerator.validate_date_range_consistency(data, '2020-01-01', '2020-03-31', 'B')
        OptimizedDataGenerator.validate_data_types(data)
    
    def test_ohlc_relationships(self):
        """Test that OHLC relationships are maintained."""
        data = OptimizedDataGenerator.generate_cached_ohlcv_data(
            tickers=('TEST',),
            start_date='2020-01-01',
            end_date='2020-01-31',
            freq='B',
            pattern='random'
        )
        
        open_prices = data[('TEST', 'Open')]
        high_prices = data[('TEST', 'High')]
        low_prices = data[('TEST', 'Low')]
        close_prices = data[('TEST', 'Close')]
        
        # High should be >= max(Open, Close)
        max_oc = np.maximum(open_prices, close_prices)
        assert (high_prices >= max_oc * 0.99).all()  # Allow small floating point errors
        
        # Low should be <= min(Open, Close)
        min_oc = np.minimum(open_prices, close_prices)
        assert (low_prices <= min_oc * 1.01).all()  # Allow small floating point errors
        
        # All prices should be positive
        assert (open_prices > 0).all()
        assert (high_prices > 0).all()
        assert (low_prices > 0).all()
        assert (close_prices > 0).all()
    
    def test_volume_characteristics(self):
        """Test volume data characteristics."""
        data = OptimizedDataGenerator.generate_cached_ohlcv_data(
            tickers=('TEST',),
            start_date='2020-01-01',
            end_date='2020-01-31',
            freq='B',
            pattern='random'
        )
        
        volume = data[('TEST', 'Volume')]
        
        # Volume should be positive integers
        assert (volume > 0).all()
        assert volume.dtype in [np.int64, np.int32, np.float64]  # May be float due to pandas
        
        # Volume should be reasonable (not too small or too large)
        assert volume.min() >= 100000  # At least 100k
        assert volume.max() <= 100000000  # At most 100M
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test invalid pattern
        with pytest.raises(ValueError, match="Unknown pattern"):
            OptimizedDataGenerator.generate_cached_ohlcv_data(
                tickers=('TEST',),
                start_date='2020-01-01',
                end_date='2020-01-31',
                pattern='invalid_pattern'
            )
        
        # Test invalid date range
        with pytest.raises(ValueError, match="Empty date range"):
            OptimizedDataGenerator.generate_cached_ohlcv_data(
                tickers=('TEST',),
                start_date='2020-12-31',
                end_date='2020-01-01',  # End before start
                freq='B'
            )
        
        # Test invalid frequency
        with pytest.raises(ValueError, match="Failed to create date range"):
            OptimizedDataGenerator.create_standardized_date_range(
                '2020-01-01', '2020-01-31', 'INVALID_FREQ'
            )
    
    def test_performance_characteristics(self):
        """Test performance characteristics of cached generation."""
        import time
        
        # Generate data first time and measure time
        start_time = time.time()
        data1 = OptimizedDataGenerator.generate_cached_ohlcv_data(
            tickers=('PERF_TEST',),
            start_date='2020-01-01',
            end_date='2020-12-31',
            freq='B',
            pattern='random',
            seed=42
        )
        first_time = time.time() - start_time
        
        # Generate same data again (should be much faster due to caching)
        start_time = time.time()
        data2 = OptimizedDataGenerator.generate_cached_ohlcv_data(
            tickers=('PERF_TEST',),
            start_date='2020-01-01',
            end_date='2020-12-31',
            freq='B',
            pattern='random',
            seed=42
        )
        second_time = time.time() - start_time
        
        # Cached version should be significantly faster
        assert second_time < first_time * 0.1  # At least 10x faster
        
        # Data should be identical
        pd.testing.assert_frame_equal(data1, data2)
    
    def test_memory_efficiency(self):
        """Test memory efficiency of the caching system."""
        # Generate multiple datasets
        datasets = []
        for i in range(5):
            data = OptimizedDataGenerator.generate_cached_ohlcv_data(
                tickers=(f'MEM_TEST_{i}',),
                start_date='2020-01-01',
                end_date='2020-03-31',
                freq='B',
                pattern='random',
                seed=i
            )
            datasets.append(data)
        
        # Check cache info
        cache_info = OptimizedDataGenerator.get_cache_info()
        assert cache_info['lru_cache_size'] <= cache_info['lru_cache_maxsize']
        
        # All datasets should be valid
        for data in datasets:
            assert isinstance(data, pd.DataFrame)
            assert not data.empty
    
    def teardown_method(self):
        """Clean up after each test."""
        OptimizedDataGenerator.clear_cache()


if __name__ == '__main__':
    pytest.main([__file__])