"""
Tests for data preprocessing cache functionality.

These tests ensure the cache provides correct results while improving performance
for repeated data operations during optimization.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.portfolio_backtester.data_cache import DataPreprocessingCache, get_global_cache, clear_global_cache


class TestDataPreprocessingCache:
    """Test data preprocessing cache functionality."""
    
    def setup_method(self):
        """Set up test data and cache."""
        # Create test price data
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        np.random.seed(42)
        
        self.test_data = pd.DataFrame({
            'AAPL': 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates))),
            'MSFT': 200 * np.cumprod(1 + np.random.normal(0.0008, 0.018, len(dates))),
            'GOOGL': 1500 * np.cumprod(1 + np.random.normal(0.0012, 0.025, len(dates)))
        }, index=dates)
        
        self.cache = DataPreprocessingCache(max_cache_size_mb=100)
    
    def test_cached_returns_computation(self):
        """Test that cached returns are computed correctly and cached."""
        # First call - should compute and cache
        returns1 = self.cache.get_cached_returns(self.test_data, "test1")
        
        # Verify returns are correct
        expected_returns = self.test_data.pct_change(fill_method=None).fillna(0)
        pd.testing.assert_frame_equal(returns1, expected_returns)
        
        # Second call - should use cache
        returns2 = self.cache.get_cached_returns(self.test_data, "test1")
        pd.testing.assert_frame_equal(returns1, returns2)
        
        # Verify cache hit
        stats = self.cache.get_cache_stats()
        assert stats['hits'] >= 1
    
    def test_date_position_mapping(self):
        """Test date-to-position mapping functionality."""
        # Get position map
        position_map = self.cache.get_date_position_map(self.test_data, "test_positions")
        
        # Verify mapping is correct
        assert len(position_map) == len(self.test_data)
        assert position_map[self.test_data.index[0]] == 0
        assert position_map[self.test_data.index[-1]] == len(self.test_data) - 1
        
        # Test caching
        position_map2 = self.cache.get_date_position_map(self.test_data, "test_positions")
        assert position_map == position_map2
        
        # Verify cache hit
        stats = self.cache.get_cache_stats()
        assert stats['hits'] >= 1
    
    def test_fast_data_slicing(self):
        """Test fast data slicing with position mapping."""
        # Get position map first
        position_map = self.cache.get_date_position_map(self.test_data, "slice_test")
        
        # Test slicing at various dates
        test_date = self.test_data.index[100]  # Some date in the middle
        
        # Fast slice using cache
        sliced_fast = self.cache.get_data_slice_fast(
            self.test_data, test_date, position_map, "slice_test"
        )
        
        # Traditional slice for comparison
        sliced_traditional = self.test_data[self.test_data.index <= test_date]
        
        # Should be identical
        pd.testing.assert_frame_equal(sliced_fast, sliced_traditional)
        
        # Test caching of slices
        sliced_fast2 = self.cache.get_data_slice_fast(
            self.test_data, test_date, position_map, "slice_test"
        )
        pd.testing.assert_frame_equal(sliced_fast, sliced_fast2)
    
    def test_cache_performance_improvement(self):
        """Test that cache provides performance improvement."""
        import time
        
        # Measure time for first computation (cache miss)
        start_time = time.time()
        returns1 = self.cache.get_cached_returns(self.test_data, "perf_test")
        first_time = time.time() - start_time
        
        # Measure time for second computation (cache hit)
        start_time = time.time()
        returns2 = self.cache.get_cached_returns(self.test_data, "perf_test")
        second_time = time.time() - start_time
        
        # Cache hit should be faster (though may be minimal for small data)
        assert second_time <= first_time
        pd.testing.assert_frame_equal(returns1, returns2)
    
    def test_cache_with_different_identifiers(self):
        """Test that different identifiers create separate cache entries."""
        returns1 = self.cache.get_cached_returns(self.test_data, "id1")
        returns2 = self.cache.get_cached_returns(self.test_data, "id2")
        
        # Should be identical data but separate cache entries
        pd.testing.assert_frame_equal(returns1, returns2)
        
        # Should have separate entries in cache
        stats = self.cache.get_cache_stats()
        assert stats['cached_returns_count'] >= 2
    
    def test_cache_size_management(self):
        """Test cache size management and cleanup."""
        # Create a small cache
        small_cache = DataPreprocessingCache(max_cache_size_mb=1)  # Very small limit
        
        # Add multiple large datasets
        for i in range(10):
            large_data = pd.DataFrame(
                np.random.randn(1000, 50),  # Large dataset
                index=pd.date_range('2020-01-01', periods=1000, freq='D'),
                columns=[f'Asset_{j}' for j in range(50)]
            )
            small_cache.get_cached_returns(large_data, f"large_data_{i}")
        
        # Cache should have been cleaned up
        stats = small_cache.get_cache_stats()
        assert stats['current_size_mb'] <= small_cache.max_cache_size_mb * 1.1  # Allow small tolerance
    
    def test_rolling_window_indices(self):
        """Test rolling window indices caching."""
        data_length = 100
        window_size = 20
        
        # Get indices
        indices1 = self.cache.get_rolling_window_indices(data_length, window_size)
        indices2 = self.cache.get_rolling_window_indices(data_length, window_size)
        
        # Should be identical (cached)
        np.testing.assert_array_equal(indices1, indices2)
        
        # Verify correctness
        expected_length = max(0, data_length - window_size + 1)
        assert len(indices1) == expected_length
        assert indices1[0] == 0
        if len(indices1) > 1:
            assert indices1[-1] == expected_length - 1
    
    def test_cache_stats_tracking(self):
        """Test cache statistics tracking."""
        # Start with clean cache
        self.cache.clear_cache()
        initial_stats = self.cache.get_cache_stats()
        assert initial_stats['hits'] == 0
        assert initial_stats['misses'] == 0
        assert initial_stats['hit_rate'] == 0
        
        # Perform operations
        self.cache.get_cached_returns(self.test_data, "stats_test")  # Miss
        self.cache.get_cached_returns(self.test_data, "stats_test")  # Hit
        
        final_stats = self.cache.get_cache_stats()
        assert final_stats['misses'] >= 1
        assert final_stats['hits'] >= 1
        assert final_stats['hit_rate'] > 0
    
    def test_cache_clear_functionality(self):
        """Test cache clearing functionality."""
        # Add data to cache
        self.cache.get_cached_returns(self.test_data, "clear_test")
        position_map = self.cache.get_date_position_map(self.test_data, "clear_test")
        
        # Verify cache has data
        stats_before = self.cache.get_cache_stats()
        assert stats_before['cached_returns_count'] > 0
        
        # Clear cache
        self.cache.clear_cache()
        
        # Verify cache is empty
        stats_after = self.cache.get_cache_stats()
        assert stats_after['cached_returns_count'] == 0
        assert stats_after['cached_slices_count'] == 0
        assert stats_after['cached_position_maps_count'] == 0
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty DataFrame
        empty_df = pd.DataFrame()
        returns_empty = self.cache.get_cached_returns(empty_df, "empty")
        assert returns_empty.empty
        
        # Single row DataFrame
        single_row = self.test_data.iloc[:1]
        returns_single = self.cache.get_cached_returns(single_row, "single")
        assert len(returns_single) == 1
        
        # DataFrame with NaN values
        nan_data = self.test_data.copy()
        nan_data.iloc[10:20] = np.nan
        returns_nan = self.cache.get_cached_returns(nan_data, "nan_test")
        assert len(returns_nan) == len(nan_data)
    
    def test_data_slice_with_missing_date(self):
        """Test data slicing with date not in index."""
        position_map = self.cache.get_date_position_map(self.test_data, "missing_date_test")
        
        # Use a date not in the index
        missing_date = self.test_data.index[50] + timedelta(hours=12)
        
        # Should fallback to boolean indexing
        sliced_data = self.cache.get_data_slice_fast(
            self.test_data, missing_date, position_map, "missing_date_test"
        )
        
        # Should still work correctly
        expected_slice = self.test_data[self.test_data.index <= missing_date]
        pd.testing.assert_frame_equal(sliced_data, expected_slice)


class TestGlobalCache:
    """Test global cache functionality."""
    
    def test_global_cache_singleton(self):
        """Test that global cache is a singleton."""
        cache1 = get_global_cache()
        cache2 = get_global_cache()
        
        # Should be the same instance
        assert cache1 is cache2
    
    def test_global_cache_clear(self):
        """Test global cache clearing."""
        cache = get_global_cache()
        
        # Add some data
        test_data = pd.DataFrame({'A': [1, 2, 3]}, index=pd.date_range('2020-01-01', periods=3))
        cache.get_cached_returns(test_data, "global_test")
        
        # Verify data exists
        stats_before = cache.get_cache_stats()
        assert stats_before['cached_returns_count'] > 0
        
        # Clear global cache
        clear_global_cache()
        
        # Verify cache is cleared
        stats_after = cache.get_cache_stats()
        assert stats_after['cached_returns_count'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])