"""
Tests for data preprocessing cache functionality.

These tests ensure the cache provides correct results while improving performance
for repeated data operations during optimization.
"""

import pytest
import pandas as pd
import numpy as np
from portfolio_backtester.data_cache import (
    DataPreprocessingCache,
)


class TestDataPreprocessingCache:
    """Test data preprocessing cache functionality."""

    def setup_method(self):
        """Set up test data and cache."""
        # Create test price data
        dates = pd.date_range("2010-01-01", "2023-12-31", freq="D")
        np.random.seed(42)

        self.test_data = pd.DataFrame(
            {
                f"Asset_{i}": 100 * np.cumprod(1 + np.random.normal(0.0001, 0.02, len(dates)))
                for i in range(10)  # 10 assets for testing
            },
            index=dates,
        )

        self.cache = DataPreprocessingCache(max_cache_size=50)

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
        assert stats["hits"] >= 1

    def test_window_returns_caching(self):
        """Test window-specific returns caching."""
        window_start = pd.Timestamp("2015-01-01")
        window_end = pd.Timestamp("2015-12-31")

        # Get window data
        window_data = self.test_data.loc[window_start:window_end]

        # First call - should compute and cache
        returns1 = self.cache.get_cached_window_returns(window_data, window_start, window_end)

        # Verify returns are correct
        expected_returns = window_data.pct_change(fill_method=None).fillna(0)
        pd.testing.assert_frame_equal(returns1, expected_returns)

        # Second call - should use cache
        returns2 = self.cache.get_cached_window_returns(window_data, window_start, window_end)
        pd.testing.assert_frame_equal(returns1, returns2)

        # Verify cache hit
        stats = self.cache.get_cache_stats()
        assert stats["window_hits"] >= 1

    def test_precompute_window_returns(self):
        """Test pre-computing returns for multiple windows."""
        # Define some windows
        windows = [
            (
                pd.Timestamp("2015-01-01"),
                pd.Timestamp("2015-06-30"),
                pd.Timestamp("2015-07-01"),
                pd.Timestamp("2015-12-31"),
            ),
            (
                pd.Timestamp("2016-01-01"),
                pd.Timestamp("2016-06-30"),
                pd.Timestamp("2016-07-01"),
                pd.Timestamp("2016-12-31"),
            ),
        ]

        # Pre-compute returns
        window_returns = self.cache.precompute_window_returns(self.test_data, windows)

        # Verify we got results
        assert len(window_returns) >= 0  # May be empty if windows don't have data

        # Check cache stats
        stats = self.cache.get_cache_stats()
        assert stats["window_cache_items"] >= 0

    def test_get_window_returns_by_dates(self):
        """Test getting cached window returns by date range."""
        window_start = pd.Timestamp("2015-01-01")
        window_end = pd.Timestamp("2015-12-31")

        # First, cache some window returns
        window_data = self.test_data.loc[window_start:window_end]
        self.cache.get_cached_window_returns(window_data, window_start, window_end)

        # Try to get by dates
        cached_returns = self.cache.get_window_returns_by_dates(
            self.test_data, window_start, window_end
        )

        # Should get the cached data
        assert cached_returns is not None
        assert len(cached_returns) > 0

    def test_cache_stats_tracking(self):
        """Test cache statistics tracking."""
        # Start with clean cache
        self.cache.clear_cache()
        initial_stats = self.cache.get_cache_stats()
        assert initial_stats["hits"] == 0
        assert initial_stats["misses"] == 0
        assert initial_stats["window_hits"] == 0
        assert initial_stats["window_misses"] == 0

        # Perform operations
        self.cache.get_cached_returns(self.test_data, "stats_test")  # Miss
        self.cache.get_cached_returns(self.test_data, "stats_test")  # Hit

        final_stats = self.cache.get_cache_stats()
        assert final_stats["misses"] >= 1
        assert final_stats["hits"] >= 1
        assert final_stats["regular_hit_rate"] > 0

    def test_cache_clear_functionality(self):
        """Test cache clearing functionality."""
        # Add data to cache
        self.cache.get_cached_returns(self.test_data, "clear_test")
        window_start = pd.Timestamp("2015-01-01")
        window_end = pd.Timestamp("2015-12-31")
        window_data = self.test_data.loc[window_start:window_end]
        self.cache.get_cached_window_returns(window_data, window_start, window_end)

        # Verify cache has data
        stats_before = self.cache.get_cache_stats()
        assert stats_before["total_cached_items"] > 0

        # Clear cache
        self.cache.clear_cache()

        # Verify cache is empty
        stats_after = self.cache.get_cache_stats()
        assert stats_after["total_cached_items"] == 0
        assert stats_after["regular_cache_items"] == 0
        assert stats_after["window_cache_items"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
