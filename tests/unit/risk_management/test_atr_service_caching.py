"""
Unit tests for ATR service caching functionality.
"""

import pandas as pd
import numpy as np

from portfolio_backtester.risk_management.atr_service import get_atr_service


class TestATRServiceCaching:
    """Test ATR service caching functionality."""

    def test_atr_service_caching_basic(self):
        """Test basic ATR service caching functionality."""
        # Create test data
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        tickers = ["AAPL", "MSFT", "GOOGL"]
        fields = ["Open", "High", "Low", "Close"]
        columns = pd.MultiIndex.from_product([tickers, fields], names=["Ticker", "Field"])

        data = {}
        for ticker in tickers:
            base_price = {"AAPL": 150.0, "MSFT": 250.0, "GOOGL": 100.0}[ticker]
            prices = base_price * (1 + np.random.randn(len(dates)) * 0.02).cumprod()
            data[(ticker, "Open")] = prices * (1 + np.random.randn(len(dates)) * 0.005)
            data[(ticker, "High")] = prices * (1 + np.abs(np.random.randn(len(dates))) * 0.01)
            data[(ticker, "Low")] = prices * (1 - np.abs(np.random.randn(len(dates))) * 0.01)
            data[(ticker, "Close")] = prices

        historical_data = pd.DataFrame(data, index=dates, columns=columns)
        current_date = pd.Timestamp("2023-01-20")
        atr_length = 14

        # Test caching
        atr_service = get_atr_service()
        atr_service.clear_cache()
        initial_cache_size = atr_service.cache_info()["size"]

        # First calculation
        result1 = atr_service.calculate_atr(historical_data, current_date, atr_length)
        cache_size_after_first = atr_service.cache_info()["size"]

        # Second calculation with same parameters
        result2 = atr_service.calculate_atr(historical_data, current_date, atr_length)
        cache_size_after_second = atr_service.cache_info()["size"]

        # Assertions
        assert cache_size_after_first == initial_cache_size + 1
        assert cache_size_after_second == initial_cache_size + 1  # Should be cache hit
        pd.testing.assert_series_equal(result1, result2)

    def test_atr_service_cache_key_uniqueness(self):
        """Test that different cache keys are created for different parameters."""
        # Create test data
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        tickers = ["AAPL"]
        fields = ["Open", "High", "Low", "Close"]
        columns = pd.MultiIndex.from_product([tickers, fields], names=["Ticker", "Field"])

        data = {}
        base_price = 150.0
        prices = base_price * (1 + np.random.randn(len(dates)) * 0.02).cumprod()
        data[("AAPL", "Open")] = prices * (1 + np.random.randn(len(dates)) * 0.005)
        data[("AAPL", "High")] = prices * (1 + np.abs(np.random.randn(len(dates))) * 0.01)
        data[("AAPL", "Low")] = prices * (1 - np.abs(np.random.randn(len(dates))) * 0.01)
        data[("AAPL", "Close")] = prices

        historical_data = pd.DataFrame(data, index=dates, columns=columns)

        atr_service = get_atr_service()
        atr_service.clear_cache()

        # Calculate ATR with different parameters
        result1 = atr_service.calculate_atr(historical_data, pd.Timestamp("2023-01-20"), 14)
        result2 = atr_service.calculate_atr(
            historical_data, pd.Timestamp("2023-01-19"), 14
        )  # Different date
        result3 = atr_service.calculate_atr(
            historical_data, pd.Timestamp("2023-01-20"), 10
        )  # Different length

        # Should have 3 different cache entries
        assert atr_service.cache_info()["size"] == 3

        # Results should be different
        assert not result1.equals(result2)  # Different date
        assert not result1.equals(result3)  # Different ATR length

    def test_atr_service_cache_invalidation(self):
        """Test that cache is properly invalidated with different data."""
        # Create test data
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        tickers = ["AAPL"]
        fields = ["Open", "High", "Low", "Close"]
        columns = pd.MultiIndex.from_product([tickers, fields], names=["Ticker", "Field"])

        # Create first dataset
        data1 = {}
        base_price = 150.0
        prices = base_price * (1 + np.random.randn(len(dates)) * 0.02).cumprod()
        data1[("AAPL", "Open")] = prices * (1 + np.random.randn(len(dates)) * 0.005)
        data1[("AAPL", "High")] = prices * (1 + np.abs(np.random.randn(len(dates))) * 0.01)
        data1[("AAPL", "Low")] = prices * (1 - np.abs(np.random.randn(len(dates))) * 0.01)
        data1[("AAPL", "Close")] = prices

        # Create second dataset (different data)
        np.random.seed(123)  # Different seed
        data2 = {}
        prices2 = base_price * (1 + np.random.randn(len(dates)) * 0.02).cumprod()
        data2[("AAPL", "Open")] = prices2 * (1 + np.random.randn(len(dates)) * 0.005)
        data2[("AAPL", "High")] = prices2 * (1 + np.abs(np.random.randn(len(dates))) * 0.01)
        data2[("AAPL", "Low")] = prices2 * (1 - np.abs(np.random.randn(len(dates))) * 0.01)
        data2[("AAPL", "Close")] = prices2

        historical_data1 = pd.DataFrame(data1, index=dates, columns=columns)
        historical_data2 = pd.DataFrame(data2, index=dates, columns=columns)

        current_date = pd.Timestamp("2023-01-20")

        atr_service = get_atr_service()
        atr_service.clear_cache()

        # Calculate ATR with first dataset
        result1 = atr_service.calculate_atr(historical_data1, current_date, 14)
        assert atr_service.cache_info()["size"] == 1

        # Calculate ATR with second dataset (should create new cache entry due to different data_timestamp)
        result2 = atr_service.calculate_atr(historical_data2, current_date, 14)
        assert atr_service.cache_info()["size"] == 2

        # Results should be different due to different underlying data
        assert not result1.equals(result2)

    def test_atr_service_thread_safety(self):
        """Test thread safety of cache operations."""
        import threading

        # Create test data
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        tickers = ["AAPL"]
        fields = ["Open", "High", "Low", "Close"]
        columns = pd.MultiIndex.from_product([tickers, fields], names=["Ticker", "Field"])

        data = {}
        base_price = 150.0
        prices = base_price * (1 + np.random.randn(len(dates)) * 0.02).cumprod()
        data[("AAPL", "Open")] = prices * (1 + np.random.randn(len(dates)) * 0.005)
        data[("AAPL", "High")] = prices * (1 + np.abs(np.random.randn(len(dates))) * 0.01)
        data[("AAPL", "Low")] = prices * (1 - np.abs(np.random.randn(len(dates))) * 0.01)
        data[("AAPL", "Close")] = prices

        historical_data = pd.DataFrame(data, index=dates, columns=columns)
        current_date = pd.Timestamp("2023-01-20")

        atr_service = get_atr_service()
        atr_service.clear_cache()

        results = []
        errors = []

        def worker():
            try:
                result = atr_service.calculate_atr(historical_data, current_date, 14)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Run multiple threads concurrently
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0, f"Errors occurred in threads: {errors}"
        assert len(results) == 10

        # All results should be identical (cache hit or same calculation)
        for result in results[1:]:
            pd.testing.assert_series_equal(results[0], result)

        # Cache should have one entry (thread-safe)
        assert atr_service.cache_info()["size"] == 1
