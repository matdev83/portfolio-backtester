"""
Performance-optimized fixtures for test data generation.
Implements caching and lazy loading to reduce test execution time.
"""

import pytest
import pandas as pd
import numpy as np
from functools import lru_cache
from typing import Dict, Any


class PerformanceOptimizedFixtures:
    """Performance-optimized test data fixtures with caching."""

    @staticmethod
    @lru_cache(maxsize=32)
    def generate_cached_ohlcv_data(
        start_date: str = "2020-01-01",
        end_date: str = "2023-12-31",
        num_assets: int = 100,
        freq: str = "D",
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate cached OHLCV data for performance tests."""
        np.random.seed(seed)

        dates = pd.date_range(start_date, end_date, freq=freq)
        tickers = [f"ASSET_{i:03d}" for i in range(num_assets)]

        # Generate synthetic price data efficiently
        returns = np.random.normal(0.0005, 0.02, (len(dates), num_assets))
        prices = 100 * np.cumprod(1 + returns, axis=0)

        # Create OHLCV data efficiently
        opens = prices * (1 + np.random.normal(0, 0.001, prices.shape))
        highs = prices * (1 + np.abs(np.random.normal(0, 0.005, prices.shape)))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.005, prices.shape)))
        volumes = np.random.randint(1000, 10000, prices.shape)

        # Create MultiIndex columns
        fields = ["Open", "High", "Low", "Close", "Volume"]
        columns = pd.MultiIndex.from_product([tickers, fields], names=["Ticker", "Field"])

        # Create data properly - each asset gets all fields for each date
        data_list = []
        for i, ticker in enumerate(tickers):
            ticker_data = np.column_stack(
                [opens[:, i], highs[:, i], lows[:, i], prices[:, i], volumes[:, i]]
            )
            data_list.append(ticker_data)

        # Concatenate all ticker data horizontally
        data = np.concatenate(data_list, axis=1)

        return pd.DataFrame(data, index=dates, columns=columns)

    @staticmethod
    @lru_cache(maxsize=16)
    def generate_cached_strategy_config(
        strategy_type: str = "momentum", lookback_months: int = 3, num_holdings: int = 10
    ) -> Dict[str, Any]:
        """Generate cached strategy configurations."""
        base_config = {
            "strategy_params": {
                "lookback_months": lookback_months,
                "skip_months": 1,
                "num_holdings": num_holdings,
                "smoothing_lambda": 0.5,
                "leverage": 1.0,
                "trade_longs": True,
                "trade_shorts": False,
                "price_column_asset": "Close",
                "price_column_benchmark": "Close",
            }
        }

        if strategy_type == "momentum":
            base_config["strategy_params"]["top_decile_fraction"] = 0.1
        elif strategy_type == "rsi":
            base_config["strategy_params"].update({"rsi_period": 14, "rsi_threshold": 30.0})

        return base_config

    @staticmethod
    @lru_cache(maxsize=8)
    def generate_cached_benchmark_data(
        start_date: str = "2020-01-01",
        end_date: str = "2023-12-31",
        freq: str = "D",
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate cached benchmark data."""
        np.random.seed(seed + 1)  # Different seed for benchmark

        dates = pd.date_range(start_date, end_date, freq=freq)
        returns = np.random.normal(0.0008, 0.015, len(dates))
        prices = 100 * np.cumprod(1 + returns)

        return pd.DataFrame({"Close": prices}, index=dates)


# Pytest fixtures using the cached generators
@pytest.fixture(scope="session")
def small_ohlcv_data():
    """Small dataset for fast unit tests."""
    return PerformanceOptimizedFixtures.generate_cached_ohlcv_data(
        start_date="2022-01-01",
        end_date="2022-12-31",
        num_assets=10,
        freq="ME",  # Monthly for faster tests
    )


@pytest.fixture(scope="session")
def medium_ohlcv_data():
    """Medium dataset for integration tests."""
    return PerformanceOptimizedFixtures.generate_cached_ohlcv_data(
        start_date="2021-01-01", end_date="2023-12-31", num_assets=50, freq="D"
    )


@pytest.fixture(scope="session")
def large_ohlcv_data():
    """Large dataset for performance tests (cached to avoid regeneration)."""
    return PerformanceOptimizedFixtures.generate_cached_ohlcv_data(
        start_date="2020-01-01", end_date="2023-12-31", num_assets=100, freq="D"
    )


@pytest.fixture(scope="session")
def benchmark_data():
    """Cached benchmark data."""
    return PerformanceOptimizedFixtures.generate_cached_benchmark_data()


@pytest.fixture(scope="session")
def momentum_strategy_config():
    """Cached momentum strategy configuration."""
    return PerformanceOptimizedFixtures.generate_cached_strategy_config("momentum")


@pytest.fixture(scope="session")
def rsi_strategy_config():
    """Cached RSI strategy configuration."""
    return PerformanceOptimizedFixtures.generate_cached_strategy_config("rsi")


@pytest.fixture
def fast_test_data():
    """Very small dataset for the fastest possible tests."""
    return PerformanceOptimizedFixtures.generate_cached_ohlcv_data(
        start_date="2023-01-01", end_date="2023-03-31", num_assets=5, freq="ME"
    )


# Performance monitoring fixture
@pytest.fixture
def performance_monitor():
    """Monitor test execution time."""
    import time

    start_time = time.time()
    yield
    end_time = time.time()
    execution_time = end_time - start_time
    if execution_time > 1.0:  # Warn if test takes more than 1 second
        print(f"\n⚠️  Slow test detected: {execution_time:.2f}s")
