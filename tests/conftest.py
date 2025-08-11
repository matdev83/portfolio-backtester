"""
Global test configuration and fixtures for the portfolio backtester test suite.

This module provides:
- Global fixtures for market data, strategy configurations, and timing scenarios
- Session-scoped fixtures for expensive data generation that can be reused across tests
- Pytest collection hooks for automatic test categorization based on file location
- Common test utilities and shared test patterns
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tests.fixtures.optimized_data_generator import OptimizedDataGenerator


# ============================================================================
# PYTEST CONFIGURATION AND HOOKS
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Register custom markers
    config.addinivalue_line("markers", "unit: marks tests as unit tests (fast, isolated)")
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (moderate speed)"
    )
    config.addinivalue_line("markers", "system: marks tests as system tests (slow, end-to-end)")
    config.addinivalue_line("markers", "fast: marks tests as fast-running tests")
    config.addinivalue_line("markers", "slow: marks tests as slow-running tests")
    config.addinivalue_line("markers", "network: marks tests that require network access")
    config.addinivalue_line("markers", "universe: marks tests related to universe/holdings data")
    config.addinivalue_line("markers", "strategy: marks tests related to strategy functionality")
    config.addinivalue_line("markers", "timing: marks tests related to timing framework")
    config.addinivalue_line(
        "markers", "data_sources: marks tests related to data source functionality"
    )
    config.addinivalue_line(
        "markers", "optimization: marks tests related to optimization functionality"
    )
    config.addinivalue_line(
        "markers", "monte_carlo: marks tests related to Monte Carlo functionality"
    )
    config.addinivalue_line("markers", "reporting: marks tests related to reporting functionality")
    config.addinivalue_line(
        "markers", "api_stability: marks tests related to API stability validation"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically add markers to tests based on path, file name, and test name."""
    tests_root = Path(__file__).parent

    def add_dir_markers(parts: tuple[str, ...], item):
        if "unit" in parts:
            item.add_marker(pytest.mark.unit)
            item.add_marker(pytest.mark.fast)
        elif "integration" in parts:
            item.add_marker(pytest.mark.integration)
        elif "system" in parts:
            item.add_marker(pytest.mark.system)
            item.add_marker(pytest.mark.slow)

        dir_to_marker = {
            "strategies": pytest.mark.strategy,
            "timing": pytest.mark.timing,
            "data_sources": pytest.mark.data_sources,
            "optimization": pytest.mark.optimization,
            "monte_carlo": pytest.mark.monte_carlo,
            "reporting": pytest.mark.reporting,
            "universe": pytest.mark.universe,
        }
        for key, marker in dir_to_marker.items():
            if key in parts:
                item.add_marker(marker)

    def add_filename_markers(filename: str, item):
        name = filename.lower()
        if "integration" in name:
            item.add_marker(pytest.mark.integration)
        if "system" in name or "end_to_end" in name:
            item.add_marker(pytest.mark.system)
            item.add_marker(pytest.mark.slow)
        if "network" in name or "web" in name:
            item.add_marker(pytest.mark.network)

    def add_testname_markers(test_name: str, item):
        name = test_name.lower()
        if "slow" in name or "performance" in name:
            item.add_marker(pytest.mark.slow)
        if "fast" in name or "smoke" in name:
            item.add_marker(pytest.mark.fast)

    for item in items:
        try:
            test_file = Path(item.fspath).relative_to(tests_root)
        except ValueError:
            test_file = Path(item.fspath)
        parts = test_file.parts

        add_dir_markers(parts, item)
        add_filename_markers(test_file.name, item)
        add_testname_markers(item.name, item)


# ============================================================================
# SESSION-SCOPED FIXTURES FOR EXPENSIVE DATA GENERATION
# ============================================================================


@pytest.fixture(scope="session")
def session_market_data():
    """
    Session-scoped fixture for expensive market data generation.

    Generates comprehensive market data that can be reused across all tests
    in a session to avoid repeated expensive data generation.
    """
    # Generate comprehensive market data for common test scenarios
    tickers = ("AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY", "QQQ", "TLT", "UVXY")

    data = {}

    # Generate different time periods and frequencies
    periods = {
        "daily_1year": ("2023-01-01", "2023-12-31", "B"),
        "daily_2years": ("2022-01-01", "2023-12-31", "B"),
        "monthly_5years": ("2019-01-01", "2023-12-31", "ME"),
        "weekly_1year": ("2023-01-01", "2023-12-31", "W"),
    }

    for period_name, (start, end, freq) in periods.items():
        data[period_name] = OptimizedDataGenerator.generate_cached_ohlcv_data(
            tickers=tickers, start_date=start, end_date=end, freq=freq, pattern="random", seed=42
        )

    return data


@pytest.fixture(scope="session")
def session_strategy_configs():
    """
    Session-scoped fixture for standard strategy configurations.

    Provides pre-configured strategy settings for common test scenarios.
    """
    return {
        "momentum_basic": {
            "lookback_period": 12,
            "rebalance_frequency": "ME",
            "top_n": 5,
            "min_weight": 0.05,
            "max_weight": 0.25,
        },
        "momentum_aggressive": {
            "lookback_period": 6,
            "rebalance_frequency": "ME",
            "top_n": 3,
            "min_weight": 0.10,
            "max_weight": 0.40,
        },
        "momentum_conservative": {
            "lookback_period": 24,
            "rebalance_frequency": "QE",
            "top_n": 10,
            "min_weight": 0.02,
            "max_weight": 0.15,
        },
        "calmar_momentum": {
            "lookback_period": 12,
            "rebalance_frequency": "ME",
            "top_n": 5,
            "risk_metric": "calmar_ratio",
            "min_weight": 0.05,
            "max_weight": 0.25,
        },
        "sortino_momentum": {
            "lookback_period": 12,
            "rebalance_frequency": "ME",
            "top_n": 5,
            "risk_metric": "sortino_ratio",
            "min_weight": 0.05,
            "max_weight": 0.25,
        },
        "uvxy_rsi": {
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "rebalance_frequency": "D",
            "position_size": 0.10,
        },
        "ema_roro": {
            "short_ema": 12,
            "long_ema": 26,
            "signal_ema": 9,
            "rebalance_frequency": "ME",
            "risk_on_assets": ["SPY", "QQQ"],
            "risk_off_assets": ["TLT"],
        },
    }


@pytest.fixture(scope="session")
def session_timing_scenarios():
    """
    Session-scoped fixture for timing framework test scenarios.

    Provides pre-configured timing scenarios for testing strategy-timing integration.
    """
    return {
        "time_based_monthly": {
            "timing_type": "time_based",
            "frequency": "ME",
            "day_of_month": -1,  # Last day of month
            "time_of_day": "16:00",
            "timezone": "US/Eastern",
        },
        "time_based_quarterly": {
            "timing_type": "time_based",
            "frequency": "QE",
            "day_of_quarter": -1,  # Last day of quarter
            "time_of_day": "16:00",
            "timezone": "US/Eastern",
        },
        "signal_based_momentum": {
            "timing_type": "signal_based",
            "signal_source": "momentum_strategy",
            "trigger_condition": "position_change",
            "min_time_between_signals": "1D",
        },
        "signal_based_volatility": {
            "timing_type": "signal_based",
            "signal_source": "volatility_indicator",
            "trigger_condition": "threshold_breach",
            "threshold": 0.20,
            "lookback_period": 20,
        },
        "hybrid_timing": {
            "timing_type": "hybrid",
            "primary_timing": "time_based",
            "primary_frequency": "ME",
            "secondary_timing": "signal_based",
            "secondary_trigger": "volatility_spike",
        },
    }


# ============================================================================
# FUNCTION-SCOPED FIXTURES FOR COMMON TEST DATA
# ============================================================================


@pytest.fixture
def market_data_basic(session_market_data):
    """Basic market data for standard tests."""
    return session_market_data["daily_1year"].copy()


@pytest.fixture
def market_data_extended(session_market_data):
    """Extended market data for longer-term tests."""
    return session_market_data["daily_2years"].copy()


@pytest.fixture
def market_data_monthly(session_market_data):
    """Monthly market data for frequency-specific tests."""
    return session_market_data["monthly_5years"].copy()


@pytest.fixture
def market_data_trending():
    """Trending market data for momentum strategy tests."""
    return OptimizedDataGenerator.generate_cached_ohlcv_data(
        tickers=("TREND_UP", "TREND_DOWN"),
        start_date="2023-01-01",
        end_date="2023-12-31",
        freq="B",
        pattern="trending_up",
        seed=123,
    )


@pytest.fixture
def market_data_volatile():
    """Volatile market data for volatility-based tests."""
    return OptimizedDataGenerator.generate_cached_ohlcv_data(
        tickers=("VOLATILE_1", "VOLATILE_2"),
        start_date="2023-01-01",
        end_date="2023-12-31",
        freq="B",
        pattern="volatile",
        seed=456,
    )


@pytest.fixture
def market_data_stable():
    """Stable market data for low-volatility tests."""
    return OptimizedDataGenerator.generate_cached_ohlcv_data(
        tickers=("STABLE_1", "STABLE_2"),
        start_date="2023-01-01",
        end_date="2023-12-31",
        freq="B",
        pattern="stable",
        seed=789,
    )


@pytest.fixture
def strategy_config_momentum(session_strategy_configs):
    """Simple momentum strategy configuration."""
    return session_strategy_configs["momentum_simple"].copy()


@pytest.fixture
def strategy_config_calmar(session_strategy_configs):
    """Calmar momentum strategy configuration."""
    return session_strategy_configs["calmar_momentum"].copy()


@pytest.fixture
def strategy_config_sortino(session_strategy_configs):
    """Sortino momentum strategy configuration."""
    return session_strategy_configs["sortino_momentum"].copy()


@pytest.fixture
def strategy_config_uvxy(session_strategy_configs):
    """UVXY RSI strategy configuration."""
    return session_strategy_configs["uvxy_rsi"].copy()


@pytest.fixture
def strategy_config_ema_roro(session_strategy_configs):
    """EMA RoRo strategy configuration."""
    return session_strategy_configs["ema_roro"].copy()


@pytest.fixture
def timing_scenario_monthly(session_timing_scenarios):
    """Monthly time-based timing scenario."""
    return session_timing_scenarios["time_based_monthly"].copy()


@pytest.fixture
def timing_scenario_signal_based(session_timing_scenarios):
    """Signal-based timing scenario."""
    return session_timing_scenarios["signal_based_momentum"].copy()


@pytest.fixture
def timing_scenario_hybrid(session_timing_scenarios):
    """Hybrid timing scenario."""
    return session_timing_scenarios["hybrid_timing"].copy()


# ============================================================================
# UTILITY FIXTURES
# ============================================================================


@pytest.fixture
def standard_date_ranges():
    """Standard date ranges for consistent testing."""
    return {
        "short_term": ("2023-10-01", "2023-12-31"),
        "medium_term": ("2023-01-01", "2023-12-31"),
        "long_term": ("2022-01-01", "2023-12-31"),
        "very_long_term": ("2019-01-01", "2023-12-31"),
    }


@pytest.fixture
def common_tickers():
    """Common ticker symbols for testing."""
    return {
        "large_cap": ["AAPL", "MSFT", "GOOGL", "AMZN"],
        "tech": ["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
        "etfs": ["SPY", "QQQ", "IWM", "TLT", "GLD"],
        "volatility": ["UVXY", "VXX", "SVXY"],
        "test_symbols": ["TEST_1", "TEST_2", "TEST_3"],
    }


@pytest.fixture
def benchmark_data():
    """Benchmark data for performance comparison."""
    return OptimizedDataGenerator.generate_cached_ohlcv_data(
        tickers=("SPY", "QQQ", "TLT"),
        start_date="2022-01-01",
        end_date="2023-12-31",
        freq="B",
        pattern="random",
        seed=999,
    )


@pytest.fixture
def test_universe():
    """Standard test universe configuration."""
    return {
        "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"],
        "start_date": "2022-01-01",
        "end_date": "2023-12-31",
        "benchmark": "SPY",
        "risk_free_rate": 0.02,
        "transaction_costs": 0.001,
    }


# ============================================================================
# PERFORMANCE AND VALIDATION FIXTURES
# ============================================================================


@pytest.fixture
def performance_thresholds():
    """Performance thresholds for test validation."""
    return {
        "max_test_runtime": 30.0,  # seconds
        "max_data_generation_time": 5.0,  # seconds
        "min_sharpe_ratio": -2.0,  # minimum acceptable Sharpe ratio
        "max_drawdown": 0.50,  # maximum acceptable drawdown (50%)
        "min_data_points": 10,  # minimum data points for valid test
    }


@pytest.fixture
def validation_rules():
    """Data validation rules for test consistency."""
    return {
        "required_ohlcv_fields": ["Open", "High", "Low", "Close", "Volume"],
        "min_price_value": 0.01,
        "max_price_change": 0.50,  # 50% max daily change
        "min_volume": 1000,
        "required_date_columns": ["Date"],
        "required_numeric_types": ["float64", "int64"],
    }


# ============================================================================
# CLEANUP AND TEARDOWN FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup_cache():
    """Automatically clean up cache after each test to prevent memory issues."""
    yield
    # Cleanup after test
    if hasattr(OptimizedDataGenerator, "clear_cache"):
        # Only clear the lazy cache, keep LRU cache for session reuse
        OptimizedDataGenerator._cache.clear()


@pytest.fixture(scope="session", autouse=True)
def session_cleanup():
    """Clean up session-level resources."""
    yield
    # Note: Don't clear cache here - let pytest_sessionfinish report stats first


# ============================================================================
# HELPER FUNCTIONS FOR TESTS
# ============================================================================


def create_test_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    freq: str = "B",
    pattern: str = "random",
    **kwargs,
) -> pd.DataFrame:
    """
    Helper function to create test data with validation.

    Args:
        tickers: List of ticker symbols
        start_date: Start date string
        end_date: End date string
        freq: Frequency string
        pattern: Data pattern
        **kwargs: Additional arguments

    Returns:
        Validated DataFrame with OHLCV data
    """
    return OptimizedDataGenerator.create_test_data_with_validation(
        tickers, start_date, end_date, freq, pattern, **kwargs
    )


def assert_valid_ohlcv_data(data: pd.DataFrame, tickers: Optional[List[str]] = None):
    """
    Helper function to assert OHLCV data validity.

    Args:
        data: DataFrame to validate
        tickers: Optional list of expected tickers
    """
    OptimizedDataGenerator.validate_ohlcv_data_structure(data)
    OptimizedDataGenerator.validate_data_types(data)

    if tickers:
        actual_tickers = set(data.columns.get_level_values("Ticker").unique())
        expected_tickers = set(tickers)
        assert (
            actual_tickers == expected_tickers
        ), f"Expected tickers {expected_tickers}, got {actual_tickers}"


def assert_strategy_signals_valid(
    signals: pd.DataFrame, expected_columns: Optional[List[str]] = None
):
    """
    Helper function to validate strategy signals.

    Args:
        signals: Strategy signals DataFrame
        expected_columns: Optional list of expected columns
    """
    assert isinstance(signals, pd.DataFrame), "Signals must be a DataFrame"
    assert not signals.empty, "Signals cannot be empty"
    assert isinstance(signals.index, pd.DatetimeIndex), "Signals must have DatetimeIndex"

    if expected_columns:
        missing_columns = set(expected_columns) - set(signals.columns)
        assert not missing_columns, f"Missing expected columns: {missing_columns}"


def assert_performance_metrics_valid(
    metrics: Dict[str, float], thresholds: Optional[Dict[str, float]] = None
):
    """
    Helper function to validate performance metrics.

    Args:
        metrics: Dictionary of performance metrics
        thresholds: Optional dictionary of threshold values
    """
    required_metrics = ["total_return", "sharpe_ratio", "max_drawdown", "volatility"]

    for metric in required_metrics:
        assert metric in metrics, f"Missing required metric: {metric}"
        assert isinstance(metrics[metric], (int, float)), f"Metric {metric} must be numeric"
        assert not pd.isna(metrics[metric]), f"Metric {metric} cannot be NaN"

    if thresholds:
        if "min_sharpe_ratio" in thresholds:
            assert (
                metrics["sharpe_ratio"] >= thresholds["min_sharpe_ratio"]
            ), f"Sharpe ratio {metrics['sharpe_ratio']} below threshold {thresholds['min_sharpe_ratio']}"

        if "max_drawdown" in thresholds:
            assert (
                abs(metrics["max_drawdown"]) <= thresholds["max_drawdown"]
            ), f"Max drawdown {abs(metrics['max_drawdown'])} exceeds threshold {thresholds['max_drawdown']}"


# ============================================================================
# PYTEST PLUGINS AND EXTENSIONS
# ============================================================================


def pytest_runtest_setup(item):
    """Setup hook that runs before each test."""
    # Add any pre-test setup logic here
    pass


def pytest_runtest_teardown(item, nextitem):
    """Teardown hook that runs after each test."""
    # Add any post-test cleanup logic here
    pass


def pytest_sessionstart(session):
    """Hook that runs at the start of the test session."""
    print("\n" + "=" * 60)
    print("Portfolio Backtester Test Suite")
    print("=" * 60)
    print(f"Session started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python version: {sys.version}")
    print(f"Pandas version: {pd.__version__}")
    print(f"NumPy version: {np.__version__}")
    print("=" * 60)


def pytest_sessionfinish(session, exitstatus):
    """Hook that runs at the end of the test session."""
    print("\n" + "=" * 60)
    print(f"Session finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Exit status: {exitstatus}")

    # Show cache statistics
    cache_info = OptimizedDataGenerator.get_cache_info()
    print("Cache performance:")
    print(f"  - LRU cache hits: {cache_info['lru_cache_hits']}")
    print(f"  - LRU cache misses: {cache_info['lru_cache_misses']}")
    print(f"  - Lazy cache entries: {cache_info['lazy_cache_size']}")

    if cache_info["lru_cache_hits"] + cache_info["lru_cache_misses"] > 0:
        hit_rate = cache_info["lru_cache_hits"] / (
            cache_info["lru_cache_hits"] + cache_info["lru_cache_misses"]
        )
        print(f"  - Cache hit rate: {hit_rate:.1%}")

    print("=" * 60)

    # Clean up cache after reporting statistics
    OptimizedDataGenerator.clear_cache()
