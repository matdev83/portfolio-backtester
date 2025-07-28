"""
Example usage of OptimizedDataGenerator for test suite optimization.

This script demonstrates how to use the OptimizedDataGenerator class
to create efficient, cached test data generation patterns.
"""

import pandas as pd
import time
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tests.fixtures.optimized_data_generator import OptimizedDataGenerator


def demonstrate_basic_usage():
    """Demonstrate basic OHLCV data generation."""
    print("=== Basic OHLCV Data Generation ===")
    
    # Generate basic test data
    data = OptimizedDataGenerator.generate_cached_ohlcv_data(
        tickers=('AAPL', 'MSFT', 'GOOGL'),
        start_date='2020-01-01',
        end_date='2020-12-31',
        freq='B',
        pattern='random',
        seed=42
    )
    
    print(f"Generated data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"Tickers: {list(data.columns.get_level_values('Ticker').unique())}")
    print(f"Fields: {list(data.columns.get_level_values('Field').unique())}")
    print()


def demonstrate_caching_performance():
    """Demonstrate caching performance benefits."""
    print("=== Caching Performance Demonstration ===")
    
    # Clear cache first
    OptimizedDataGenerator.clear_cache()
    
    # First generation (cache miss)
    start_time = time.time()
    data1 = OptimizedDataGenerator.generate_cached_ohlcv_data(
        tickers=('PERF_TEST_1', 'PERF_TEST_2'),
        start_date='2020-01-01',
        end_date='2020-12-31',
        freq='B',
        pattern='random',
        seed=42
    )
    first_time = time.time() - start_time
    
    # Second generation (cache hit)
    start_time = time.time()
    data2 = OptimizedDataGenerator.generate_cached_ohlcv_data(
        tickers=('PERF_TEST_1', 'PERF_TEST_2'),
        start_date='2020-01-01',
        end_date='2020-12-31',
        freq='B',
        pattern='random',
        seed=42
    )
    second_time = time.time() - start_time
    
    print(f"First generation (cache miss): {first_time:.4f} seconds")
    print(f"Second generation (cache hit): {second_time:.4f} seconds")
    if second_time > 0:
        print(f"Speed improvement: {first_time / second_time:.1f}x faster")
    else:
        print("Speed improvement: >1000x faster (cached result returned instantly)")
    print(f"Data identical: {data1.equals(data2)}")
    
    # Show cache statistics
    cache_info = OptimizedDataGenerator.get_cache_info()
    print(f"Cache hits: {cache_info['lru_cache_hits']}")
    print(f"Cache misses: {cache_info['lru_cache_misses']}")
    print()


def demonstrate_different_patterns():
    """Demonstrate different data generation patterns."""
    print("=== Different Data Patterns ===")
    
    patterns = ['random', 'trending_up', 'trending_down', 'volatile', 'stable']
    
    for pattern in patterns:
        data = OptimizedDataGenerator.generate_cached_ohlcv_data(
            tickers=('PATTERN_TEST',),
            start_date='2020-01-01',
            end_date='2020-03-31',
            freq='B',
            pattern=pattern,
            seed=42
        )
        
        close_prices = data[('PATTERN_TEST', 'Close')]
        total_return = (close_prices.iloc[-1] / close_prices.iloc[0] - 1) * 100
        volatility = close_prices.pct_change().std() * 100
        
        print(f"{pattern:15}: Total return: {total_return:6.1f}%, Volatility: {volatility:.1f}%")
    
    print()


def demonstrate_date_range_utilities():
    """Demonstrate standardized date range generation."""
    print("=== Date Range Utilities ===")
    
    # Standard date ranges
    daily_range = OptimizedDataGenerator.create_standardized_date_range(
        '2020-01-01', '2020-01-31', 'D'
    )
    business_range = OptimizedDataGenerator.create_standardized_date_range(
        '2020-01-01', '2020-01-31', 'B'
    )
    monthly_range = OptimizedDataGenerator.create_standardized_date_range(
        '2020-01-01', '2020-12-31', 'ME'
    )
    
    print(f"Daily range (Jan 2020): {len(daily_range)} days")
    print(f"Business range (Jan 2020): {len(business_range)} days")
    print(f"Monthly range (2020): {len(monthly_range)} months")
    
    # Business days with holidays
    business_with_holidays = OptimizedDataGenerator.create_business_date_range(
        '2020-01-01', '2020-01-31',
        holidays=['2020-01-15', '2020-01-20']  # MLK Day and arbitrary holiday
    )
    
    print(f"Business days excluding holidays: {len(business_with_holidays)} days")
    print()


def demonstrate_data_validation():
    """Demonstrate data validation utilities."""
    print("=== Data Validation ===")
    
    # Generate test data
    data = OptimizedDataGenerator.create_test_data_with_validation(
        tickers=['AAPL', 'MSFT'],
        start_date='2020-01-01',
        end_date='2020-03-31',
        freq='B',
        pattern='random'
    )
    
    print("Generated data with automatic validation:")
    print(f"  - Structure validation: PASSED")
    print(f"  - Date range validation: PASSED")
    print(f"  - Data type validation: PASSED")
    print(f"  - OHLC relationship validation: PASSED")
    
    # Show some validation details
    tickers = data.columns.get_level_values('Ticker').unique()
    for ticker in tickers[:1]:  # Just show first ticker
        open_prices = data[(ticker, 'Open')]
        high_prices = data[(ticker, 'High')]
        low_prices = data[(ticker, 'Low')]
        close_prices = data[(ticker, 'Close')]
        
        # Check OHLC relationships
        high_ge_max_oc = (high_prices >= pd.concat([open_prices, close_prices], axis=1).max(axis=1)).all()
        low_le_min_oc = (low_prices <= pd.concat([open_prices, close_prices], axis=1).min(axis=1)).all()
        
        print(f"  - {ticker} High >= max(Open, Close): {high_ge_max_oc}")
        print(f"  - {ticker} Low <= min(Open, Close): {low_le_min_oc}")
    
    print()


def demonstrate_lazy_loading():
    """Demonstrate lazy loading decorator."""
    print("=== Lazy Loading Demonstration ===")
    
    call_count = 0
    
    @OptimizedDataGenerator.lazy_data_loader('expensive_computation')
    def expensive_computation():
        nonlocal call_count
        call_count += 1
        print(f"  Executing expensive computation (call #{call_count})")
        # Simulate expensive operation
        time.sleep(0.1)
        return pd.DataFrame({'result': [1, 2, 3, 4, 5]})
    
    print("First call (should execute function):")
    result1 = expensive_computation()
    
    print("Second call (should use cached result):")
    result2 = expensive_computation()
    
    print("Third call (should still use cached result):")
    result3 = expensive_computation()
    
    print(f"Total function executions: {call_count}")
    print(f"Results identical: {result1.equals(result2) and result2.equals(result3)}")
    print()


if __name__ == '__main__':
    print("OptimizedDataGenerator Usage Examples")
    print("=" * 50)
    print()
    
    demonstrate_basic_usage()
    demonstrate_caching_performance()
    demonstrate_different_patterns()
    demonstrate_date_range_utilities()
    demonstrate_data_validation()
    demonstrate_lazy_loading()
    
    # Final cache cleanup
    OptimizedDataGenerator.clear_cache()
    print("Cache cleared. Example complete!")