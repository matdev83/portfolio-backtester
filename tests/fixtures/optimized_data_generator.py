"""
Optimized data generation utilities for test suite performance.

This module provides the OptimizedDataGenerator class with cached OHLCV data generation,
standardized date range methods, lazy loading mechanisms, and data validation utilities
to improve test suite performance and consistency.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from functools import lru_cache, wraps
import warnings


class OptimizedDataGenerator:
    """
    Optimized data generation utilities with caching and lazy loading.
    
    This class provides high-performance data generation methods with:
    - LRU caching for expensive OHLCV data generation
    - Standardized date range generation with consistent frequency patterns
    - Lazy loading mechanisms for expensive test data
    - Data validation utilities for consistent data structure formats
    """
    
    # Class-level cache for expensive computations
    _cache = {}
    
    @staticmethod
    @lru_cache(maxsize=50)
    def generate_cached_ohlcv_data(
        tickers: tuple,
        start_date: str,
        end_date: str,
        freq: str = 'B',
        pattern: str = 'random',
        base_price: float = 100.0,
        volatility: float = 0.02,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Generate cached OHLCV data with specified patterns.
        
        Uses LRU cache to avoid regenerating identical datasets, significantly
        improving test performance when the same data parameters are used.
        
        Args:
            tickers: Tuple of ticker symbols (must be tuple for hashing)
            start_date: Start date string (YYYY-MM-DD format)
            end_date: End date string (YYYY-MM-DD format)
            freq: Frequency for date range ('B', 'D', 'ME', etc.)
            pattern: Data pattern ('random', 'trending_up', 'trending_down', 'volatile', 'stable')
            base_price: Starting price for all assets
            volatility: Base volatility for price generation
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with MultiIndex columns (Ticker, Field) and datetime index
        """
        np.random.seed(seed)
        
        date_range = OptimizedDataGenerator.create_standardized_date_range(
            start_date, end_date, freq
        )
        
        if len(date_range) == 0:
            raise ValueError(f"Empty date range generated for {start_date} to {end_date} with freq {freq}")
        
        fields = ['Open', 'High', 'Low', 'Close', 'Volume']
        columns = pd.MultiIndex.from_product([tickers, fields], names=['Ticker', 'Field'])
        
        data = {}
        
        for ticker in tickers:
            prices = OptimizedDataGenerator._generate_price_series(
                len(date_range), pattern, base_price, volatility
            )
            
            # Generate OHLCV data efficiently
            ohlcv_data = OptimizedDataGenerator._create_ohlcv_from_prices(
                prices, pattern, volatility
            )
            
            # Populate data dictionary
            for field_idx, field in enumerate(fields):
                data[(ticker, field)] = ohlcv_data[:, field_idx]
        
        df = pd.DataFrame(data, index=date_range, columns=columns)
        
        # Validate the generated data
        OptimizedDataGenerator.validate_ohlcv_data_structure(df)
        
        return df
    
    @staticmethod
    def _generate_price_series(
        length: int,
        pattern: str,
        base_price: float,
        volatility: float
    ) -> np.ndarray:
        """
        Generate price series based on specified pattern.
        
        Args:
            length: Number of price points to generate
            pattern: Price pattern type
            base_price: Starting price
            volatility: Price volatility
            
        Returns:
            Array of prices
        """
        if pattern == 'random':
            returns = np.random.normal(0.0005, volatility, length)
        elif pattern == 'trending_up':
            trend = np.linspace(0, 0.5, length)  # 50% total return over period
            returns = trend / length + np.random.normal(0, volatility * 0.5, length)
        elif pattern == 'trending_down':
            trend = np.linspace(0, -0.3, length)  # -30% total return over period
            returns = trend / length + np.random.normal(0, volatility * 0.5, length)
        elif pattern == 'volatile':
            returns = np.random.normal(0, volatility * 2.5, length)
        elif pattern == 'stable':
            # Mean-reverting pattern
            returns = np.random.normal(0, volatility * 0.3, length)
            # Add mean reversion
            for i in range(1, length):
                if i > 10:  # Allow some history for mean reversion
                    recent_returns = returns[max(0, i-10):i]
                    cumulative_return = np.sum(recent_returns)
                    returns[i] -= 0.1 * cumulative_return  # Mean reversion strength
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
        
        # Convert returns to prices
        prices = np.zeros(length)
        prices[0] = base_price
        
        for i in range(1, length):
            prices[i] = prices[i-1] * (1 + returns[i])
        
        return prices
    
    @staticmethod
    def _create_ohlcv_from_prices(
        prices: np.ndarray,
        pattern: str,
        volatility: float
    ) -> np.ndarray:
        """
        Create OHLCV data from price series.
        
        Args:
            prices: Array of close prices
            pattern: Pattern type for intraday range adjustment
            volatility: Base volatility for intraday movements
            
        Returns:
            Array of shape (len(prices), 5) with OHLCV data
        """
        length = len(prices)
        ohlcv = np.zeros((length, 5))
        
        # Adjust intraday volatility based on pattern
        if pattern == 'volatile':
            intraday_vol = volatility * 1.5
        elif pattern == 'stable':
            intraday_vol = volatility * 0.3
        else:
            intraday_vol = volatility * 0.5
        
        for i in range(length):
            close_price = prices[i]
            
            # Generate OHLC with realistic relationships
            open_price = close_price * (1 + np.random.normal(0, intraday_vol * 0.5))
            
            # High and low based on open and close
            high_base = max(open_price, close_price)
            low_base = min(open_price, close_price)
            
            high_price = high_base * (1 + abs(np.random.normal(0, intraday_vol)))
            low_price = low_base * (1 - abs(np.random.normal(0, intraday_vol)))
            
            # Volume with some correlation to price movement
            price_change = abs(close_price - open_price) / open_price if open_price > 0 else 0
            base_volume = 1000000
            volume_multiplier = 1 + price_change * 2  # Higher volume on bigger moves
            volume = int(base_volume * volume_multiplier * (1 + np.random.normal(0, 0.3)))
            volume = max(volume, 100000)  # Minimum volume
            
            ohlcv[i] = [open_price, high_price, low_price, close_price, volume]
        
        return ohlcv
    
    @staticmethod
    def create_standardized_date_range(
        start: str,
        end: str,
        freq: str = 'ME'
    ) -> pd.DatetimeIndex:
        """
        Create standardized date ranges with consistent frequency patterns.
        
        Handles the transition from deprecated pandas frequency strings to
        current standards (e.g., 'M' -> 'ME').
        
        Args:
            start: Start date string (YYYY-MM-DD format)
            end: End date string (YYYY-MM-DD format)
            freq: Frequency string ('ME' for month end, 'D' for daily, 'B' for business days)
            
        Returns:
            DatetimeIndex with standardized frequency
        """
        # Handle deprecated frequency strings
        freq_mapping = {
            'M': 'ME',  # Month end
            'Q': 'QE',  # Quarter end
            'Y': 'YE',  # Year end
            'A': 'YE',  # Annual (year end)
        }
        
        standardized_freq = freq_mapping.get(freq, freq)
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                return pd.date_range(start=start, end=end, freq=standardized_freq)
        except Exception as e:
            raise ValueError(f"Failed to create date range from {start} to {end} with freq {standardized_freq}: {e}")
    
    @staticmethod
    def create_business_date_range(
        start: str,
        end: str,
        holidays: Optional[List[str]] = None
    ) -> pd.DatetimeIndex:
        """
        Create business day date ranges excluding weekends and optional holidays.
        
        Args:
            start: Start date string
            end: End date string
            holidays: Optional list of holiday date strings to exclude
            
        Returns:
            DatetimeIndex with business days only
        """
        bdate_range = pd.bdate_range(start=start, end=end)
        
        if holidays:
            holiday_dates = pd.to_datetime(holidays)
            bdate_range = bdate_range.difference(holiday_dates)
        
        return bdate_range
    
    @staticmethod
    def lazy_data_loader(cache_key: str):
        """
        Decorator for lazy loading of expensive test data.
        
        Args:
            cache_key: Unique key for caching the data
            
        Returns:
            Decorator function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Check if data is already cached
                if cache_key in OptimizedDataGenerator._cache:
                    return OptimizedDataGenerator._cache[cache_key]
                
                # Generate data and cache it
                data = func(*args, **kwargs)
                OptimizedDataGenerator._cache[cache_key] = data
                return data
            
            return wrapper
        return decorator
    
    @staticmethod
    def clear_cache():
        """Clear all cached data to free memory."""
        OptimizedDataGenerator._cache.clear()
        OptimizedDataGenerator.generate_cached_ohlcv_data.cache_clear()
    
    @staticmethod
    def get_cache_info() -> Dict[str, Any]:
        """
        Get information about cache usage.
        
        Returns:
            Dictionary with cache statistics
        """
        lru_info = OptimizedDataGenerator.generate_cached_ohlcv_data.cache_info()
        return {
            'lru_cache_hits': lru_info.hits,
            'lru_cache_misses': lru_info.misses,
            'lru_cache_size': lru_info.currsize,
            'lru_cache_maxsize': lru_info.maxsize,
            'lazy_cache_size': len(OptimizedDataGenerator._cache)
        }
    
    @staticmethod
    def validate_ohlcv_data_structure(data: pd.DataFrame) -> bool:
        """
        Validate OHLCV data structure for consistency across tests.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if valid, raises ValueError if invalid
            
        Raises:
            ValueError: If data structure is invalid
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        
        if data.empty:
            raise ValueError("Data cannot be empty")
        
        # Check for MultiIndex columns
        if not isinstance(data.columns, pd.MultiIndex):
            raise ValueError("Data must have MultiIndex columns (Ticker, Field)")
        
        # Check column level names
        expected_names = ['Ticker', 'Field']
        if list(data.columns.names) != expected_names:
            raise ValueError(f"Column names must be {expected_names}, got {list(data.columns.names)}")
        
        # Check for required fields
        fields = data.columns.get_level_values('Field').unique()
        required_fields = {'Open', 'High', 'Low', 'Close', 'Volume'}
        missing_fields = required_fields - set(fields)
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Check datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be DatetimeIndex")
        
        # Check for OHLC relationships
        tickers = data.columns.get_level_values('Ticker').unique()
        for ticker in tickers:
            if (ticker, 'Open') in data.columns and (ticker, 'High') in data.columns:
                open_prices = data[(ticker, 'Open')]
                high_prices = data[(ticker, 'High')]
                low_prices = data[(ticker, 'Low')]
                close_prices = data[(ticker, 'Close')]
                
                # Check that High >= max(Open, Close) and Low <= min(Open, Close)
                max_oc = np.maximum(open_prices, close_prices)
                min_oc = np.minimum(open_prices, close_prices)
                
                if not (high_prices >= max_oc).all():
                    warnings.warn(f"High prices should be >= max(Open, Close) for {ticker}")
                
                if not (low_prices <= min_oc).all():
                    warnings.warn(f"Low prices should be <= min(Open, Close) for {ticker}")
        
        return True
    
    @staticmethod
    def validate_date_range_consistency(
        data: pd.DataFrame,
        expected_start: str,
        expected_end: str,
        expected_freq: str
    ) -> bool:
        """
        Validate that data has consistent date range and frequency.
        
        Args:
            data: DataFrame to validate
            expected_start: Expected start date
            expected_end: Expected end date
            expected_freq: Expected frequency
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
        
        actual_start = data.index[0].strftime('%Y-%m-%d')
        actual_end = data.index[-1].strftime('%Y-%m-%d')
        
        expected_range = OptimizedDataGenerator.create_standardized_date_range(
            expected_start, expected_end, expected_freq
        )
        
        if len(data.index) != len(expected_range):
            raise ValueError(
                f"Date range length mismatch. Expected {len(expected_range)}, got {len(data.index)}"
            )
        
        # Check if dates match (allowing for some flexibility in business day calculations)
        if expected_freq in ['B', 'BM', 'BQ']:
            # For business frequencies, just check start and end dates are reasonable
            start_diff = abs((pd.to_datetime(expected_start) - data.index[0]).days)
            end_diff = abs((pd.to_datetime(expected_end) - data.index[-1]).days)
            
            if start_diff > 7 or end_diff > 7:  # Allow up to a week difference for business days
                raise ValueError(
                    f"Date range mismatch. Expected {expected_start} to {expected_end}, "
                    f"got {actual_start} to {actual_end}"
                )
        else:
            # For other frequencies, check exact match
            if not data.index.equals(expected_range):
                raise ValueError(
                    f"Date index mismatch. Expected range from {expected_start} to {expected_end} "
                    f"with freq {expected_freq}"
                )
        
        return True
    
    @staticmethod
    def validate_data_types(data: pd.DataFrame) -> bool:
        """
        Validate data types for OHLCV data.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        
        # Check that price fields are numeric
        price_fields = ['Open', 'High', 'Low', 'Close']
        tickers = data.columns.get_level_values('Ticker').unique()
        
        for ticker in tickers:
            for field in price_fields:
                if (ticker, field) in data.columns:
                    series = data[(ticker, field)]
                    if not pd.api.types.is_numeric_dtype(series):
                        raise ValueError(f"Price field {ticker}.{field} must be numeric")
                    
                    # Check for negative prices
                    if (series < 0).any():
                        raise ValueError(f"Price field {ticker}.{field} contains negative values")
            
            # Check volume field
            if (ticker, 'Volume') in data.columns:
                volume_series = data[(ticker, 'Volume')]
                if not pd.api.types.is_numeric_dtype(volume_series):
                    raise ValueError(f"Volume field {ticker}.Volume must be numeric")
                
                # Check for negative volume
                if (volume_series < 0).any():
                    raise ValueError(f"Volume field {ticker}.Volume contains negative values")
        
        return True
    
    @staticmethod
    def create_test_data_with_validation(
        tickers: List[str],
        start_date: str,
        end_date: str,
        freq: str = 'B',
        pattern: str = 'random',
        **kwargs
    ) -> pd.DataFrame:
        """
        Create test data with automatic validation.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date string
            end_date: End date string
            freq: Frequency string
            pattern: Data pattern
            **kwargs: Additional arguments for data generation
            
        Returns:
            Validated DataFrame with OHLCV data
        """
        # Convert list to tuple for caching
        tickers_tuple = tuple(tickers)
        
        # Generate data
        data = OptimizedDataGenerator.generate_cached_ohlcv_data(
            tickers_tuple, start_date, end_date, freq, pattern, **kwargs
        )
        
        # Validate structure
        OptimizedDataGenerator.validate_ohlcv_data_structure(data)
        OptimizedDataGenerator.validate_date_range_consistency(data, start_date, end_date, freq)
        OptimizedDataGenerator.validate_data_types(data)
        
        return data