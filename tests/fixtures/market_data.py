"""
Market data fixtures for generating standardized OHLCV test data.

This module provides the MarketDataFixture class with methods for generating
various market data patterns used across different test scenarios.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from functools import lru_cache


class MarketDataFixture:
    """
    Fixture class for generating standardized market data for tests.
    
    Provides methods to create OHLCV data with different patterns:
    - Basic: Simple random walk data
    - Trending: Data with clear upward/downward trends
    - Volatile: High volatility data
    - Stable: Low volatility, mean-reverting data
    """
    
    @staticmethod
    @lru_cache(maxsize=10)
    def create_basic_data(
        tickers: tuple = ('AAPL', 'MSFT', 'GOOGL'),
        start_date: str = '2020-01-01',
        end_date: str = '2023-12-31',
        freq: str = 'B',
        base_price: float = 100.0,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Create basic OHLCV data with random walk patterns.
        
        Args:
            tickers: Tuple of ticker symbols
            start_date: Start date for data generation
            end_date: End date for data generation
            freq: Frequency for date range ('B' for business days, 'D' for daily)
            base_price: Starting price for all assets
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with MultiIndex columns (Ticker, Field) and datetime index
        """
        np.random.seed(seed)
        
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        fields = ['Open', 'High', 'Low', 'Close', 'Volume']
        columns = pd.MultiIndex.from_product([tickers, fields], names=['Ticker', 'Field'])
        
        data = {}
        
        for ticker in tickers:
            # Generate price series with random walk
            returns = np.random.normal(0.0005, 0.02, len(date_range))  # ~0.05% daily return, 2% volatility
            prices = [base_price]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            # Create OHLCV data
            for i, date in enumerate(date_range):
                close_price = prices[i]
                
                # Generate intraday price movements
                open_price = close_price * (1 + np.random.normal(0, 0.005))
                high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
                low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
                volume = int(1000000 + np.random.normal(0, 200000))
                
                data[(ticker, 'Open')] = data.get((ticker, 'Open'), []) + [open_price]
                data[(ticker, 'High')] = data.get((ticker, 'High'), []) + [high_price]
                data[(ticker, 'Low')] = data.get((ticker, 'Low'), []) + [low_price]
                data[(ticker, 'Close')] = data.get((ticker, 'Close'), []) + [close_price]
                data[(ticker, 'Volume')] = data.get((ticker, 'Volume'), []) + [volume]
        
        return pd.DataFrame(data, index=date_range, columns=columns)
    
    @staticmethod
    @lru_cache(maxsize=10)
    def create_trending_data(
        tickers: tuple = ('TREND_UP', 'TREND_DOWN', 'TREND_MILD'),
        start_date: str = '2020-01-01',
        end_date: str = '2023-12-31',
        freq: str = 'B',
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Create OHLCV data with clear trending patterns.
        
        Args:
            tickers: Tuple of ticker symbols with trend characteristics
            start_date: Start date for data generation
            end_date: End date for data generation
            freq: Frequency for date range
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with trending price patterns
        """
        np.random.seed(seed)
        
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        fields = ['Open', 'High', 'Low', 'Close', 'Volume']
        columns = pd.MultiIndex.from_product([tickers, fields], names=['Ticker', 'Field'])
        
        data = {}
        
        for ticker in tickers:
            # Define trend characteristics
            if 'UP' in ticker:
                base_trend = np.linspace(80, 200, len(date_range))
                volatility = 0.015
            elif 'DOWN' in ticker:
                base_trend = np.linspace(150, 50, len(date_range))
                volatility = 0.02
            else:  # MILD trend
                base_trend = np.linspace(95, 115, len(date_range))
                volatility = 0.01
            
            # Add noise to the trend
            noise = np.random.normal(0, volatility, len(date_range))
            prices = base_trend * (1 + noise)
            
            # Create OHLCV data
            for i, date in enumerate(date_range):
                close_price = prices[i]
                
                open_price = close_price * (1 + np.random.normal(0, 0.003))
                high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.008)))
                low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.008)))
                volume = int(1500000 + np.random.normal(0, 300000))
                
                data[(ticker, 'Open')] = data.get((ticker, 'Open'), []) + [open_price]
                data[(ticker, 'High')] = data.get((ticker, 'High'), []) + [high_price]
                data[(ticker, 'Low')] = data.get((ticker, 'Low'), []) + [low_price]
                data[(ticker, 'Close')] = data.get((ticker, 'Close'), []) + [close_price]
                data[(ticker, 'Volume')] = data.get((ticker, 'Volume'), []) + [volume]
        
        return pd.DataFrame(data, index=date_range, columns=columns)
    
    @staticmethod
    @lru_cache(maxsize=10)
    def create_volatile_data(
        tickers: tuple = ('VOL_HIGH', 'VOL_EXTREME'),
        start_date: str = '2020-01-01',
        end_date: str = '2023-12-31',
        freq: str = 'B',
        base_price: float = 100.0,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Create OHLCV data with high volatility patterns.
        
        Args:
            tickers: Tuple of ticker symbols
            start_date: Start date for data generation
            end_date: End date for data generation
            freq: Frequency for date range
            base_price: Starting price for all assets
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with high volatility patterns
        """
        np.random.seed(seed)
        
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        fields = ['Open', 'High', 'Low', 'Close', 'Volume']
        columns = pd.MultiIndex.from_product([tickers, fields], names=['Ticker', 'Field'])
        
        data = {}
        
        for ticker in tickers:
            # High volatility returns
            if 'EXTREME' in ticker:
                volatility = 0.08  # 8% daily volatility
            else:
                volatility = 0.05  # 5% daily volatility
            
            returns = np.random.normal(0, volatility, len(date_range))
            prices = [base_price]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            # Create OHLCV data with wider intraday ranges
            for i, date in enumerate(date_range):
                close_price = prices[i]
                
                open_price = close_price * (1 + np.random.normal(0, 0.02))
                high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.03)))
                low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.03)))
                volume = int(2000000 + np.random.normal(0, 500000))  # Higher volume for volatile stocks
                
                data[(ticker, 'Open')] = data.get((ticker, 'Open'), []) + [open_price]
                data[(ticker, 'High')] = data.get((ticker, 'High'), []) + [high_price]
                data[(ticker, 'Low')] = data.get((ticker, 'Low'), []) + [low_price]
                data[(ticker, 'Close')] = data.get((ticker, 'Close'), []) + [close_price]
                data[(ticker, 'Volume')] = data.get((ticker, 'Volume'), []) + [volume]
        
        return pd.DataFrame(data, index=date_range, columns=columns)
    
    @staticmethod
    @lru_cache(maxsize=10)
    def create_stable_data(
        tickers: tuple = ('STABLE_A', 'STABLE_B'),
        start_date: str = '2020-01-01',
        end_date: str = '2023-12-31',
        freq: str = 'B',
        base_price: float = 100.0,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Create OHLCV data with stable, low volatility patterns.
        
        Args:
            tickers: Tuple of ticker symbols
            start_date: Start date for data generation
            end_date: End date for data generation
            freq: Frequency for date range
            base_price: Starting price for all assets
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with stable, low volatility patterns
        """
        np.random.seed(seed)
        
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        fields = ['Open', 'High', 'Low', 'Close', 'Volume']
        columns = pd.MultiIndex.from_product([tickers, fields], names=['Ticker', 'Field'])
        
        data = {}
        
        for ticker in tickers:
            # Low volatility with mean reversion
            volatility = 0.005  # 0.5% daily volatility
            mean_reversion_strength = 0.1
            
            prices = [base_price]
            
            for i in range(1, len(date_range)):
                # Mean reversion component
                deviation_from_mean = (prices[-1] - base_price) / base_price
                mean_reversion = -mean_reversion_strength * deviation_from_mean
                
                # Random component
                random_return = np.random.normal(0, volatility)
                
                # Combined return
                total_return = mean_reversion + random_return
                new_price = prices[-1] * (1 + total_return)
                prices.append(new_price)
            
            # Create OHLCV data with tight intraday ranges
            for i, date in enumerate(date_range):
                close_price = prices[i]
                
                open_price = close_price * (1 + np.random.normal(0, 0.001))
                high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.002)))
                low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.002)))
                volume = int(800000 + np.random.normal(0, 100000))  # Lower volume for stable stocks
                
                data[(ticker, 'Open')] = data.get((ticker, 'Open'), []) + [open_price]
                data[(ticker, 'High')] = data.get((ticker, 'High'), []) + [high_price]
                data[(ticker, 'Low')] = data.get((ticker, 'Low'), []) + [low_price]
                data[(ticker, 'Close')] = data.get((ticker, 'Close'), []) + [close_price]
                data[(ticker, 'Volume')] = data.get((ticker, 'Volume'), []) + [volume]
        
        return pd.DataFrame(data, index=date_range, columns=columns)
    
    @staticmethod
    @lru_cache(maxsize=10)
    def create_benchmark_data(
        ticker: str = 'SPY',
        start_date: str = '2020-01-01',
        end_date: str = '2023-12-31',
        freq: str = 'B',
        base_price: float = 300.0,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Create benchmark OHLCV data (typically SPY).
        
        Args:
            ticker: Benchmark ticker symbol
            start_date: Start date for data generation
            end_date: End date for data generation
            freq: Frequency for date range
            base_price: Starting price for benchmark
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with benchmark data in MultiIndex format
        """
        np.random.seed(seed)
        
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        fields = ['Open', 'High', 'Low', 'Close', 'Volume']
        columns = pd.MultiIndex.from_product([[ticker], fields], names=['Ticker', 'Field'])
        
        # Generate benchmark returns (slightly lower volatility than individual stocks)
        returns = np.random.normal(0.0003, 0.015, len(date_range))  # ~0.03% daily return, 1.5% volatility
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = {}
        
        for i, date in enumerate(date_range):
            close_price = prices[i]
            
            open_price = close_price * (1 + np.random.normal(0, 0.003))
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.008)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.008)))
            volume = int(50000000 + np.random.normal(0, 10000000))  # High volume for benchmark
            
            data[(ticker, 'Open')] = data.get((ticker, 'Open'), []) + [open_price]
            data[(ticker, 'High')] = data.get((ticker, 'High'), []) + [high_price]
            data[(ticker, 'Low')] = data.get((ticker, 'Low'), []) + [low_price]
            data[(ticker, 'Close')] = data.get((ticker, 'Close'), []) + [close_price]
            data[(ticker, 'Volume')] = data.get((ticker, 'Volume'), []) + [volume]
        
        return pd.DataFrame(data, index=date_range, columns=columns)
    
    @staticmethod
    def create_standardized_date_range(
        start: str,
        end: str,
        freq: str = 'ME'
    ) -> pd.DatetimeIndex:
        """
        Create standardized date ranges for consistent test patterns.
        
        Args:
            start: Start date string
            end: End date string
            freq: Frequency ('ME' for month end, 'D' for daily, 'B' for business days)
            
        Returns:
            DatetimeIndex with standardized frequency
        """
        return pd.date_range(start=start, end=end, freq=freq)
    
    @staticmethod
    def add_missing_data(
        data: pd.DataFrame,
        missing_dates: List[str],
        tickers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Add missing data (NaN values) to test data for robustness testing.
        
        Args:
            data: Original OHLCV DataFrame
            missing_dates: List of date strings where data should be missing
            tickers: List of tickers to affect (if None, affects all)
            
        Returns:
            DataFrame with missing data introduced
        """
        data_with_nans = data.copy()
        missing_dates_parsed = pd.to_datetime(missing_dates)
        
        if tickers is None:
            tickers = data.columns.get_level_values('Ticker').unique()
        
        for date in missing_dates_parsed:
            if date in data_with_nans.index:
                for ticker in tickers:
                    for field in ['Open', 'High', 'Low', 'Close']:
                        if (ticker, field) in data_with_nans.columns:
                            data_with_nans.loc[date, (ticker, field)] = np.nan
        
        return data_with_nans