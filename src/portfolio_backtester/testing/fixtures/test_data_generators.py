"""
Test data generators for the portfolio backtester.

This module contains reusable test data generation utilities that can be
used across different test files to ensure consistent test data.
"""

import pandas as pd
import numpy as np
from typing import List


def generate_sample_price_data(
    symbols: List[str],
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    initial_price: float = 100.0,
    volatility: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate sample price data for testing.

    Args:
        symbols: List of symbol names
        start_date: Start date for the data
        end_date: End date for the data
        initial_price: Starting price for all symbols
        volatility: Daily volatility for price movements
        seed: Random seed for reproducibility

    Returns:
        DataFrame with MultiIndex columns (symbol, field) and date index
    """
    np.random.seed(seed)

    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Generate price data for each symbol
    data = {}
    for symbol in symbols:
        # Generate random returns
        returns = np.random.normal(0, volatility, len(date_range))

        # Calculate cumulative prices
        prices = [initial_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        # Create OHLC data (simplified)
        high_prices = [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices]
        low_prices = [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]
        volumes = [np.random.randint(1000000, 10000000) for _ in prices]

        data[(symbol, "Open")] = prices
        data[(symbol, "High")] = high_prices
        data[(symbol, "Low")] = low_prices
        data[(symbol, "Close")] = prices
        data[(symbol, "Volume")] = [float(v) for v in volumes]

    # Create MultiIndex DataFrame
    df = pd.DataFrame(data, index=date_range)

    return df


def generate_simple_signals_data(
    symbols: List[str],
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    signal_probability: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate simple random signals data for testing.

    Args:
        symbols: List of symbol names
        start_date: Start date for the data
        end_date: End date for the data
        signal_probability: Probability of generating a signal on any given day
        seed: Random seed for reproducibility

    Returns:
        DataFrame with symbols as columns and dates as index, containing 0/1 signals
    """
    np.random.seed(seed)

    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Generate random signals
    signals = {}
    for symbol in symbols:
        signals[symbol] = np.random.choice(
            [0, 1], size=len(date_range), p=[1 - signal_probability, signal_probability]
        )

    return pd.DataFrame(signals, index=date_range)


def generate_benchmark_data(
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    initial_price: float = 100.0,
    annual_return: float = 0.08,
    volatility: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate benchmark data for testing.

    Args:
        start_date: Start date for the data
        end_date: End date for the data
        initial_price: Starting price
        annual_return: Expected annual return
        volatility: Annual volatility
        seed: Random seed for reproducibility

    Returns:
        DataFrame with benchmark price data
    """
    np.random.seed(seed)

    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Calculate daily parameters
    daily_return = annual_return / 252  # 252 trading days per year
    daily_vol = volatility / np.sqrt(252)

    # Generate returns
    returns = np.random.normal(daily_return, daily_vol, len(date_range))

    # Calculate cumulative prices
    prices = [initial_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    return pd.DataFrame({"Close": prices}, index=date_range)
