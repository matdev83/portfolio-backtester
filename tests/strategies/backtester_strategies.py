"""
Hypothesis strategies for testing backtester logic components.

This module provides reusable Hypothesis strategies for generating test data
for backtester logic components, including price data, scenario configurations,
and signals.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple

from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from hypothesis.extra import pandas as hpd


@st.composite
def price_data_daily_ohlc(draw, min_assets: int = 2, max_assets: int = 10, 
                         min_days: int = 50, max_days: int = 200):
    """
    Generate daily OHLC price data for backtesting.
    
    Args:
        min_assets: Minimum number of assets
        max_assets: Maximum number of assets
        min_days: Minimum number of days
        max_days: Maximum number of days
    
    Returns:
        A DataFrame with MultiIndex columns (Ticker, Field)
    """
    n_assets = draw(st.integers(min_value=min_assets, max_value=max_assets))
    n_days = draw(st.integers(min_value=min_days, max_value=max_days))
    
    # Generate dates
    start_date = draw(st.datetimes(min_value=datetime(2000, 1, 1), max_value=datetime(2020, 1, 1)))
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    index = pd.DatetimeIndex(dates)
    
    # Generate tickers
    tickers = [f"ASSET{i}" for i in range(n_assets)]
    
    # Create MultiIndex columns
    fields = ["Open", "High", "Low", "Close", "Volume"]
    columns = pd.MultiIndex.from_product([tickers, fields], names=["Ticker", "Field"])
    
    # Create empty DataFrame
    data = pd.DataFrame(index=index, columns=columns)
    
    # Fill with realistic OHLC data
    for ticker in tickers:
        # Start with a base price
        base_price = draw(st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
        
        # Generate daily price changes
        daily_changes = draw(
            hnp.arrays(
                dtype=float,
                shape=n_days,
                elements=st.floats(min_value=-0.05, max_value=0.05, allow_nan=False, allow_infinity=False),
            )
        )
        
        # Calculate close prices using cumulative product
        close_prices = base_price * np.cumprod(1 + daily_changes)
        
        # Fill OHLC data
        for i, date in enumerate(index):
            close = close_prices[i]
            
            # Generate open, high, low based on close
            open_price = close * draw(st.floats(min_value=0.98, max_value=1.02, allow_nan=False, allow_infinity=False))
            high_price = max(close, open_price) * draw(st.floats(min_value=1.0, max_value=1.05, allow_nan=False, allow_infinity=False))
            low_price = min(close, open_price) * draw(st.floats(min_value=0.95, max_value=1.0, allow_nan=False, allow_infinity=False))
            
            # Assign values to DataFrame
            data.loc[date, (ticker, "Open")] = open_price
            data.loc[date, (ticker, "High")] = high_price
            data.loc[date, (ticker, "Low")] = low_price
            data.loc[date, (ticker, "Close")] = close
            data.loc[date, (ticker, "Volume")] = draw(st.integers(min_value=1000, max_value=10000000))
    
    return data


@st.composite
def price_data_monthly_closes(draw, price_data_daily=None):
    """
    Generate monthly close price data from daily data or create new data.
    
    Args:
        price_data_daily: Optional daily price data to resample
    
    Returns:
        A DataFrame with monthly close prices
    """
    if price_data_daily is not None:
        # Extract close prices from daily data
        if isinstance(price_data_daily.columns, pd.MultiIndex):
            close_prices = price_data_daily.xs("Close", level="Field", axis=1)
        else:
            close_prices = price_data_daily
        
        # Resample to month-end
        monthly_data = close_prices.resample("ME").last()
        return monthly_data
    
    # Generate new monthly data if daily data not provided
    n_assets = draw(st.integers(min_value=2, max_value=10))
    n_months = draw(st.integers(min_value=12, max_value=60))
    
    # Generate dates
    start_date = draw(st.datetimes(min_value=datetime(2000, 1, 1), max_value=datetime(2020, 1, 1)))
    dates = pd.date_range(start=start_date, periods=n_months, freq="ME")
    
    # Generate tickers
    tickers = [f"ASSET{i}" for i in range(n_assets)]
    
    # Create empty DataFrame
    data = pd.DataFrame(index=dates, columns=tickers)
    
    # Fill with realistic price data
    for ticker in tickers:
        # Start with a base price
        base_price = draw(st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
        
        # Generate monthly price changes
        monthly_changes = draw(
            hnp.arrays(
                dtype=float,
                shape=n_months,
                elements=st.floats(min_value=-0.15, max_value=0.15, allow_nan=False, allow_infinity=False),
            )
        )
        
        # Calculate prices using cumulative product
        prices = base_price * np.cumprod(1 + monthly_changes)
        
        # Assign to DataFrame
        data[ticker] = prices
    
    return data


@st.composite
def daily_returns(draw, price_data_daily=None):
    """
    Generate daily returns data from price data or create new data.
    
    Args:
        price_data_daily: Optional daily price data to calculate returns from
    
    Returns:
        A DataFrame with daily returns
    """
    if price_data_daily is not None:
        # Extract close prices from daily data
        if isinstance(price_data_daily.columns, pd.MultiIndex):
            close_prices = price_data_daily.xs("Close", level="Field", axis=1)
        else:
            close_prices = price_data_daily
        
        # Calculate returns
        returns = close_prices.pct_change().fillna(0)
        return returns
    
    # Generate new returns data if price data not provided
    n_assets = draw(st.integers(min_value=2, max_value=10))
    n_days = draw(st.integers(min_value=50, max_value=200))
    
    # Generate dates
    start_date = draw(st.datetimes(min_value=datetime(2000, 1, 1), max_value=datetime(2020, 1, 1)))
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    index = pd.DatetimeIndex(dates)
    
    # Generate tickers
    tickers = [f"ASSET{i}" for i in range(n_assets)]
    
    # Create empty DataFrame
    data = pd.DataFrame(index=index, columns=tickers)
    
    # Fill with realistic return data
    for ticker in tickers:
        # Generate daily returns
        mean_return = draw(st.floats(min_value=-0.001, max_value=0.001, allow_nan=False, allow_infinity=False))
        std_dev = draw(st.floats(min_value=0.01, max_value=0.03, allow_nan=False, allow_infinity=False))
        
        returns = draw(
            hnp.arrays(
                dtype=float,
                shape=n_days,
                elements=st.floats(min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False),
            )
        )
        
        # Add autocorrelation for realism
        for i in range(1, n_days):
            returns[i] = 0.1 * returns[i-1] + 0.9 * returns[i]
        
        # Scale to desired mean and std
        if np.std(returns) > 1e-10:
            returns = (returns - np.mean(returns)) / np.std(returns) * std_dev + mean_return
        else:
            returns = np.ones(n_days) * mean_return
        
        # Assign to DataFrame
        data[ticker] = returns
    
    return data


@st.composite
def scenario_configs(draw):
    """
    Generate scenario configurations for backtesting.
    
    Returns:
        A dictionary with scenario configuration
    """
    strategy_types = ["portfolio", "signal", "meta"]
    strategy_type = draw(st.sampled_from(strategy_types))
    
    strategy_names = {
        "portfolio": ["EqualWeight", "FixedWeight", "MinimumVariance"],
        "signal": ["MovingAverageCrossover", "RSI", "Momentum"],
        "meta": ["Ensemble", "Rotation", "Seasonal"]
    }
    
    strategy_name = draw(st.sampled_from(strategy_names[strategy_type]))
    
    # Generate base scenario config
    config = {
        "name": f"{strategy_name}_test",
        "strategy": f"{strategy_name}Strategy",
        "strategy_type": strategy_type,
        "universe": draw(st.lists(
            st.sampled_from(["AAPL", "MSFT", "GOOGL", "AMZN", "FB", "TSLA", "V", "JPM", "JNJ", "WMT"]),
            min_size=2,
            max_size=10,
            unique=True
        )),
        "benchmark": "SPY",
            "start_date": draw(st.datetimes(min_value=datetime(2000, 1, 1), max_value=datetime(2015, 1, 1))).strftime("%Y-%m-%d"),
    "end_date": draw(st.datetimes(min_value=datetime(2015, 1, 2), max_value=datetime(2020, 1, 1))).strftime("%Y-%m-%d"),
        "strategy_params": {}
    }
    
    # Add timing config
    timing_modes = ["time_based", "signal_based"]
    timing_mode = draw(st.sampled_from(timing_modes))
    
    if timing_mode == "time_based":
        config["timing_config"] = {
            "mode": "time_based",
            "rebalance_frequency": draw(st.sampled_from(["D", "W", "M", "Q", "Y"]))
        }
    else:
        config["timing_config"] = {
            "mode": "signal_based",
            "signal_source": draw(st.sampled_from(["price_change", "volatility", "rsi"])),
            "trigger_condition": draw(st.sampled_from(["above", "below", "cross_above", "cross_below"])),
            "threshold": draw(st.floats(min_value=0.01, max_value=0.1, allow_nan=False, allow_infinity=False))
        }
    
    # Add strategy-specific parameters
    if strategy_type == "portfolio":
        if strategy_name == "FixedWeight":
            config["strategy_params"]["weights"] = {
                ticker: draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
                for ticker in config["universe"]
            }
            # Normalize weights
            total = sum(config["strategy_params"]["weights"].values())
            if total > 0:
                for ticker in config["strategy_params"]["weights"]:
                    config["strategy_params"]["weights"][ticker] /= total
        elif strategy_name == "MinimumVariance":
            config["strategy_params"]["lookback_period"] = draw(st.integers(min_value=30, max_value=252))
    elif strategy_type == "signal":
        if strategy_name == "MovingAverageCrossover":
            config["strategy_params"]["fast_period"] = draw(st.integers(min_value=5, max_value=50))
            config["strategy_params"]["slow_period"] = draw(st.integers(min_value=51, max_value=200))
        elif strategy_name == "RSI":
            config["strategy_params"]["rsi_period"] = draw(st.integers(min_value=5, max_value=30))
            config["strategy_params"]["overbought"] = draw(st.integers(min_value=65, max_value=85))
            config["strategy_params"]["oversold"] = draw(st.integers(min_value=15, max_value=35))
        elif strategy_name == "Momentum":
            config["strategy_params"]["momentum_period"] = draw(st.integers(min_value=20, max_value=252))
    
    return config


@st.composite
def signal_dataframes(draw, universe=None, dates=None):
    """
    Generate signal DataFrames for backtesting.
    
    Args:
        universe: Optional list of tickers
        dates: Optional list of dates
    
    Returns:
        A DataFrame with signals (weights)
    """
    if universe is None:
        n_assets = draw(st.integers(min_value=2, max_value=10))
        universe = [f"ASSET{i}" for i in range(n_assets)]
    else:
        n_assets = len(universe)
    
    if dates is None:
        n_dates = draw(st.integers(min_value=5, max_value=50))
        start_date = draw(st.datetimes(min_value=datetime(2000, 1, 1), max_value=datetime(2020, 1, 1)))
        dates = pd.date_range(start=start_date, periods=n_dates, freq="ME")
    else:
        n_dates = len(dates)
    
    # Generate signal weights
    weights_array = draw(
        hnp.arrays(
            dtype=float,
            shape=(n_dates, n_assets),
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        )
    )
    
    # Normalize weights to sum to 1.0 for each date
    row_sums = weights_array.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0
    weights_array = weights_array / row_sums
    
    # Create DataFrame
    signals = pd.DataFrame(weights_array, index=dates, columns=universe)
    
    return signals
