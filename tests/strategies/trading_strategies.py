"""
Hypothesis strategies for testing trading components.

This module provides reusable Hypothesis strategies for generating test data
for trading components, including weights, prices, and commissions.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple

from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from hypothesis.extra import pandas as hpd


@st.composite
def weights_and_prices(draw, min_days: int = 10, max_days: int = 100, 
                      min_assets: int = 2, max_assets: int = 10):
    """
    Generate weights and prices for trade tracking.
    
    Args:
        min_days: Minimum number of days
        max_days: Maximum number of days
        min_assets: Minimum number of assets
        max_assets: Maximum number of assets
    
    Returns:
        A tuple of (weights_df, prices_df, transaction_costs)
    """
    n_days = draw(st.integers(min_value=min_days, max_value=max_days))
    n_assets = draw(st.integers(min_value=min_assets, max_value=max_assets))
    
    # Generate dates
    start_date = draw(st.datetimes(min_value=datetime(2000, 1, 1), max_value=datetime(2020, 1, 1)))
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    index = pd.DatetimeIndex(dates)
    
    # Generate tickers
    tickers = [f"ASSET{i}" for i in range(n_assets)]
    
    # Generate weights
    weights_array = draw(
        hnp.arrays(
            dtype=float,
            shape=(n_days, n_assets),
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        )
    )
    
    # Normalize weights to sum to 1.0 for each day
    row_sums = weights_array.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0
    weights_array = weights_array / row_sums
    
    # Generate prices
    prices_array = draw(
        hnp.arrays(
            dtype=float,
            shape=(n_days, n_assets),
            elements=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        )
    )
    
    # Generate transaction costs
    transaction_costs = draw(st.floats(min_value=0.0001, max_value=0.01, allow_nan=False, allow_infinity=False))
    
    # Create DataFrames
    weights_df = pd.DataFrame(weights_array, index=index, columns=tickers)
    prices_df = pd.DataFrame(prices_array, index=index, columns=tickers)
    
    # Create transaction costs Series
    transaction_costs_series = pd.Series(transaction_costs, index=index)
    
    return weights_df, prices_df, transaction_costs_series


@st.composite
def trade_tracking_inputs(draw):
    """
    Generate inputs for trade tracking.
    
    Returns:
        A tuple of (weights_daily, price_data_daily_ohlc, transaction_costs, portfolio_value)
    """
    weights_df, prices_df, transaction_costs = draw(weights_and_prices())
    
    # Create OHLC price data
    price_data_daily_ohlc = pd.DataFrame(index=prices_df.index)
    for ticker in prices_df.columns:
        # Create MultiIndex columns
        for field in ["Open", "High", "Low", "Close", "Volume"]:
            if field == "Close":
                price_data_daily_ohlc[(ticker, field)] = prices_df[ticker]
            elif field == "Open":
                price_data_daily_ohlc[(ticker, field)] = prices_df[ticker] * draw(
                    st.floats(min_value=0.98, max_value=1.02, allow_nan=False, allow_infinity=False)
                )
            elif field == "High":
                price_data_daily_ohlc[(ticker, field)] = prices_df[ticker] * draw(
                    st.floats(min_value=1.0, max_value=1.05, allow_nan=False, allow_infinity=False)
                )
            elif field == "Low":
                price_data_daily_ohlc[(ticker, field)] = prices_df[ticker] * draw(
                    st.floats(min_value=0.95, max_value=1.0, allow_nan=False, allow_infinity=False)
                )
            else:  # Volume
                price_data_daily_ohlc[(ticker, field)] = draw(
                    st.integers(min_value=1000, max_value=10000000)
                )
    
    # Set MultiIndex columns
    price_data_daily_ohlc.columns = pd.MultiIndex.from_tuples(
        price_data_daily_ohlc.columns, names=["Ticker", "Field"]
    )
    
    # Generate portfolio value
    portfolio_value = draw(st.floats(min_value=10000.0, max_value=1000000.0, allow_nan=False, allow_infinity=False))
    
    return weights_df, price_data_daily_ohlc, transaction_costs, portfolio_value


@st.composite
def vectorized_tracking_inputs(draw):
    """
    Generate inputs for vectorized trade tracking.
    
    Returns:
        A tuple of (weights_array, prices_array, transaction_costs_array, portfolio_value)
    """
    n_days = draw(st.integers(min_value=10, max_value=100))
    n_assets = draw(st.integers(min_value=2, max_value=10))
    
    # Generate weights
    weights_array = draw(
        hnp.arrays(
            dtype=float,
            shape=(n_days, n_assets),
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        )
    )
    
    # Normalize weights to sum to 1.0 for each day
    row_sums = weights_array.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0
    weights_array = weights_array / row_sums
    
    # Generate prices
    prices_array = draw(
        hnp.arrays(
            dtype=float,
            shape=(n_days, n_assets),
            elements=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        )
    )
    
    # Generate transaction costs
    transaction_costs_array = draw(
        hnp.arrays(
            dtype=float,
            shape=(n_days, n_assets),
            elements=st.floats(min_value=0.0001, max_value=0.01, allow_nan=False, allow_infinity=False),
        )
    )
    
    # Generate portfolio value
    portfolio_value = draw(st.floats(min_value=10000.0, max_value=1000000.0, allow_nan=False, allow_infinity=False))
    
    return weights_array, prices_array, transaction_costs_array, portfolio_value
