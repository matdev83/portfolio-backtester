"""
Common Hypothesis strategies for property-based testing of the portfolio backtester.

This module provides reusable strategies for generating test data with appropriate
constraints for testing different components of the portfolio backtester.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional

from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from hypothesis.extra import pandas as hpd


@st.composite
def timestamps(draw, min_year: int = 2000, max_year: int = 2023, size: int = None):
    """
    Generate a list of pandas Timestamps.
    
    Args:
        min_year: Minimum year for timestamps
        max_year: Maximum year for timestamps
        size: Number of timestamps to generate (if None, generates 1)
    
    Returns:
        A list of pandas Timestamps or a single Timestamp if size=None
    """
    if size is None:
        dt = draw(
            st.datetimes(
                min_value=datetime(min_year, 1, 1),
                max_value=datetime(max_year, 12, 31),
            )
        )
        return pd.Timestamp(dt)
    
    dates = draw(
        st.lists(
            st.datetimes(
                min_value=datetime(min_year, 1, 1),
                max_value=datetime(max_year, 12, 31),
            ),
            min_size=size,
            max_size=size,
            unique=True,
        )
    )
    dates.sort()
    return [pd.Timestamp(dt) for dt in dates]


@st.composite
def frequencies(draw):
    """
    Generate valid pandas frequency strings.
    
    Returns:
        A valid pandas frequency string
    """
    return draw(
        st.sampled_from([
            # Daily and weekly
            "D", "B", "W", 
            "W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI",
            # Monthly
            "M", "ME", "BM", "MS",
            # Quarterly
            "Q", "QE", "QS", "BQ",
            # Annual
            "A", "Y", "YE", "YS",
        ])
    )


@st.composite
def price_series(
    draw,
    min_size: int = 10,
    max_size: int = 100,
    min_price: float = 1.0,
    max_price: float = 1000.0,
    allow_missing: bool = False,
):
    """
    Generate a pandas Series of prices.
    
    Args:
        min_size: Minimum number of prices
        max_size: Maximum number of prices
        min_price: Minimum price value
        max_price: Maximum price value
        allow_missing: Whether to allow NaN values
    
    Returns:
        A pandas Series of prices with DatetimeIndex
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    
    # Generate dates
    dates = draw(
        st.lists(
            st.datetimes(
                min_value=datetime(2000, 1, 1),
                max_value=datetime(2023, 12, 31),
            ),
            min_size=size,
            max_size=size,
            unique=True,
        )
    )
    dates.sort()
    index = pd.DatetimeIndex([pd.Timestamp(dt) for dt in dates])
    
    # Generate prices
    if allow_missing:
        values = draw(
            hnp.arrays(
                dtype=float,
                shape=size,
                elements=st.one_of(
                    st.floats(min_value=min_price, max_value=max_price, allow_nan=False, allow_infinity=False),
                    st.just(np.nan),
                ),
                fill=st.nothing(),
            )
        )
    else:
        values = draw(
            hnp.arrays(
                dtype=float,
                shape=size,
                elements=st.floats(min_value=min_price, max_value=max_price, allow_nan=False, allow_infinity=False),
                fill=st.nothing(),
            )
        )
    
    return pd.Series(values, index=index)


@st.composite
def return_series(
    draw,
    min_size: int = 10,
    max_size: int = 100,
    min_return: float = -0.2,
    max_return: float = 0.2,
    allow_missing: bool = False,
):
    """
    Generate a pandas Series of returns.
    
    Args:
        min_size: Minimum number of returns
        max_size: Maximum number of returns
        min_return: Minimum return value
        max_return: Maximum return value
        allow_missing: Whether to allow NaN values
    
    Returns:
        A pandas Series of returns with DatetimeIndex
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    
    # Generate dates
    dates = draw(
        st.lists(
            st.datetimes(
                min_value=datetime(2000, 1, 1),
                max_value=datetime(2023, 12, 31),
            ),
            min_size=size,
            max_size=size,
            unique=True,
        )
    )
    dates.sort()
    index = pd.DatetimeIndex([pd.Timestamp(dt) for dt in dates])
    
    # Generate returns
    if allow_missing:
        values = draw(
            hnp.arrays(
                dtype=float,
                shape=size,
                elements=st.one_of(
                    st.floats(min_value=min_return, max_value=max_return, allow_nan=False, allow_infinity=False),
                    st.just(np.nan),
                ),
                fill=st.nothing(),
            )
        )
    else:
        values = draw(
            hnp.arrays(
                dtype=float,
                shape=size,
                elements=st.floats(min_value=min_return, max_value=max_return, allow_nan=False, allow_infinity=False),
                fill=st.nothing(),
            )
        )
    
    return pd.Series(values, index=index)


@st.composite
def return_matrices(
    draw,
    min_rows: int = 10,
    max_rows: int = 100,
    min_cols: int = 2,
    max_cols: int = 10,
    min_return: float = -0.2,
    max_return: float = 0.2,
    ensure_nonzero_variance: bool = False,
):
    """
    Generate a 2D numpy array of returns.
    
    Args:
        min_rows: Minimum number of time periods
        max_rows: Maximum number of time periods
        min_cols: Minimum number of assets
        max_cols: Maximum number of assets
        min_return: Minimum return value
        max_return: Maximum return value
        ensure_nonzero_variance: If True, ensures each column has non-zero variance
    
    Returns:
        A 2D numpy array of returns with shape (time_periods, assets)
    """
    rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    cols = draw(st.integers(min_value=min_cols, max_value=max_cols))
    
    elements = st.floats(min_value=min_return, max_value=max_return, allow_nan=False, allow_infinity=False)
    
    if ensure_nonzero_variance:
        # Create a base matrix
        base = draw(hnp.arrays(dtype=float, shape=(rows, cols), elements=elements))
        
        # Ensure each column has non-zero variance by adding a spike
        result = base.copy()
        for col in range(cols):
            # Add a spike in the middle of the column
            mid_idx = rows // 2
            result[mid_idx, col] = max_return
            result[mid_idx + 1 if mid_idx + 1 < rows else mid_idx - 1, col] = min_return
        
        return result
    else:
        return draw(hnp.arrays(dtype=float, shape=(rows, cols), elements=elements))


@st.composite
def price_dataframes(
    draw,
    min_rows: int = 10,
    max_rows: int = 100,
    min_cols: int = 2,
    max_cols: int = 10,
    min_price: float = 10.0,
    max_price: float = 1000.0,
    include_ohlcv: bool = False,
):
    """
    Generate a pandas DataFrame of prices.
    
    Args:
        min_rows: Minimum number of time periods
        max_rows: Maximum number of time periods
        min_cols: Minimum number of assets
        max_cols: Maximum number of assets
        min_price: Minimum price value
        max_price: Maximum price value
        include_ohlcv: If True, includes OHLCV columns for each asset
    
    Returns:
        A pandas DataFrame of prices with DatetimeIndex
    """
    rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    
    if include_ohlcv:
        # For OHLCV data, we'll create a multi-index DataFrame
        n_assets = draw(st.integers(min_value=min_cols, max_value=max_cols))
        tickers = [f"ASSET{i}" for i in range(n_assets)]
        
        # Generate dates
        dates = draw(
            st.lists(
                st.datetimes(
                    min_value=datetime(2000, 1, 1),
                    max_value=datetime(2023, 12, 31),
                ),
                min_size=rows,
                max_size=rows,
                unique=True,
            )
        )
        dates.sort()
        index = pd.DatetimeIndex([pd.Timestamp(dt) for dt in dates])
        
        # Create multi-index columns
        cols = pd.MultiIndex.from_product(
            [tickers, ["Open", "High", "Low", "Close", "Volume"]],
            names=["Ticker", "Field"]
        )
        
        # Generate price data ensuring High >= Open >= Low and High >= Close >= Low
        df = pd.DataFrame(index=index, columns=cols)
        
        for ticker in tickers:
            # Base prices around a random starting point
            base_price = draw(st.floats(min_value=min_price, max_value=max_price))
            
            # Generate random daily changes
            daily_changes = draw(
                hnp.arrays(
                    dtype=float,
                    shape=rows,
                    elements=st.floats(min_value=-0.05, max_value=0.05, allow_nan=False, allow_infinity=False),
                )
            )
            
            # Calculate prices with cumulative changes
            prices = base_price * np.cumprod(1 + daily_changes)
            
            for i in range(rows):
                price = prices[i]
                
                # Generate OHLC ensuring proper relationships
                open_price = price * draw(st.floats(min_value=0.99, max_value=1.01))
                high_price = price * draw(st.floats(min_value=1.01, max_value=1.03))
                low_price = price * draw(st.floats(min_value=0.97, max_value=0.99))
                close_price = price * draw(st.floats(min_value=0.99, max_value=1.01))
                
                # Ensure proper OHLC relationships
                high_price = max(high_price, open_price, close_price)
                low_price = min(low_price, open_price, close_price)
                
                # Set values
                df.loc[index[i], (ticker, "Open")] = open_price
                df.loc[index[i], (ticker, "High")] = high_price
                df.loc[index[i], (ticker, "Low")] = low_price
                df.loc[index[i], (ticker, "Close")] = close_price
                df.loc[index[i], (ticker, "Volume")] = draw(st.integers(min_value=1000, max_value=1000000))
        
        return df
    else:
        # For simple price data, we'll create a regular DataFrame
        cols = draw(st.integers(min_value=min_cols, max_value=max_cols))
        tickers = [f"ASSET{i}" for i in range(cols)]
        
        # Generate dates
        dates = draw(
            st.lists(
                st.datetimes(
                    min_value=datetime(2000, 1, 1),
                    max_value=datetime(2023, 12, 31),
                ),
                min_size=rows,
                max_size=rows,
                unique=True,
            )
        )
        dates.sort()
        index = pd.DatetimeIndex([pd.Timestamp(dt) for dt in dates])
        
        # Generate price data
        data = {}
        for ticker in tickers:
            # Base prices around a random starting point
            base_price = draw(st.floats(min_value=min_price, max_value=max_price))
            
            # Generate random daily changes
            daily_changes = draw(
                hnp.arrays(
                    dtype=float,
                    shape=rows,
                    elements=st.floats(min_value=-0.05, max_value=0.05, allow_nan=False, allow_infinity=False),
                )
            )
            
            # Calculate prices with cumulative changes
            prices = base_price * np.cumprod(1 + daily_changes)
            data[ticker] = prices
        
        return pd.DataFrame(data, index=index)


@st.composite
def weights_and_leverage(
    draw,
    min_assets: int = 2,
    max_assets: int = 10,
    min_rows: int = 1,
    max_rows: int = 30,
    allow_negative: bool = False,
    min_leverage: float = 0.1,
    max_leverage: float = 3.0,
):
    """
    Generate a DataFrame of weights and a leverage factor.
    
    Args:
        min_assets: Minimum number of assets
        max_assets: Maximum number of assets
        min_rows: Minimum number of time periods
        max_rows: Maximum number of time periods
        allow_negative: Whether to allow negative weights (short positions)
        min_leverage: Minimum leverage value
        max_leverage: Maximum leverage value
    
    Returns:
        A tuple of (weights_df, leverage) where weights_df is a DataFrame of weights
        and leverage is a float
    """
    n_assets = draw(st.integers(min_value=min_assets, max_value=max_assets))
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    
    # Generate asset names
    assets = [f"ASSET{i}" for i in range(n_assets)]
    
    # Generate weights
    if allow_negative:
        elements = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    else:
        elements = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    
    weights_array = draw(hnp.arrays(dtype=float, shape=(n_rows, n_assets), elements=elements))
    weights_df = pd.DataFrame(weights_array, columns=assets)
    
    # Generate leverage
    leverage = draw(st.floats(min_value=min_leverage, max_value=max_leverage, allow_nan=False, allow_infinity=False))
    
    return weights_df, leverage
