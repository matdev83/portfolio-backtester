"""
Property-based tests for price data utilities.

This module uses Hypothesis to test invariants and properties of the price data utility
functions in the utils/price_data_utils.py module.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

from hypothesis import given, settings, strategies as st, assume
from hypothesis.extra import numpy as hnp

from portfolio_backtester.utils.price_data_utils import extract_current_prices

from tests.strategies import price_dataframes, timestamps


@st.composite
def price_data_with_date(draw):
    """Generate price data and a date for extract_current_prices."""
    # Generate price data
    price_data = draw(price_dataframes(min_rows=10, max_rows=50, min_cols=2, max_cols=10))
    
    # Select a date from the index or a date outside the index
    use_valid_date = draw(st.booleans())
    
    if use_valid_date and len(price_data.index) > 0:
        # Select a random date from the index
        date_idx = draw(st.integers(min_value=0, max_value=len(price_data.index) - 1))
        date = price_data.index[date_idx]
    else:
        # Generate a date outside the index
        date = draw(timestamps(min_year=1990, max_year=2030))
    
    # Get universe tickers
    universe_tickers = price_data.columns
    
    return price_data, date, universe_tickers


@given(price_data_with_date())
@settings(deadline=None)
def test_extract_current_prices_properties(data):
    """Test properties of extract_current_prices."""
    price_data, current_date, universe_tickers = data
    
    # Default fill value
    fill_value = np.nan
    
    # Extract current prices
    prices = extract_current_prices(price_data, current_date, universe_tickers, fill_value)
    
    # Check that prices have the correct index
    assert prices.index.equals(universe_tickers)
    
    # Check behavior when date is in the index
    if current_date in price_data.index:
        # Check that prices match the original data
        expected = price_data.loc[current_date]
        if isinstance(expected, pd.DataFrame):
            expected = expected.iloc[0]
        
        # Compare non-NaN values
        for ticker in universe_tickers:
            if ticker in expected.index and not pd.isna(expected[ticker]):
                assert prices[ticker] == expected[ticker]
    else:
        # If date not in index, all values should be fill_value
        if pd.isna(fill_value):
            assert prices.isna().all()
        else:
            assert (prices == fill_value).all()


@given(price_data_with_date(), st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False))
@settings(deadline=None)
def test_extract_current_prices_custom_fill(data, custom_fill):
    """Test that extract_current_prices respects custom fill values."""
    price_data, current_date, universe_tickers = data
    
    # Extract current prices with custom fill value
    prices = extract_current_prices(price_data, current_date, universe_tickers, custom_fill)
    
    # Check that prices have the correct index
    assert prices.index.equals(universe_tickers)
    
    # Check behavior when date is not in the index
    if current_date not in price_data.index:
        # All values should be the custom fill value
        assert (prices == custom_fill).all()


@st.composite
def ohlcv_data_with_date(draw):
    """Generate OHLCV price data and a date for extract_current_prices."""
    # Generate OHLCV price data
    price_data = draw(price_dataframes(min_rows=10, max_rows=50, min_cols=2, max_cols=5, include_ohlcv=True))
    
    # Select a date from the index or a date outside the index
    use_valid_date = draw(st.booleans())
    
    if use_valid_date and len(price_data.index) > 0:
        # Select a random date from the index
        date_idx = draw(st.integers(min_value=0, max_value=len(price_data.index) - 1))
        date = price_data.index[date_idx]
    else:
        # Generate a date outside the index
        date = draw(timestamps(min_year=1990, max_year=2030))
    
    # Get universe tickers and select one field (e.g., 'Close')
    tickers = price_data.columns.levels[0]
    field = draw(st.sampled_from(['Open', 'High', 'Low', 'Close', 'Volume']))
    
    # Create a view of just that field
    field_data = price_data.xs(field, level=1, axis=1)
    
    return field_data, date, tickers


@given(ohlcv_data_with_date())
@settings(deadline=None)
def test_extract_current_prices_with_ohlcv_data(data):
    """Test extract_current_prices with OHLCV data."""
    price_data, current_date, universe_tickers = data
    
    # Default fill value
    fill_value = np.nan
    
    # Extract current prices
    prices = extract_current_prices(price_data, current_date, universe_tickers, fill_value)
    
    # Check that prices have the correct index
    assert prices.index.equals(universe_tickers)
    
    # Check behavior when date is in the index
    if current_date in price_data.index:
        # Check that prices match the original data
        expected = price_data.loc[current_date]
        if isinstance(expected, pd.DataFrame):
            expected = expected.iloc[0]
        elif isinstance(expected, pd.Series):
            pass
        else:
            expected = pd.Series([expected], index=[universe_tickers[0]])
        
        # Compare non-NaN values
        for ticker in universe_tickers:
            if ticker in expected.index and not pd.isna(expected[ticker]):
                assert prices[ticker] == expected[ticker]
    else:
        # If date not in index, all values should be fill_value
        if pd.isna(fill_value):
            assert prices.isna().all()
        else:
            assert (prices == fill_value).all()
