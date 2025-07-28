"""
Utility functions for handling price data extraction and normalization.

This module provides common utilities to eliminate code duplication across strategy files.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional


def extract_current_prices(
    price_data: pd.DataFrame,
    current_date: pd.Timestamp,
    universe_tickers: pd.Index,
    fill_value: float = np.nan
) -> pd.Series:
    """
    Extract current prices for assets, handling various data type scenarios.
    
    This utility function replaces the repetitive temp_prices pattern found across
    multiple strategy files (VAMS, Calmar, Sortino momentum strategies).
    
    Args:
        price_data: DataFrame containing price data
        current_date: The date for which to extract prices
        universe_tickers: Index of ticker symbols to extract prices for
        fill_value: Value to use for missing prices (default: np.nan)
        
    Returns:
        Series of prices indexed by ticker symbols
        
    Example:
        >>> prices = extract_current_prices(asset_prices_df, current_date, tickers)
        >>> # Replaces this repetitive pattern:
        >>> # temp_prices = asset_prices_df.loc[current_date]
        >>> # if isinstance(temp_prices, pd.DataFrame):
        >>> #     temp_prices = temp_prices.squeeze()
        >>> # if not isinstance(temp_prices, pd.Series):
        >>> #     temp_prices = pd.Series([temp_prices], index=[tickers[0]]) if not tickers.empty else pd.Series(dtype=float)
        >>> # current_prices = temp_prices.reindex(tickers).fillna(np.nan)
    """
    if current_date not in price_data.index:
        # Return series with fill_value for all tickers if date not found
        return pd.Series(fill_value, index=universe_tickers)
    
    # Extract prices for the current date
    temp_prices = price_data.loc[current_date]
    
    # Handle case where result is a DataFrame (squeeze to Series)
    if isinstance(temp_prices, pd.DataFrame):
        temp_prices = temp_prices.squeeze()
    
    # Handle case where result is a scalar (convert to Series)
    if not isinstance(temp_prices, pd.Series):
        if not universe_tickers.empty:
            # Create Series with single value for first ticker
            temp_prices = pd.Series([temp_prices], index=[universe_tickers[0]])
        else:
            # Create empty Series if no tickers
            temp_prices = pd.Series(dtype=float)
    
    # Reindex to match universe tickers and fill missing values
    return temp_prices.reindex(universe_tickers).fillna(fill_value)


def validate_price_data_sufficiency(
    price_data: pd.DataFrame,
    current_date: pd.Timestamp,
    min_required_periods: int
) -> tuple[bool, str]:
    """
    Validate that price data has sufficient history for strategy calculations.
    
    Args:
        price_data: DataFrame containing price data
        current_date: The current date for validation
        min_required_periods: Minimum number of periods required
        
    Returns:
        Tuple of (is_sufficient, reason_if_not)
    """
    if price_data.empty:
        return False, "Price data is empty"
    
    # Get data up to current date
    available_data = price_data[price_data.index <= current_date]
    
    if len(available_data) < min_required_periods:
        return False, f"Insufficient data: {len(available_data)} periods available, {min_required_periods} required"
    
    return True, "Sufficient data available"


def normalize_price_series_to_dataframe(
    price_series: Union[pd.Series, pd.DataFrame],
    target_columns: Optional[pd.Index] = None
) -> pd.DataFrame:
    """
    Normalize price data to DataFrame format with consistent structure.
    
    Args:
        price_series: Series or DataFrame to normalize
        target_columns: Optional target column index to reindex to
        
    Returns:
        DataFrame with normalized structure
    """
    # Convert Series to DataFrame
    if isinstance(price_series, pd.Series):
        price_df = price_series.to_frame()
    else:
        price_df = price_series.copy()
    
    # Reindex to target columns if provided
    if target_columns is not None:
        price_df = price_df.reindex(columns=target_columns)
    
    return price_df