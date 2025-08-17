"""
Property-based tests for take profit handlers.

This module uses Hypothesis to test invariants and properties of the take profit handlers.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple

from hypothesis import given, settings, strategies as st, assume
from hypothesis.extra import numpy as hnp

from portfolio_backtester.risk_management.take_profit_handlers import (
    NoTakeProfit,
    AtrBasedTakeProfit,
)
from portfolio_backtester.risk_management.stop_loss_handlers import AtrBasedStopLoss
from portfolio_backtester.risk_management.atr_service import calculate_atr_fast


@st.composite
def ohlc_data_frames(draw, min_rows=30, max_rows=100, min_assets=1, max_assets=5):
    """
    Generate OHLCV data frames for testing.
    """
    rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_assets = draw(st.integers(min_value=min_assets, max_value=max_assets))
    
    # Generate dates
    start_year = draw(st.integers(min_value=2000, max_value=2020))
    start_month = draw(st.integers(min_value=1, max_value=12))
    start_day = draw(st.integers(min_value=1, max_value=28))  # Avoid month end issues
    start_date = datetime(start_year, start_month, start_day)
    dates = pd.date_range(start=start_date, periods=rows, freq='B')
    
    # Create a dictionary to hold data for each asset
    data_dict = {}
    
    for i in range(n_assets):
        asset_name = f"ASSET{i}"
        
        # Generate prices with some realistic properties
        base_price = draw(st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
        volatility = draw(st.floats(min_value=0.01, max_value=0.1, allow_nan=False, allow_infinity=False))
        
        # Generate price series
        prices = np.random.normal(0, volatility, rows).cumsum() + base_price
        prices = np.maximum(prices, 1.0)  # Ensure prices are positive
        
        # Generate OHLCV data
        opens = prices * draw(hnp.arrays(dtype=float, shape=rows, elements=st.floats(min_value=0.98, max_value=1.02)))
        highs = np.maximum(prices * draw(hnp.arrays(dtype=float, shape=rows, elements=st.floats(min_value=1.0, max_value=1.05))), opens)
        lows = np.minimum(prices * draw(hnp.arrays(dtype=float, shape=rows, elements=st.floats(min_value=0.95, max_value=1.0))), opens)
        closes = prices
        volumes = draw(hnp.arrays(dtype=float, shape=rows, elements=st.floats(min_value=1000, max_value=1000000)))
        
        # Create DataFrame for this asset
        asset_df = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': closes,
            'Volume': volumes
        }, index=dates)
        
        data_dict[asset_name] = asset_df
    
    return data_dict


@st.composite
def ohlc_data_with_positions(draw):
    """
    Generate OHLCV data frames with positions, entry prices, and current prices.
    """
    # Generate OHLCV data
    ohlc_dict = draw(ohlc_data_frames())
    
    # Get list of assets
    assets = list(ohlc_dict.keys())
    
    # Convert dictionary of DataFrames to MultiIndex DataFrame
    dfs = []
    for asset, df in ohlc_dict.items():
        # Add ticker level to columns
        df_copy = df.copy()
        df_copy.columns = pd.MultiIndex.from_product([[asset], df_copy.columns], names=["Ticker", "Attribute"])
        dfs.append(df_copy)
    
    # Concatenate all DataFrames
    if dfs:
        ohlc_data = pd.concat(dfs, axis=1)
    else:
        # Create empty DataFrame with correct structure if no assets
        ohlc_data = pd.DataFrame(columns=pd.MultiIndex.from_product([["DUMMY"], ["Open", "High", "Low", "Close", "Volume"]], names=["Ticker", "Attribute"]))
    
    # Generate weights (positions)
    weights_elements = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    weights_values = draw(hnp.arrays(dtype=float, shape=len(assets), elements=weights_elements))
    weights = pd.Series(weights_values, index=assets)
    
    # Generate entry prices
    entry_prices_values = []
    for asset in assets:
        # Use a price from the history as entry price
        asset_data = ohlc_dict[asset]
        entry_idx = draw(st.integers(min_value=0, max_value=len(asset_data) - 10))
        entry_prices_values.append(asset_data.iloc[entry_idx]['Close'])
    
    entry_prices = pd.Series(entry_prices_values, index=assets)
    
    # Select current date (near the end of the data)
    if assets:
        asset_data = ohlc_dict[assets[0]]
        current_date_idx = draw(st.integers(min_value=len(asset_data) - 10, max_value=len(asset_data) - 1))
        current_date = asset_data.index[current_date_idx]
        
        # Get current prices
        current_prices_values = []
        for asset in assets:
            current_prices_values.append(ohlc_dict[asset].loc[current_date, 'Close'])
        
        current_prices = pd.Series(current_prices_values, index=assets)
    else:
        # Handle empty case
        current_date = pd.Timestamp('2020-01-01')
        current_prices = pd.Series(dtype=float)
    
    return ohlc_data, weights, entry_prices, current_date, current_prices


@st.composite
def atr_take_profit_configs(draw):
    """
    Generate ATR-based take profit configurations.
    """
    take_profit_config = {
        "type": "AtrBasedTakeProfit",
        "atr_length": draw(st.integers(min_value=5, max_value=20)),
        "atr_multiple": draw(st.floats(min_value=0.5, max_value=3.0, allow_nan=False, allow_infinity=False))
    }
    
    return take_profit_config


@given(ohlc_data_with_positions())
@settings(deadline=None)
def test_no_take_profit_handler_properties(data):
    """Test properties of NoTakeProfit handler."""
    ohlc_data, weights, entry_prices, current_date, current_prices = data
    
    # Create NoTakeProfit handler
    handler = NoTakeProfit({}, {})
    
    # Calculate take profit levels
    take_profit_levels = handler.calculate_take_profit_levels(
        current_date=current_date,
        asset_ohlc_history=ohlc_data,
        current_weights=weights,
        entry_prices=entry_prices,
    )
    
    # Check that all take profit levels are NaN
    for asset in weights.index:
        assert pd.isna(take_profit_levels[asset]), f"Take profit level should be NaN for {asset}"
    
    # Apply take profit
    adjusted_weights = handler.apply_take_profit(
        current_date=current_date,
        current_asset_prices=current_prices,
        target_weights=weights,
        entry_prices=entry_prices,
        take_profit_levels=take_profit_levels,
    )
    
    # Check that all weights are unchanged
    for asset in weights.index:
        assert adjusted_weights[asset] == weights[asset], f"Weight should be unchanged for {asset}"


@given(ohlc_data_with_positions(), atr_take_profit_configs())
@settings(deadline=None)
def test_atr_based_take_profit_handler_properties(data, take_profit_config):
    """Test properties of AtrBasedTakeProfit handler."""
    ohlc_data, weights, entry_prices, current_date, current_prices = data
    
    # Ensure all Series have the same index
    common_index = weights.index
    weights = weights.copy()
    entry_prices = entry_prices.reindex(common_index)
    current_prices = current_prices.reindex(common_index)
    
    # Create AtrBasedTakeProfit handler
    handler = AtrBasedTakeProfit({}, take_profit_config)
    
    # Calculate ATR values first to check if they're valid
    atr_length = take_profit_config["atr_length"]
    atr_values = calculate_atr_fast(ohlc_data, current_date, atr_length)
    
    # Only proceed if we have valid ATR values
    if len(atr_values) > 0:
        # Calculate take profit levels
        take_profit_levels = handler.calculate_take_profit_levels(
            current_date=current_date,
            asset_ohlc_history=ohlc_data,
            current_weights=weights,
            entry_prices=entry_prices,
        )
        
        # Check that take profit levels are defined for assets with positions
        for asset in weights.index:
            # Skip if asset is not in ATR values
            if asset not in atr_values.index:
                continue
                
            # Check individual scalar values directly to avoid Series truth value error
            asset_weight = weights[asset]
            asset_entry_price = entry_prices[asset]
            asset_atr = atr_values.loc[asset]
            
            # Handle potential Series return from .loc
            if isinstance(asset_atr, pd.Series):
                if asset_atr.isna().all():
                    continue
                asset_atr = asset_atr.iloc[0]  # Take first value if Series
            
            # Skip if any required value is NaN
            if pd.isna(asset_weight) or pd.isna(asset_entry_price) or pd.isna(asset_atr):
                continue
                
            # Check for non-zero weight - must explicitly compare to zero rather than use as boolean
            if asset_weight == 0:
                continue
            
            # At this point we should have a valid take profit level
            asset_take_profit_level = take_profit_levels[asset]
            assert not pd.isna(asset_take_profit_level), f"Take profit level should be defined for {asset}"
            
            # For long positions, take profit level should be above entry price
            if asset_weight > 0:
                assert asset_take_profit_level > asset_entry_price, f"Take profit level for long position should be above entry price for {asset}"
            
            # For short positions, take profit level should be below entry price
            if asset_weight < 0:
                assert asset_take_profit_level < asset_entry_price, f"Take profit level for short position should be below entry price for {asset}"
        
        # Apply take profit
        adjusted_weights = handler.apply_take_profit(
            current_date=current_date,
            current_asset_prices=current_prices,
            target_weights=weights,
            entry_prices=entry_prices,
            take_profit_levels=take_profit_levels,
        )
        
        # Check that weights are adjusted correctly
        for asset in weights.index:
            # Get individual scalar values
            asset_weight = weights[asset]
            asset_entry_price = entry_prices[asset]
            if asset not in take_profit_levels.index:
                continue
            asset_take_profit_level = take_profit_levels[asset]
            asset_current_price = current_prices[asset] if asset in current_prices.index else None
            
            # Skip if any required value is missing
            if pd.isna(asset_weight) or pd.isna(asset_entry_price) or pd.isna(asset_take_profit_level) or pd.isna(asset_current_price):
                continue
                
            # Skip if zero weight
            if asset_weight == 0:
                continue
                
            # For long positions that hit take profit
            if asset_weight > 0 and asset_current_price >= asset_take_profit_level:
                assert adjusted_weights[asset] == 0, f"Long position that hit take profit should be closed for {asset}"
            
            # For short positions that hit take profit
            elif asset_weight < 0 and asset_current_price <= asset_take_profit_level:
                assert adjusted_weights[asset] == 0, f"Short position that hit take profit should be closed for {asset}"
            
            # For positions that don't hit take profit
            elif (asset_weight > 0 and asset_current_price < asset_take_profit_level) or (asset_weight < 0 and asset_current_price > asset_take_profit_level):
                assert adjusted_weights[asset] == asset_weight, f"Position that didn't hit take profit should be unchanged for {asset}"
            else:
                # Weights for assets without positions or take profit levels should be unchanged
                assert adjusted_weights[asset] == asset_weight, f"Weight should be unchanged for {asset}"


@given(ohlc_data_with_positions(), atr_take_profit_configs())
@settings(deadline=None)
def test_stop_loss_and_take_profit_relationship(data, take_profit_config):
    """Test the relationship between stop loss and take profit levels."""
    ohlc_data, weights, entry_prices, current_date, current_prices = data
    
    # Ensure all Series have the same index
    common_index = weights.index
    weights = weights.copy()
    entry_prices = entry_prices.reindex(common_index)
    current_prices = current_prices.reindex(common_index)
    
    # Create ATR-based handlers with the same parameters
    stop_loss_config = {
        "type": "AtrBasedStopLoss",
        "atr_length": take_profit_config["atr_length"],
        "atr_multiple": take_profit_config["atr_multiple"],
    }
    
    stop_loss_handler = AtrBasedStopLoss({}, stop_loss_config)
    take_profit_handler = AtrBasedTakeProfit({}, take_profit_config)
    
    # Calculate ATR values first to check if they're valid
    atr_length = take_profit_config["atr_length"]
    atr_values = calculate_atr_fast(ohlc_data, current_date, atr_length)
    
    # Only proceed if we have valid ATR values
    if len(atr_values) > 0:
        # Calculate levels
        stop_levels = stop_loss_handler.calculate_stop_levels(
            current_date=current_date,
            asset_ohlc_history=ohlc_data,
            current_weights=weights,
            entry_prices=entry_prices,
        )
        
        take_profit_levels = take_profit_handler.calculate_take_profit_levels(
            current_date=current_date,
            asset_ohlc_history=ohlc_data,
            current_weights=weights,
            entry_prices=entry_prices,
        )
        
        # Check the relationship between stop loss and take profit levels
        for asset in weights.index:
            # Skip if asset is not in both stop_levels and take_profit_levels
            if asset not in stop_levels.index or asset not in take_profit_levels.index:
                continue
                
            # Get individual scalar values
            asset_weight = weights[asset]
            asset_entry_price = entry_prices[asset]
            asset_stop_level = stop_levels[asset]
            asset_take_profit_level = take_profit_levels[asset]
            
            # Skip if any value is NaN or asset weight is zero
            if pd.isna(asset_weight) or pd.isna(asset_entry_price) or pd.isna(asset_stop_level) or pd.isna(asset_take_profit_level):
                continue
                
            # For long positions
            if asset_weight > 0:
                # For long positions, take profit should be above entry price and stop loss below
                assert (
                    asset_stop_level < asset_entry_price < asset_take_profit_level
                ), f"For long positions, stop_loss < entry_price < take_profit should hold for {asset}"
            
            # For short positions
            elif asset_weight < 0:
                # For short positions, take profit should be below entry price and stop loss above
                assert (
                    asset_take_profit_level < asset_entry_price < asset_stop_level
                ), f"For short positions, take_profit < entry_price < stop_loss should hold for {asset}"
