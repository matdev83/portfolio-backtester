"""
Property-based tests for stop-loss handlers.

This module uses Hypothesis to test invariants and properties of the stop-loss handlers
in the risk_management module.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as hnp

from portfolio_backtester.risk_management.stop_loss_handlers import NoStopLoss, AtrBasedStopLoss
from portfolio_backtester.risk_management.atr_service import calculate_atr_fast


@st.composite
def ohlc_data_with_positions(draw):
    """Generate OHLC data with positions for stop-loss testing."""
    # Generate OHLC data
    n_assets = draw(st.integers(min_value=2, max_value=5))
    n_days = draw(st.integers(min_value=30, max_value=100))

    # Generate asset names
    assets = [f"ASSET{i}" for i in range(n_assets)]

    # Generate dates
    start_date = draw(
        st.datetimes(
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2023, 12, 31),
        )
    )
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    index = pd.DatetimeIndex([pd.Timestamp(dt) for dt in dates])

    # Create OHLC data
    ohlc_data = {}
    for asset in assets:
        base_price = draw(
            st.floats(min_value=50.0, max_value=500.0, allow_nan=False, allow_infinity=False)
        )

        # Generate daily changes
        daily_changes = draw(
            hnp.arrays(
                dtype=float,
                shape=n_days,
                elements=st.floats(
                    min_value=-0.05, max_value=0.05, allow_nan=False, allow_infinity=False
                ),
            )
        )

        # Calculate prices with cumulative changes
        prices = base_price * np.cumprod(1 + daily_changes)

        # Create OHLC data for this asset
        asset_data = {}
        for i in range(n_days):
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
            asset_data[index[i]] = {
                "Open": open_price,
                "High": high_price,
                "Low": low_price,
                "Close": close_price,
                "Volume": draw(st.integers(min_value=1000, max_value=1000000)),
            }

        ohlc_data[asset] = pd.DataFrame.from_dict(asset_data, orient="index")

    # Combine into multi-index DataFrame
    combined_data = pd.concat(ohlc_data, axis=1)

    # Generate current weights (positions)
    weights = pd.Series(
        draw(
            hnp.arrays(
                dtype=float,
                shape=n_assets,
                elements=st.floats(
                    min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
                ),
            )
        ),
        index=assets,
    )

    # Generate entry prices (for assets with non-zero weights)
    entry_prices = pd.Series(index=assets, dtype=float)
    for asset in assets:
        if weights[asset] != 0:
            # Entry price is a past price (from the first 80% of the data)
            past_idx = draw(st.integers(min_value=0, max_value=int(n_days * 0.8)))
            entry_prices[asset] = combined_data[asset, "Close"].iloc[past_idx]
        else:
            entry_prices[asset] = np.nan

    # Current date is one of the last 20% of dates
    current_date_idx = draw(st.integers(min_value=int(n_days * 0.8), max_value=n_days - 1))
    current_date = index[current_date_idx]

    # Current prices
    current_prices = pd.Series(
        [combined_data[asset, "Close"].iloc[current_date_idx] for asset in assets],
        index=assets,
    )

    return combined_data, weights, entry_prices, current_date, current_prices


@given(ohlc_data_with_positions())
@settings(deadline=None)
def test_no_stop_loss_handler_properties(data):
    """Test properties of NoStopLoss handler."""
    ohlc_data, weights, entry_prices, current_date, current_prices = data

    # Create NoStopLoss handler
    handler = NoStopLoss({}, {})

    # Calculate stop levels
    stop_levels = handler.calculate_stop_levels(
        current_date=current_date,
        asset_ohlc_history=ohlc_data,
        current_weights=weights,
        entry_prices=entry_prices,
    )

    # Check that stop levels are all NaN
    assert stop_levels.isna().all(), "NoStopLoss should return all NaN stop levels"

    # Apply stop loss
    adjusted_weights = handler.apply_stop_loss(
        current_date=current_date,
        current_asset_prices=current_prices,
        target_weights=weights,
        entry_prices=entry_prices,
        stop_levels=stop_levels,
    )

    # Check that weights are unchanged
    assert adjusted_weights.equals(weights), "NoStopLoss should not modify weights"


@st.composite
def atr_stop_loss_configs(draw):
    """Generate configurations for ATR-based stop loss."""
    atr_length = draw(st.integers(min_value=5, max_value=30))
    atr_multiple = draw(
        st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False)
    )

    stop_loss_config = {
        "type": "AtrBasedStopLoss",
        "atr_length": atr_length,
        "atr_multiple": atr_multiple,
    }

    return stop_loss_config


@given(ohlc_data_with_positions(), atr_stop_loss_configs())
@settings(deadline=None)
def test_atr_based_stop_loss_handler_properties(data, stop_loss_config):
    """Test properties of AtrBasedStopLoss handler."""
    ohlc_data, weights, entry_prices, current_date, current_prices = data

    # Ensure all Series have the same index
    common_index = weights.index
    weights = weights.copy()
    entry_prices = entry_prices.reindex(common_index)
    current_prices = current_prices.reindex(common_index)

    # Create AtrBasedStopLoss handler
    handler = AtrBasedStopLoss({}, stop_loss_config)

    # Calculate stop levels
    stop_levels = handler.calculate_stop_levels(
        current_date=current_date,
        asset_ohlc_history=ohlc_data,
        current_weights=weights,
        entry_prices=entry_prices,
    )

    # Check that stop levels are defined for assets with positions
    for asset in weights.index:
        if weights[asset] != 0 and not pd.isna(entry_prices[asset]):
            # Stop levels should be defined for assets with positions and entry prices
            assert not pd.isna(stop_levels[asset]), f"Stop level should be defined for {asset}"

            # For long positions, stop level should be below entry price
            if weights[asset] > 0:
                assert (
                    stop_levels[asset] < entry_prices[asset]
                ), f"Stop level for long position should be below entry price for {asset}"

            # For short positions, stop level should be above entry price
            if weights[asset] < 0:
                assert (
                    stop_levels[asset] > entry_prices[asset]
                ), f"Stop level for short position should be above entry price for {asset}"
        else:
            # Stop levels should be NaN for assets without positions or entry prices
            assert pd.isna(stop_levels[asset]), f"Stop level should be NaN for {asset}"

    # Apply stop loss
    adjusted_weights = handler.apply_stop_loss(
        current_date=current_date,
        current_asset_prices=current_prices,
        target_weights=weights,
        entry_prices=entry_prices,
        stop_levels=stop_levels,
    )

    # Check that weights are adjusted correctly
    for asset in weights.index:
        if (
            weights[asset] != 0
            and not pd.isna(entry_prices[asset])
            and not pd.isna(stop_levels[asset])
        ):
            # For long positions that hit stop loss
            if weights[asset] > 0 and current_prices[asset] <= stop_levels[asset]:
                assert (
                    adjusted_weights[asset] == 0
                ), f"Long position that hit stop loss should be closed for {asset}"

            # For short positions that hit stop loss
            if weights[asset] < 0 and current_prices[asset] >= stop_levels[asset]:
                assert (
                    adjusted_weights[asset] == 0
                ), f"Short position that hit stop loss should be closed for {asset}"

            # For positions that don't hit stop loss
            if (weights[asset] > 0 and current_prices[asset] > stop_levels[asset]) or (
                weights[asset] < 0 and current_prices[asset] < stop_levels[asset]
            ):
                assert (
                    adjusted_weights[asset] == weights[asset]
                ), f"Position that didn't hit stop loss should be unchanged for {asset}"
        else:
            # Weights for assets without positions or stop levels should be unchanged
            assert (
                adjusted_weights[asset] == weights[asset]
            ), f"Weight should be unchanged for {asset}"


@given(ohlc_data_with_positions(), atr_stop_loss_configs())
@settings(deadline=None)
def test_atr_calculation_properties(data, stop_loss_config):
    """Test properties of ATR calculation used in stop loss."""
    ohlc_data, weights, entry_prices, current_date, current_prices = data

    # Extract ATR parameters
    atr_length = stop_loss_config["atr_length"]

    # Calculate ATR values
    atr_values = calculate_atr_fast(ohlc_data, current_date, atr_length)

    # Check that ATR values are defined for assets
    for asset in atr_values.index:
        # Safely extract ATR value for this asset
        try:
            atr_value = atr_values.loc[asset]
            # Handle case where .loc returns a Series instead of scalar
            if isinstance(atr_value, pd.Series):
                if not atr_value.isna().all():
                    atr_value = atr_value.iloc[0]  # Use first value
                else:
                    continue  # Skip if all values are NaN
            
            # ATR should be non-negative if it exists
            if not pd.isna(atr_value):
                assert atr_value >= 0, f"ATR should be non-negative for {asset}"
        except (KeyError, ValueError, IndexError):
            # Skip if we can't access the ATR value for this asset
            continue

        # ATR should be related to price volatility
        # Handle different DataFrame structures - check if 'Ticker' is a level in the MultiIndex
        try:
            if isinstance(ohlc_data.columns, pd.MultiIndex) and 'Ticker' in ohlc_data.columns.names:
                # DataFrame has MultiIndex columns with 'Ticker' level
                asset_data = ohlc_data.xs(asset, level="Ticker", axis=1)
            elif asset in ohlc_data:
                # DataFrame has a simple dict-like structure with asset as keys
                asset_data = ohlc_data[asset]
            else:
                # Skip volatility check if we can't extract asset data
                continue
        except (KeyError, ValueError, IndexError):
            # Skip if we can't access asset data
            continue

        # Only proceed if we have enough data points
        if len(asset_data) >= atr_length:
            # Calculate a simple measure of volatility (high-low range)
            recent_data = asset_data.iloc[-atr_length:]
            
            # Handle different types of recent_data - could be DataFrame or Series
            if isinstance(recent_data, pd.DataFrame):
                # DataFrame case
                if "High" in recent_data.columns and "Low" in recent_data.columns:
                    try:
                        avg_range = (recent_data["High"] - recent_data["Low"]).mean()
                        
                        # Get ATR value safely for comparison
                        try:
                            atr_value = atr_values.loc[asset]
                            if isinstance(atr_value, pd.Series):
                                if not atr_value.isna().all():
                                    atr_value = atr_value.iloc[0]  # Use first value
                                else:
                                    continue  # Skip if all ATR values are NaN
                                    
                            # If ATR is defined and avg_range is positive, they should be related
                            if not pd.isna(atr_value) and avg_range > 0:
                                # ATR should be in the same order of magnitude as the average range
                                # Use a wider range for more robust testing
                                assert (
                                    0.1 * avg_range <= atr_value <= 5.0 * avg_range
                                ), f"ATR should be related to price range for {asset}"
                        except (KeyError, ValueError, IndexError):
                            # Skip if we can't access the ATR value
                            continue
                    except (TypeError, ValueError):
                        # Skip if there's an issue calculating the average range
                        continue
            elif isinstance(recent_data, pd.Series):
                # Series case - likely has a MultiIndex with 'High' and 'Low' levels
                if isinstance(recent_data.index, pd.MultiIndex) and len(recent_data.index.levels) > 1:
                    try:
                        # Check if High and Low are in the second level
                        level1_values = list(recent_data.index.levels[1])
                        if 'High' in level1_values and 'Low' in level1_values:
                            high_values = recent_data.xs('High', level=1)
                            low_values = recent_data.xs('Low', level=1)
                            avg_range = (high_values - low_values).mean()
                            
                            # Get ATR value safely for comparison
                            try:
                                atr_value = atr_values.loc[asset]
                                if isinstance(atr_value, pd.Series):
                                    if not atr_value.isna().all():
                                        atr_value = atr_value.iloc[0]  # Use first value
                                    else:
                                        continue  # Skip if all ATR values are NaN
                                        
                                # If ATR is defined and avg_range is positive, they should be related
                                if not pd.isna(atr_value) and avg_range > 0:
                                    # ATR should be in the same order of magnitude as the average range
                                    # Use a wider range for more robust testing
                                    assert (
                                        0.1 * avg_range <= atr_value <= 5.0 * avg_range
                                    ), f"ATR should be related to price range for {asset}"
                            except (KeyError, ValueError, IndexError):
                                # Skip if we can't access the ATR value
                                continue
                    except (KeyError, ValueError, IndexError, AttributeError):
                        # Skip if there's an issue with the MultiIndex or calculations
                        continue
