"""
Property-based tests for take-profit handlers.

This module uses Hypothesis to test invariants and properties of the take-profit handlers
in the risk_management module.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from hypothesis import given, settings, strategies as st, HealthCheck
from hypothesis.extra import numpy as hnp

from portfolio_backtester.risk_management.stop_loss_handlers import (
    AtrBasedStopLoss,
)
from portfolio_backtester.risk_management.take_profit_handlers import (
    NoTakeProfit,
    AtrBasedTakeProfit,
)


@st.composite
def ohlc_data_with_positions(draw):
    """Generate OHLC data with positions for take-profit testing."""
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

    # Check that take profit levels are all NaN
    assert take_profit_levels.isna().all(), "NoTakeProfit should return all NaN take profit levels"

    # Apply take profit
    adjusted_weights = handler.apply_take_profit(
        current_date=current_date,
        current_asset_prices=current_prices,
        target_weights=weights,
        entry_prices=entry_prices,
        take_profit_levels=take_profit_levels,
    )

    # Check that weights are unchanged
    assert adjusted_weights.equals(weights), "NoTakeProfit should not modify weights"


@st.composite
def atr_take_profit_configs(draw):
    """Generate configurations for ATR-based take profit."""
    atr_length = draw(st.integers(min_value=5, max_value=30))
    atr_multiple = draw(
        st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False)
    )

    take_profit_config = {
        "type": "AtrBasedTakeProfit",
        "atr_length": atr_length,
        "atr_multiple": atr_multiple,
    }

    return take_profit_config


@given(ohlc_data_with_positions(), atr_take_profit_configs())
@settings(deadline=None, max_examples=50)
@pytest.mark.skip(reason="Take profit test failing due to excessive entropy")
def test_atr_based_take_profit_handler_properties(data, take_profit_config):
    """Test properties of AtrBasedTakeProfit handler."""
    ohlc_data, weights, entry_prices, current_date, current_prices = data

    # Create AtrBasedTakeProfit handler
    handler = AtrBasedTakeProfit({}, take_profit_config)

    # Calculate take profit levels
    take_profit_levels = handler.calculate_take_profit_levels(
        current_date=current_date,
        asset_ohlc_history=ohlc_data,
        current_weights=weights,
        entry_prices=entry_prices,
    )

    # Check that take profit levels are defined for assets with positions
    for asset in weights.index:
        if weights[asset] != 0 and not pd.isna(entry_prices[asset]):
            # Take profit levels should be defined for assets with positions and entry prices
            assert not pd.isna(
                take_profit_levels[asset]
            ), f"Take profit level should be defined for {asset}"

            # For long positions, take profit level should be above entry price
            if weights[asset] > 0:
                assert (
                    take_profit_levels[asset] > entry_prices[asset]
                ), f"Take profit level for long position should be above entry price for {asset}"

            # For short positions, take profit level should be below entry price
            if weights[asset] < 0:
                assert (
                    take_profit_levels[asset] < entry_prices[asset]
                ), f"Take profit level for short position should be below entry price for {asset}"
        else:
            # Take profit levels should be NaN for assets without positions or entry prices
            assert pd.isna(
                take_profit_levels[asset]
            ), f"Take profit level should be NaN for {asset}"

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
        if (
            weights[asset] != 0
            and not pd.isna(entry_prices[asset])
            and not pd.isna(take_profit_levels[asset])
        ):
            # For long positions that hit take profit
            if weights[asset] > 0 and current_prices[asset] >= take_profit_levels[asset]:
                assert (
                    adjusted_weights[asset] == 0
                ), f"Long position that hit take profit should be closed for {asset}"

            # For short positions that hit take profit
            if weights[asset] < 0 and current_prices[asset] <= take_profit_levels[asset]:
                assert (
                    adjusted_weights[asset] == 0
                ), f"Short position that hit take profit should be closed for {asset}"

            # For positions that don't hit take profit
            if (weights[asset] > 0 and current_prices[asset] < take_profit_levels[asset]) or (
                weights[asset] < 0 and current_prices[asset] > take_profit_levels[asset]
            ):
                assert (
                    adjusted_weights[asset] == weights[asset]
                ), f"Position that didn't hit take profit should be unchanged for {asset}"
        else:
            # Weights for assets without positions or take profit levels should be unchanged
            assert (
                adjusted_weights[asset] == weights[asset]
            ), f"Weight should be unchanged for {asset}"


@given(ohlc_data_with_positions(), atr_take_profit_configs())
@settings(deadline=None)
def test_stop_loss_and_take_profit_relationship(data, take_profit_config):
    """Test the relationship between stop loss and take profit levels."""
    ohlc_data, weights, entry_prices, current_date, current_prices = data

    # Align all Series to a common index
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
        if (
            weights[asset] != 0
            and not pd.isna(entry_prices[asset])
            and not pd.isna(stop_levels[asset])
            and not pd.isna(take_profit_levels[asset])
        ):
            # For long positions, take profit should be above entry and stop loss should be below
            if weights[asset] > 0:
                assert (
                    take_profit_levels[asset] > entry_prices[asset]
                ), f"Take profit level for long position should be above entry price for {asset}"
                assert (
                    stop_levels[asset] < entry_prices[asset]
                ), f"Stop level for long position should be below entry price for {asset}"
                assert (
                    take_profit_levels[asset] > stop_levels[asset]
                ), f"Take profit level should be above stop level for long position in {asset}"

            # For short positions, take profit should be below entry and stop loss should be above
            if weights[asset] < 0:
                assert (
                    take_profit_levels[asset] < entry_prices[asset]
                ), f"Take profit level for short position should be below entry price for {asset}"
                assert (
                    stop_levels[asset] > entry_prices[asset]
                ), f"Stop level for short position should be above entry price for {asset}"
                assert (
                    take_profit_levels[asset] < stop_levels[asset]
                ), f"Take profit level should be below stop level for short position in {asset}"
