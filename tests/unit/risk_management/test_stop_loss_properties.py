import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as hnp

from portfolio_backtester.risk_management.stop_loss_handlers import NoStopLoss, AtrBasedStopLoss
from portfolio_backtester.risk_management.atr_service import calculate_atr_fast


# Replicate the composite strategy
@st.composite
def ohlc_data_with_positions(draw):
    n_assets = draw(st.integers(min_value=2, max_value=4))
    n_days = draw(st.integers(min_value=30, max_value=60))
    assets = [f"ASSET{i}" for i in range(n_assets)]

    start_date = draw(
        st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2023, 12, 31))
    )
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    index = pd.DatetimeIndex([pd.Timestamp(dt) for dt in dates])

    ohlc_data = {}
    for asset in assets:
        base_price = draw(
            st.floats(min_value=50.0, max_value=500.0, allow_nan=False, allow_infinity=False)
        )
        daily_changes = draw(
            hnp.arrays(
                dtype=float,
                shape=n_days,
                elements=st.floats(
                    min_value=-0.05, max_value=0.05, allow_nan=False, allow_infinity=False
                ),
            )
        )
        prices = base_price * np.cumprod(1 + daily_changes)

        asset_data = {}
        for i in range(n_days):
            price = prices[i]
            open_price = price * draw(st.floats(min_value=0.99, max_value=1.01))
            high_price = price * draw(st.floats(min_value=1.01, max_value=1.03))
            low_price = price * draw(st.floats(min_value=0.97, max_value=0.99))
            close_price = price * draw(st.floats(min_value=0.99, max_value=1.01))
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            asset_data[index[i]] = {
                "Open": open_price,
                "High": high_price,
                "Low": low_price,
                "Close": close_price,
                "Volume": draw(st.integers(min_value=1000, max_value=1000000)),
            }
        ohlc_data[asset] = pd.DataFrame.from_dict(asset_data, orient="index")

    combined_data = pd.concat(ohlc_data, axis=1)

    # Weights
    weights = pd.Series(
        draw(
            hnp.arrays(
                dtype=float, shape=n_assets, elements=st.floats(min_value=-1.0, max_value=1.0)
            )
        ),
        index=assets,
    )

    entry_prices = pd.Series(index=assets, dtype=float)
    for asset in assets:
        if weights[asset] != 0:
            past_idx = draw(st.integers(min_value=0, max_value=int(n_days * 0.8)))
            entry_prices[asset] = combined_data[asset, "Close"].iloc[past_idx]
        else:
            entry_prices[asset] = np.nan

    current_date_idx = draw(st.integers(min_value=int(n_days * 0.8), max_value=n_days - 1))
    current_date = index[current_date_idx]
    current_prices = pd.Series(
        [combined_data[asset, "Close"].iloc[current_date_idx] for asset in assets], index=assets
    )

    return combined_data, weights, entry_prices, current_date, current_prices


@given(ohlc_data_with_positions())
@settings(deadline=None, max_examples=20)
def test_no_stop_loss_handler_properties(data):
    ohlc_data, weights, entry_prices, current_date, current_prices = data
    handler = NoStopLoss({}, {})
    stop_levels = handler.calculate_stop_levels(current_date, ohlc_data, weights, entry_prices)
    assert stop_levels.isna().all()
    adjusted = handler.apply_stop_loss(
        current_date, current_prices, weights, entry_prices, stop_levels
    )
    assert adjusted.equals(weights)


@st.composite
def atr_stop_loss_configs(draw):
    return {
        "type": "AtrBasedStopLoss",
        "atr_length": draw(st.integers(min_value=5, max_value=30)),
        "atr_multiple": draw(st.floats(min_value=1.0, max_value=5.0)),
    }


@given(ohlc_data_with_positions(), atr_stop_loss_configs())
@settings(deadline=None, max_examples=25)
def test_atr_based_stop_loss_handler_properties(data, stop_loss_config):
    ohlc_data, weights, entry_prices, current_date, current_prices = data
    # Ensure all Series have same index
    common_index = weights.index
    weights = weights.copy()
    entry_prices = entry_prices.reindex(common_index)
    current_prices = current_prices.reindex(common_index)

    handler = AtrBasedStopLoss({}, stop_loss_config)
    stop_levels = handler.calculate_stop_levels(current_date, ohlc_data, weights, entry_prices)

    for asset in weights.index:
        if weights[asset] != 0 and not pd.isna(entry_prices[asset]):
            # Valid position. Stop level MIGHT be defined if enough data exists.
            # We can't strictly assert it IS defined unless we check data sufficiency.
            if not pd.isna(stop_levels[asset]):
                if weights[asset] > 0:
                    assert stop_levels[asset] < entry_prices[asset]
                else:
                    assert stop_levels[asset] > entry_prices[asset]
        else:
            assert pd.isna(stop_levels[asset])

    adjusted_weights = handler.apply_stop_loss(
        current_date, current_prices, weights, entry_prices, stop_levels
    )

    # Check logic
    for asset in weights.index:
        if weights[asset] != 0 and not pd.isna(stop_levels[asset]):
            if weights[asset] > 0:
                if current_prices[asset] <= stop_levels[asset]:
                    assert adjusted_weights[asset] == 0
                else:
                    assert adjusted_weights[asset] == weights[asset]
            elif weights[asset] < 0:
                if current_prices[asset] >= stop_levels[asset]:
                    assert adjusted_weights[asset] == 0
                else:
                    assert adjusted_weights[asset] == weights[asset]


@given(ohlc_data_with_positions(), atr_stop_loss_configs())
@settings(deadline=None, max_examples=15)
def test_atr_calculation_properties(data, stop_loss_config):
    ohlc_data, weights, entry_prices, current_date, current_prices = data
    atr_length = stop_loss_config["atr_length"]

    atr_values = calculate_atr_fast(ohlc_data, current_date, atr_length)

    # If ohlc_data is MultiIndex, we need to extract asset-specific data carefully
    for asset, atr_val in atr_values.items():
        if pd.isna(atr_val):
            continue

        # Basic sanity check
        assert atr_val >= 0

        # Approximate volatility check
        # Extract asset dataframe safely
        try:
            if isinstance(ohlc_data.columns, pd.MultiIndex):
                # Use xs to guarantee we get the cross-section for the asset
                # This handles MultiIndex columns robustly
                asset_df = ohlc_data.xs(asset, axis=1, level=0)
            else:
                continue
        except (KeyError, ValueError):
            continue

        # Ensure we have a DataFrame with required columns
        if (
            not isinstance(asset_df, pd.DataFrame)
            or "High" not in asset_df.columns
            or "Low" not in asset_df.columns
        ):
            continue

        # Filter to dates up to current_date
        history = asset_df.loc[:current_date]
        if len(history) < atr_length:
            continue

        recent = history.iloc[-atr_length:]

        # Double check recent is a DataFrame (it should be if history is)
        if not isinstance(recent, pd.DataFrame):
            continue

        high_low_range = (recent["High"] - recent["Low"]).mean()

        # ATR should be somewhat close to average High-Low range (ignoring gaps for simplicity)
        # Typically ATR >= High-Low range because it includes gaps.
        # Allow factor of 5 for safety
        if high_low_range > 0:
            assert atr_val >= high_low_range * 0.5
            assert atr_val <= high_low_range * 5.0
