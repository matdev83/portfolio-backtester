import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from hypothesis import given, settings, strategies as st, assume
from hypothesis.extra import numpy as hnp

from portfolio_backtester.risk_management.stop_loss_handlers import AtrBasedStopLoss
from portfolio_backtester.risk_management.take_profit_handlers import NoTakeProfit, AtrBasedTakeProfit

# Reused simplified composite strategy
@st.composite
def ohlc_data_with_positions(draw):
    n_assets = draw(st.integers(min_value=2, max_value=5))
    n_days = draw(st.integers(min_value=30, max_value=100))
    assets = [f"ASSET{i}" for i in range(n_assets)]

    start_date = draw(st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2023, 12, 31)))
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    index = pd.DatetimeIndex([pd.Timestamp(dt) for dt in dates])

    ohlc_data = {}
    for asset in assets:
        base_price = draw(st.floats(min_value=50.0, max_value=500.0, allow_nan=False, allow_infinity=False))
        daily_changes = draw(hnp.arrays(dtype=float, shape=n_days, elements=st.floats(min_value=-0.05, max_value=0.05, allow_nan=False, allow_infinity=False)))
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
                "Open": open_price, "High": high_price, "Low": low_price, "Close": close_price,
                "Volume": draw(st.integers(min_value=1000, max_value=1000000)),
            }
        ohlc_data[asset] = pd.DataFrame.from_dict(asset_data, orient="index")

    combined_data = pd.concat(ohlc_data, axis=1)
    weights = pd.Series(draw(hnp.arrays(dtype=float, shape=n_assets, elements=st.floats(min_value=-1.0, max_value=1.0))), index=assets)
    
    entry_prices = pd.Series(index=assets, dtype=float)
    for asset in assets:
        if weights[asset] != 0:
            past_idx = draw(st.integers(min_value=0, max_value=int(n_days * 0.8)))
            entry_prices[asset] = combined_data[asset, "Close"].iloc[past_idx]
        else:
            entry_prices[asset] = np.nan

    current_date_idx = draw(st.integers(min_value=int(n_days * 0.8), max_value=n_days - 1))
    current_date = index[current_date_idx]
    current_prices = pd.Series([combined_data[asset, "Close"].iloc[current_date_idx] for asset in assets], index=assets)

    return combined_data, weights, entry_prices, current_date, current_prices

@given(ohlc_data_with_positions())
@settings(deadline=None)
def test_no_take_profit_handler_properties(data):
    ohlc_data, weights, entry_prices, current_date, current_prices = data
    handler = NoTakeProfit({}, {})
    tp_levels = handler.calculate_take_profit_levels(current_date, ohlc_data, weights, entry_prices)
    assert tp_levels.isna().all()
    adjusted = handler.apply_take_profit(current_date, current_prices, weights, entry_prices, tp_levels)
    assert adjusted.equals(weights)

@st.composite
def atr_take_profit_configs(draw):
    return {
        "type": "AtrBasedTakeProfit",
        "atr_length": draw(st.integers(min_value=5, max_value=30)),
        "atr_multiple": draw(st.floats(min_value=1.0, max_value=5.0)),
    }

@given(ohlc_data_with_positions(), atr_take_profit_configs())
@settings(deadline=None, max_examples=30)
def test_atr_based_take_profit_handler_properties(data, take_profit_config):
    ohlc_data, weights, entry_prices, current_date, current_prices = data
    # Ensure all Series have same index
    common_index = weights.index
    weights = weights.copy()
    entry_prices = entry_prices.reindex(common_index)
    current_prices = current_prices.reindex(common_index)

    handler = AtrBasedTakeProfit({}, take_profit_config)
    tp_levels = handler.calculate_take_profit_levels(current_date, ohlc_data, weights, entry_prices)

    for asset in weights.index:
        if weights[asset] != 0 and not pd.isna(entry_prices[asset]):
            # TP level might be NaN if not enough history
            if not pd.isna(tp_levels[asset]):
                if weights[asset] > 0:
                    assert tp_levels[asset] > entry_prices[asset]
                else:
                    assert tp_levels[asset] < entry_prices[asset]
        else:
            assert pd.isna(tp_levels[asset])

    adjusted_weights = handler.apply_take_profit(current_date, current_prices, weights, entry_prices, tp_levels)
    
    for asset in weights.index:
        if weights[asset] != 0 and not pd.isna(tp_levels[asset]):
            if weights[asset] > 0:
                if current_prices[asset] >= tp_levels[asset]:
                    assert adjusted_weights[asset] == 0
                else:
                    assert adjusted_weights[asset] == weights[asset]
            elif weights[asset] < 0:
                if current_prices[asset] <= tp_levels[asset]:
                    assert adjusted_weights[asset] == 0
                else:
                    assert adjusted_weights[asset] == weights[asset]

@given(ohlc_data_with_positions(), atr_take_profit_configs())
@settings(deadline=None, max_examples=30)
def test_stop_loss_and_take_profit_relationship(data, take_profit_config):
    ohlc_data, weights, entry_prices, current_date, current_prices = data
    # Align
    common_index = weights.index
    weights = weights.copy()
    entry_prices = entry_prices.reindex(common_index)
    current_prices = current_prices.reindex(common_index)

    stop_loss_config = {
        "type": "AtrBasedStopLoss",
        "atr_length": take_profit_config["atr_length"],
        "atr_multiple": take_profit_config["atr_multiple"],
    }

    sl_handler = AtrBasedStopLoss({}, stop_loss_config)
    tp_handler = AtrBasedTakeProfit({}, take_profit_config)

    sl_levels = sl_handler.calculate_stop_levels(current_date, ohlc_data, weights, entry_prices)
    tp_levels = tp_handler.calculate_take_profit_levels(current_date, ohlc_data, weights, entry_prices)

    for asset in weights.index:
        if (weights[asset] != 0 
            and not pd.isna(entry_prices[asset]) 
            and not pd.isna(sl_levels[asset]) 
            and not pd.isna(tp_levels[asset])):
            
            if weights[asset] > 0:
                # Long: SL < Entry < TP
                assert sl_levels[asset] < entry_prices[asset]
                assert tp_levels[asset] > entry_prices[asset]
                assert tp_levels[asset] > sl_levels[asset]
            elif weights[asset] < 0:
                # Short: TP < Entry < SL
                assert sl_levels[asset] > entry_prices[asset]
                assert tp_levels[asset] < entry_prices[asset]
                assert tp_levels[asset] < sl_levels[asset]