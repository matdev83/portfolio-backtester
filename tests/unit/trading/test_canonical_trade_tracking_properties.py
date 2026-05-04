"""Property-style tests for the canonical kernel-aligned trade-tracking helper."""

import warnings

import numpy as np
import pandas as pd
from hypothesis import assume, given, settings

from portfolio_backtester.trading.numba_trade_tracker import (
    track_trades_canonical,
    track_trades_vectorized,
)

from tests.strategies.trading_strategies import (
    trade_tracking_inputs,
    weights_and_prices,
)


def test_track_trades_canonical_fixed_allocation_mode_is_wired():
    dates = pd.date_range("2023-01-03", periods=3, freq="B")
    weights_daily = pd.DataFrame({"A": [1.0, 1.0, 0.0]}, index=dates)
    ohlc = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0]},
        index=dates,
    )
    price_data = pd.DataFrame(
        index=dates,
        columns=pd.MultiIndex.from_tuples(
            [("A", f) for f in ["Open", "High", "Low", "Close", "Volume"]],
            names=["Ticker", "Field"],
        ),
    )
    for f in ["Open", "High", "Low", "Close"]:
        price_data[("A", f)] = ohlc["A"]
    price_data[("A", "Volume")] = 10000.0
    costs = pd.Series(0.0, index=dates)
    stats = track_trades_canonical(
        weights_daily,
        price_data,
        costs,
        portfolio_value=50_000.0,
        allocation_mode="fixed_capital",
    )
    assert stats.get("allocation_mode") == "fixed_capital"


def test_track_trades_vectorized_emits_deprecation_warning():
    dates = pd.date_range("2023-01-03", periods=2, freq="B")
    weights_daily = pd.DataFrame({"A": [1.0, 1.0]}, index=dates)
    price_data = pd.DataFrame({"A": [100.0, 110.0]}, index=dates)
    costs = pd.Series(0.0, index=dates)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        stats_new = track_trades_canonical(
            weights_daily, price_data, costs, portfolio_value=100_000.0
        )
        stats_old = track_trades_vectorized(
            weights_daily, price_data, costs, portfolio_value=100_000.0
        )
        assert any(
            issubclass(x.category, DeprecationWarning)
            and "track_trades_vectorized" in str(x.message)
            for x in w
        )
    assert stats_new == stats_old


@given(trade_tracking_inputs())
@settings(deadline=None)
def test_track_trades_canonical_returns_trade_statistics(inputs):
    """Canonical helper returns TradeTracker-shaped statistics dictionaries."""
    weights_daily, price_data_daily_ohlc, transaction_costs, portfolio_value = inputs

    assume(not weights_daily.empty and not price_data_daily_ohlc.empty)

    trade_stats = track_trades_canonical(
        weights_daily, price_data_daily_ohlc, transaction_costs, portfolio_value
    )

    assert isinstance(trade_stats, dict)
    assert "all_num_trades" in trade_stats
    assert "allocation_mode" in trade_stats


@given(weights_and_prices())
@settings(deadline=None)
def test_track_trades_canonical_deterministic(data):
    """Repeated calls share stable numeric aggregates when inputs match."""
    weights_df, prices_df, transaction_costs = data

    assume(not weights_df.empty and not prices_df.empty)

    price_data_daily_ohlc = pd.DataFrame(index=prices_df.index)
    for ticker in prices_df.columns:
        for field in ["Open", "High", "Low", "Close", "Volume"]:
            if field == "Close":
                price_data_daily_ohlc[(ticker, field)] = prices_df[ticker]
            elif field == "Open":
                price_data_daily_ohlc[(ticker, field)] = prices_df[ticker] * 0.99
            elif field == "High":
                price_data_daily_ohlc[(ticker, field)] = prices_df[ticker] * 1.01
            elif field == "Low":
                price_data_daily_ohlc[(ticker, field)] = prices_df[ticker] * 0.98
            else:
                price_data_daily_ohlc[(ticker, field)] = 10000

    price_data_daily_ohlc.columns = pd.MultiIndex.from_tuples(
        price_data_daily_ohlc.columns, names=["Ticker", "Field"]
    )

    portfolio_value = 100000.0
    stats_a = track_trades_canonical(
        weights_df, price_data_daily_ohlc, transaction_costs, portfolio_value
    )
    stats_b = track_trades_canonical(
        weights_df, price_data_daily_ohlc, transaction_costs, portfolio_value
    )

    numeric_keys = {
        k for k, v in stats_a.items() if isinstance(v, (int, float, np.integer, np.floating))
    }

    for key in numeric_keys:
        assert isinstance(stats_a[key], (int, float, np.integer, np.floating))
        assert stats_a[key] == stats_b[key]
