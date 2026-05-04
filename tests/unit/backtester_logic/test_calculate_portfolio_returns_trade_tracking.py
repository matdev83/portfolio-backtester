from __future__ import annotations

import pandas as pd
import pytest

from portfolio_backtester.backtester_logic.portfolio_logic import calculate_portfolio_returns


def _zero_cost_global(portfolio_value: float = 100_000.0) -> dict:
    return {
        "portfolio_value": portfolio_value,
        "commission_per_share": 0.0,
        "commission_min_per_order": 0.0,
        "commission_max_percent_of_trade": 0.0,
        "slippage_bps": 0.0,
    }


def test_track_trades_next_bar_open_entry_uses_execution_open_not_close():
    dates = pd.date_range("2023-01-03", periods=3, freq="B")
    ticker = "X"
    tuples = [(ticker, "Open"), (ticker, "Close")]
    ohlc = pd.DataFrame(
        {
            tuples[0]: [100.0, 50.0, 100.0],
            tuples[1]: [100.0, 100.0, 120.0],
        },
        index=dates,
    )
    ohlc.columns = pd.MultiIndex.from_tuples(tuples, names=["Ticker", "Field"])
    close_df = ohlc.xs("Close", level="Field", axis=1)
    rets = close_df.pct_change(fill_method=None).fillna(0.0)
    sized = pd.DataFrame(
        {ticker: [1.0, 0.0]},
        index=pd.DatetimeIndex([dates[0], dates[1]]),
    )
    scenario = {
        "timing_config": {"mode": "signal_based", "trade_execution_timing": "next_bar_open"},
    }
    g = _zero_cost_global(100_000.0)
    rets_net, tt = calculate_portfolio_returns(
        sized,
        scenario,
        ohlc,
        rets,
        [ticker],
        g,
        track_trades=True,
    )
    assert tt is not None
    completed = tt.trade_lifecycle_manager.get_completed_trades()
    assert len(completed) >= 1
    tr = completed[0]
    assert float(tr.entry_price) == pytest.approx(50.0)
    assert tr.entry_date == dates[1]
    assert float(tr.exit_price) == pytest.approx(100.0)
    assert tr.exit_date == dates[2]
    assert rets_net is not None


def test_track_trades_bar_close_day_zero_entry_survives_to_completed_trade():
    dates = pd.date_range("2023-01-03", periods=2, freq="B")
    ticker = "X"
    tuples = [(ticker, "Open"), (ticker, "Close")]
    ohlc = pd.DataFrame(
        {
            tuples[0]: [95.0, 100.0],
            tuples[1]: [100.0, 105.0],
        },
        index=dates,
    )
    ohlc.columns = pd.MultiIndex.from_tuples(tuples, names=["Ticker", "Field"])
    close_df = ohlc.xs("Close", level="Field", axis=1)
    rets = close_df.pct_change(fill_method=None).fillna(0.0)
    sized = pd.DataFrame(
        {ticker: [1.0, 0.0]},
        index=pd.DatetimeIndex([dates[0], dates[1]]),
    )
    scenario = {
        "timing_config": {"mode": "signal_based", "trade_execution_timing": "bar_close"},
    }
    g = _zero_cost_global(10_000.0)
    _, tt = calculate_portfolio_returns(
        sized,
        scenario,
        ohlc,
        rets,
        [ticker],
        g,
        track_trades=True,
    )
    assert tt is not None
    completed = tt.trade_lifecycle_manager.get_completed_trades()
    assert len(completed) >= 1
    tr = completed[0]
    assert tr.entry_date == dates[0]
    assert float(tr.entry_price) == pytest.approx(100.0)
