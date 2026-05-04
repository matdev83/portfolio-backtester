import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.trading.canonical_trade_analytics import (
    build_canonical_trade_analytics,
)


def _ledger_row(
    date_idx: int,
    date: pd.Timestamp,
    ticker: str,
    quantity: float,
    execution_price: float,
    position_before: float,
    position_after: float,
    cost: float = 0.0,
) -> dict:
    ev = quantity * execution_price
    return {
        "execution_date_idx": date_idx,
        "execution_date": date,
        "date_idx": date_idx,
        "date": date,
        "ticker": ticker,
        "quantity": quantity,
        "execution_price": execution_price,
        "execution_value": ev,
        "cash_before": 0.0,
        "cash_after": 0.0,
        "position_before": position_before,
        "position_after": position_after,
        "cost": cost,
    }


_COMPLETED_COLS = [
    "ticker",
    "side",
    "entry_date",
    "exit_date",
    "entry_price",
    "exit_price",
    "quantity",
    "entry_value",
    "entry_cost",
    "exit_cost",
    "duration_days",
    "pnl_gross",
    "pnl_net",
    "mfe",
    "mae",
]

_OPEN_COLS = [
    "ticker",
    "side",
    "entry_date",
    "entry_price",
    "quantity",
    "entry_value",
    "entry_cost",
    "current_quantity",
]

_SUMMARY_COLS = [
    "num_completed",
    "num_open",
    "gross_pnl",
    "net_pnl",
    "long_count",
    "short_count",
]


def test_build_canonical_trade_analytics_completed_long_round_trip():
    dates = pd.date_range("2023-01-01", periods=3, freq="D")
    close_df = pd.DataFrame({"X": [100.0, 100.0, 110.0]}, index=dates)
    rows = [
        _ledger_row(0, dates[0], "X", 10.0, 100.0, 0.0, 10.0, cost=1.0),
        _ledger_row(2, dates[2], "X", -10.0, 110.0, 10.0, 0.0, cost=2.0),
    ]
    ledger = pd.DataFrame(rows)
    pv = pd.Series(100_000.0, index=dates)
    pos = pd.DataFrame({"X": [0.0, 10.0, 0.0]}, index=dates)
    out = build_canonical_trade_analytics(ledger, pv, pos, close_df)
    done = out["completed_trades"]
    assert list(done.columns) == _COMPLETED_COLS
    assert len(done) == 1
    r = done.iloc[0]
    assert r["ticker"] == "X"
    assert r["side"] == "long"
    assert r["entry_price"] == pytest.approx(100.0)
    assert r["exit_price"] == pytest.approx(110.0)
    assert r["quantity"] == pytest.approx(10.0)
    assert r["entry_value"] == pytest.approx(1000.0)
    assert r["entry_cost"] == pytest.approx(1.0)
    assert r["exit_cost"] == pytest.approx(2.0)
    assert r["pnl_gross"] == pytest.approx(100.0)
    assert r["pnl_net"] == pytest.approx(97.0)
    per_sh = np.array([0.0, 0.0, 10.0])
    assert r["mfe"] == pytest.approx(float(per_sh.max()) * 10.0)
    assert r["mae"] == pytest.approx(float(per_sh.min()) * 10.0)
    open_df = out["open_trades"]
    assert list(open_df.columns) == _OPEN_COLS
    assert open_df.empty
    summ = out["summary"]
    assert list(summ.columns) == _SUMMARY_COLS
    assert summ.iloc[0]["num_completed"] == 1
    assert summ.iloc[0]["num_open"] == 0
    assert summ.iloc[0]["gross_pnl"] == pytest.approx(100.0)
    assert summ.iloc[0]["net_pnl"] == pytest.approx(97.0)
    assert summ.iloc[0]["long_count"] == 1
    assert summ.iloc[0]["short_count"] == 0


def test_build_canonical_trade_analytics_open_position():
    dates = pd.date_range("2023-01-01", periods=2, freq="D")
    close_df = pd.DataFrame({"Y": [50.0, 52.0]}, index=dates)
    rows = [
        _ledger_row(0, dates[0], "Y", 5.0, 50.0, 0.0, 5.0, cost=0.5),
    ]
    ledger = pd.DataFrame(rows)
    pv = pd.Series([100_000.0, 100_250.0], index=dates)
    pos = pd.DataFrame({"Y": [5.0, 5.0]}, index=dates)
    out = build_canonical_trade_analytics(ledger, pv, pos, close_df)
    assert out["completed_trades"].empty
    o = out["open_trades"]
    assert len(o) == 1
    r = o.iloc[0]
    assert r["ticker"] == "Y"
    assert r["side"] == "long"
    assert r["entry_price"] == pytest.approx(50.0)
    assert r["quantity"] == pytest.approx(5.0)
    assert r["current_quantity"] == pytest.approx(5.0)
    assert r["entry_cost"] == pytest.approx(0.5)
    summ = out["summary"]
    assert summ.iloc[0]["num_completed"] == 0
    assert summ.iloc[0]["num_open"] == 1
    assert summ.iloc[0]["gross_pnl"] == pytest.approx(0.0)
    assert summ.iloc[0]["net_pnl"] == pytest.approx(0.0)


def test_build_canonical_trade_analytics_partial_close():
    dates = pd.date_range("2023-01-01", periods=3, freq="D")
    close_df = pd.DataFrame({"X": [100.0, 130.0, 100.0]}, index=dates)
    rows = [
        _ledger_row(0, dates[0], "X", 10.0, 100.0, 0.0, 10.0),
        _ledger_row(1, dates[1], "X", -3.0, 100.0, 10.0, 7.0),
        _ledger_row(2, dates[2], "X", -7.0, 100.0, 7.0, 0.0),
    ]
    ledger = pd.DataFrame(rows)
    pv = pd.Series(100_000.0, index=dates)
    pos = pd.DataFrame({"X": [10.0, 7.0, 0.0]}, index=dates)
    out = build_canonical_trade_analytics(ledger, pv, pos, close_df)
    done = out["completed_trades"]
    assert len(done) == 2
    qties = sorted(float(x) for x in done["quantity"].tolist())
    assert qties == [3.0, 7.0]
    summ = out["summary"]
    assert summ.iloc[0]["num_completed"] == 2
    assert summ.iloc[0]["long_count"] == 2
    assert summ.iloc[0]["short_count"] == 0


def test_build_canonical_trade_analytics_flip_two_completed():
    dates = pd.date_range("2023-01-01", periods=3, freq="D")
    close_df = pd.DataFrame({"X": [100.0, 120.0, 100.0]}, index=dates)
    rows = [
        _ledger_row(0, dates[0], "X", 10.0, 100.0, 0.0, 10.0),
        _ledger_row(1, dates[1], "X", -20.0, 120.0, 10.0, -10.0),
        _ledger_row(2, dates[2], "X", 10.0, 100.0, -10.0, 0.0),
    ]
    ledger = pd.DataFrame(rows)
    pv = pd.Series(100_000.0, index=dates)
    pos = pd.DataFrame({"X": [10.0, -10.0, 0.0]}, index=dates)
    out = build_canonical_trade_analytics(ledger, pv, pos, close_df)
    done = out["completed_trades"]
    assert len(done) == 2
    sides = set(done["side"].tolist())
    assert sides == {"long", "short"}
    summ = out["summary"]
    assert summ.iloc[0]["long_count"] == 1
    assert summ.iloc[0]["short_count"] == 1


def test_build_canonical_trade_analytics_long_short_net_gross():
    dates = pd.date_range("2023-01-01", periods=4, freq="D")
    close_df = pd.DataFrame(
        {"X": [100.0, 110.0, 100.0, 100.0], "Z": [200.0, 200.0, 220.0, 210.0]},
        index=dates,
    )
    rows = [
        _ledger_row(0, dates[0], "X", 10.0, 100.0, 0.0, 10.0, cost=1.0),
        _ledger_row(1, dates[1], "X", -10.0, 110.0, 10.0, 0.0, cost=1.0),
        _ledger_row(2, dates[2], "Z", -5.0, 220.0, 0.0, -5.0, cost=1.0),
        _ledger_row(3, dates[3], "Z", 5.0, 210.0, -5.0, 0.0, cost=1.0),
    ]
    ledger = pd.DataFrame(rows)
    pv = pd.Series(100_000.0, index=dates)
    pos = pd.DataFrame(
        {"X": [10.0, 0.0, 0.0, 0.0], "Z": [0.0, 0.0, -5.0, 0.0]},
        index=dates,
    )
    out = build_canonical_trade_analytics(ledger, pv, pos, close_df)
    done = out["completed_trades"]
    assert len(done) == 2
    gross = float(done["pnl_gross"].sum())
    net = float(done["pnl_net"].sum())
    assert gross == pytest.approx(100.0 + 50.0)
    assert net == pytest.approx(98.0 + 48.0)
    summ = out["summary"]
    assert summ.iloc[0]["gross_pnl"] == pytest.approx(gross)
    assert summ.iloc[0]["net_pnl"] == pytest.approx(net)
