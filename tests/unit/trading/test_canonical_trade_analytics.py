import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.trading.canonical_trade_analytics import (
    build_canonical_trade_analytics,
)
from portfolio_backtester.trading.trade_tracker import TradeTracker


def _ledger_row(
    execution_date_idx: int,
    execution_date: pd.Timestamp,
    ticker: str,
    quantity: float,
    execution_price: float,
    position_before: float,
    position_after: float,
    cost: float = 0.0,
    *,
    decision_date_idx: int | None = None,
    decision_date: pd.Timestamp | None = None,
) -> dict:
    ev = quantity * execution_price
    d_idx = execution_date_idx if decision_date_idx is None else decision_date_idx
    d_dt = execution_date if decision_date is None else decision_date
    return {
        "decision_date_idx": d_idx,
        "decision_date": d_dt,
        "execution_date_idx": execution_date_idx,
        "execution_date": execution_date,
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


def test_build_canonical_trade_analytics_replay_orders_by_execution_session_not_decision_session():
    dates = pd.date_range("2023-01-01", periods=2, freq="D")
    close_df = pd.DataFrame({"X": [100.0, 100.0]}, index=dates)
    rows = [
        _ledger_row(
            1,
            dates[1],
            "X",
            1.0,
            100.0,
            0.0,
            1.0,
            decision_date_idx=0,
            decision_date=dates[0],
        ),
    ]
    ledger = pd.DataFrame(rows)
    pv = pd.Series([100_000.0, 100_100.0], index=dates)
    pos = pd.DataFrame({"X": [0.0, 1.0]}, index=dates)
    out = build_canonical_trade_analytics(ledger, pv, pos, close_df)
    o = out["open_trades"]
    assert len(o) == 1
    assert pd.Timestamp(o.iloc[0]["entry_date"]) == dates[1]


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


def _expected_per_share_mfe_mae_long(path: np.ndarray, entry: float) -> tuple[float, float]:
    mfe_ps = 0.0
    mae_ps = 0.0
    for cur in path:
        pnl_ps = float(cur) - entry
        if pnl_ps > mfe_ps:
            mfe_ps = pnl_ps
        if pnl_ps < mae_ps:
            mae_ps = pnl_ps
    return mfe_ps, mae_ps


def _expected_per_share_mfe_mae_short(path: np.ndarray, entry: float) -> tuple[float, float]:
    mfe_ps = 0.0
    mae_ps = 0.0
    for cur in path:
        pnl_ps = entry - float(cur)
        if pnl_ps > mfe_ps:
            mfe_ps = pnl_ps
        if pnl_ps < mae_ps:
            mae_ps = pnl_ps
    return mfe_ps, mae_ps


def test_populate_from_execution_ledger_refreshes_open_leg_mfe_mae_after_same_direction_short_add():
    dates = pd.date_range("2023-01-01", periods=2, freq="D")
    close_df = pd.DataFrame({"X": [100.0, 60.0]}, index=dates)
    rows = [
        _ledger_row(0, dates[0], "X", -10.0, 100.0, 0.0, -10.0),
        _ledger_row(1, dates[1], "X", -10.0, 60.0, -10.0, -20.0),
    ]
    ledger = pd.DataFrame(rows)
    pv = pd.Series(100_000.0, index=dates)
    pos = pd.DataFrame({"X": [-10.0, -20.0]}, index=dates)
    tracker = TradeTracker()
    tracker.populate_from_execution_ledger(ledger, pv, pos, close_df)
    t = tracker.trade_lifecycle_manager.open_positions["X"]
    assert t.entry_price == pytest.approx(80.0)
    mfe_ps, mae_ps = _expected_per_share_mfe_mae_short(
        close_df["X"].to_numpy(dtype=float, copy=False), 80.0
    )
    assert t.mfe == pytest.approx(mfe_ps)
    assert t.mae == pytest.approx(mae_ps)


def test_populate_from_execution_ledger_refreshes_open_leg_mfe_mae_after_same_direction_add():
    dates = pd.date_range("2023-01-01", periods=2, freq="D")
    close_df = pd.DataFrame({"X": [100.0, 160.0]}, index=dates)
    rows = [
        _ledger_row(0, dates[0], "X", 10.0, 100.0, 0.0, 10.0),
        _ledger_row(1, dates[1], "X", 10.0, 160.0, 10.0, 20.0),
    ]
    ledger = pd.DataFrame(rows)
    pv = pd.Series(100_000.0, index=dates)
    pos = pd.DataFrame({"X": [10.0, 20.0]}, index=dates)
    tracker = TradeTracker()
    tracker.populate_from_execution_ledger(ledger, pv, pos, close_df)
    t = tracker.trade_lifecycle_manager.open_positions["X"]
    assert t.entry_price == pytest.approx(130.0)
    mfe_ps, mae_ps = _expected_per_share_mfe_mae_long(
        close_df["X"].to_numpy(dtype=float, copy=False), 130.0
    )
    assert t.mfe == pytest.approx(mfe_ps)
    assert t.mae == pytest.approx(mae_ps)


def test_build_canonical_trade_analytics_long_same_direction_add_mfe_mae_uses_averaged_basis():
    dates = pd.date_range("2023-01-01", periods=4, freq="D")
    close_df = pd.DataFrame({"X": [100.0, 160.0, 140.0, 130.0]}, index=dates)
    rows = [
        _ledger_row(0, dates[0], "X", 10.0, 100.0, 0.0, 10.0),
        _ledger_row(1, dates[1], "X", 10.0, 160.0, 10.0, 20.0),
        _ledger_row(3, dates[3], "X", -20.0, 130.0, 20.0, 0.0),
    ]
    ledger = pd.DataFrame(rows)
    pv = pd.Series(100_000.0, index=dates)
    pos = pd.DataFrame({"X": [10.0, 20.0, 20.0, 0.0]}, index=dates)
    out = build_canonical_trade_analytics(ledger, pv, pos, close_df)
    done = out["completed_trades"]
    assert len(done) == 1
    r = done.iloc[0]
    assert r["entry_price"] == pytest.approx(130.0)
    assert r["quantity"] == pytest.approx(20.0)
    path = close_df["X"].to_numpy(dtype=float, copy=False)
    mfe_ps, mae_ps = _expected_per_share_mfe_mae_long(path, 130.0)
    assert r["mfe"] == pytest.approx(mfe_ps * 20.0)
    assert r["mae"] == pytest.approx(mae_ps * 20.0)


def test_build_canonical_trade_analytics_short_same_direction_add_mfe_mae_uses_averaged_basis():
    dates = pd.date_range("2023-01-01", periods=4, freq="D")
    close_df = pd.DataFrame({"X": [100.0, 60.0, 90.0, 95.0]}, index=dates)
    rows = [
        _ledger_row(0, dates[0], "X", -10.0, 100.0, 0.0, -10.0),
        _ledger_row(1, dates[1], "X", -10.0, 60.0, -10.0, -20.0),
        _ledger_row(3, dates[3], "X", 20.0, 95.0, -20.0, 0.0),
    ]
    ledger = pd.DataFrame(rows)
    pv = pd.Series(100_000.0, index=dates)
    pos = pd.DataFrame({"X": [-10.0, -20.0, -20.0, 0.0]}, index=dates)
    out = build_canonical_trade_analytics(ledger, pv, pos, close_df)
    done = out["completed_trades"]
    assert len(done) == 1
    r = done.iloc[0]
    assert r["side"] == "short"
    assert r["entry_price"] == pytest.approx(80.0)
    assert r["quantity"] == pytest.approx(-20.0)
    path = close_df["X"].to_numpy(dtype=float, copy=False)
    mfe_ps, mae_ps = _expected_per_share_mfe_mae_short(path, 80.0)
    assert r["mfe"] == pytest.approx(mfe_ps * 20.0)
    assert r["mae"] == pytest.approx(mae_ps * 20.0)


def test_build_canonical_trade_analytics_long_add_then_partial_close_mfe_mae():
    dates = pd.date_range("2023-01-01", periods=4, freq="D")
    close_df = pd.DataFrame({"X": [100.0, 200.0, 180.0, 160.0]}, index=dates)
    rows = [
        _ledger_row(0, dates[0], "X", 10.0, 100.0, 0.0, 10.0),
        _ledger_row(1, dates[1], "X", 10.0, 200.0, 10.0, 20.0),
        _ledger_row(2, dates[2], "X", -5.0, 180.0, 20.0, 15.0),
        _ledger_row(3, dates[3], "X", -15.0, 160.0, 15.0, 0.0),
    ]
    ledger = pd.DataFrame(rows)
    pv = pd.Series(100_000.0, index=dates)
    pos = pd.DataFrame({"X": [10.0, 20.0, 15.0, 0.0]}, index=dates)
    out = build_canonical_trade_analytics(ledger, pv, pos, close_df)
    done = out["completed_trades"]
    assert len(done) == 2
    q5 = done.loc[done["quantity"].abs() == 5.0].iloc[0]
    q15 = done.loc[done["quantity"].abs() == 15.0].iloc[0]
    assert q5["entry_price"] == pytest.approx(150.0)
    path_to_d2 = close_df["X"].iloc[:3].to_numpy(dtype=float, copy=False)
    mfe5, mae5 = _expected_per_share_mfe_mae_long(path_to_d2, 150.0)
    assert float(q5["mfe"]) == pytest.approx(mfe5 * 5.0)
    assert float(q5["mae"]) == pytest.approx(mae5 * 5.0)
    path_full = close_df["X"].to_numpy(dtype=float, copy=False)
    mfe15, mae15 = _expected_per_share_mfe_mae_long(path_full, 150.0)
    assert float(q15["mfe"]) == pytest.approx(mfe15 * 15.0)
    assert float(q15["mae"]) == pytest.approx(mae15 * 15.0)


def test_build_canonical_trade_analytics_long_add_then_flip_mfe_on_first_leg():
    dates = pd.date_range("2023-01-01", periods=4, freq="D")
    close_df = pd.DataFrame({"X": [100.0, 200.0, 180.0, 170.0]}, index=dates)
    rows = [
        _ledger_row(0, dates[0], "X", 10.0, 100.0, 0.0, 10.0),
        _ledger_row(1, dates[1], "X", 10.0, 200.0, 10.0, 20.0),
        _ledger_row(2, dates[2], "X", -30.0, 180.0, 20.0, -10.0),
        _ledger_row(3, dates[3], "X", 10.0, 170.0, -10.0, 0.0),
    ]
    ledger = pd.DataFrame(rows)
    pv = pd.Series(100_000.0, index=dates)
    pos = pd.DataFrame({"X": [10.0, 20.0, -10.0, 0.0]}, index=dates)
    out = build_canonical_trade_analytics(ledger, pv, pos, close_df)
    done = out["completed_trades"]
    assert len(done) == 2
    long_leg = done.loc[done["side"] == "long"].iloc[0]
    assert long_leg["entry_price"] == pytest.approx(150.0)
    path_to_flip = close_df["X"].iloc[:3].to_numpy(dtype=float, copy=False)
    mfe_ps, mae_ps = _expected_per_share_mfe_mae_long(path_to_flip, 150.0)
    assert float(long_leg["mfe"]) == pytest.approx(mfe_ps * 20.0)
    assert float(long_leg["mae"]) == pytest.approx(mae_ps * 20.0)


def test_build_canonical_trade_analytics_long_same_direction_add_same_day_stable_sort():
    dates = pd.date_range("2023-01-01", periods=2, freq="D")
    close_df = pd.DataFrame({"X": [100.0, 130.0]}, index=dates)
    rows = [
        _ledger_row(0, dates[0], "X", 10.0, 100.0, 0.0, 10.0),
        _ledger_row(0, dates[0], "X", 10.0, 120.0, 10.0, 20.0),
        _ledger_row(1, dates[1], "X", -20.0, 130.0, 20.0, 0.0),
    ]
    ledger = pd.DataFrame(rows)
    pv = pd.Series(100_000.0, index=dates)
    pos = pd.DataFrame({"X": [20.0, 0.0]}, index=dates)
    out = build_canonical_trade_analytics(ledger, pv, pos, close_df)
    done = out["completed_trades"]
    assert len(done) == 1
    assert done.iloc[0]["entry_price"] == pytest.approx(110.0)
    path = close_df["X"].to_numpy(dtype=float, copy=False)
    mfe_ps, mae_ps = _expected_per_share_mfe_mae_long(path, 110.0)
    assert float(done.iloc[0]["mfe"]) == pytest.approx(mfe_ps * 20.0)
    assert float(done.iloc[0]["mae"]) == pytest.approx(mae_ps * 20.0)


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
