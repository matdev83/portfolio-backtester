from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.backtester_logic.portfolio_simulation_input import (
    EXECUTION_TIMING_NEXT_BAR_OPEN,
    PortfolioSimulationInput,
    build_portfolio_simulation_input,
    ledger_decision_idx_next_bar_open_legacy,
)
from portfolio_backtester.simulation.kernel import simulate_portfolio


def _scalar_float(value: object) -> float:
    return float(np.asarray(value).item())


def _global_cfg(pv: float = 10_000.0) -> dict:
    return {
        "portfolio_value": pv,
        "commission_per_share": 0.0,
        "commission_min_per_order": 0.0,
        "commission_max_percent_of_trade": 0.0,
        "slippage_bps": 0.0,
    }


def test_next_bar_open_cash_nav_use_exec_price_for_trade_close_for_valuation():
    dates = pd.date_range("2023-01-01", periods=3, freq="D")
    w = np.array([[0.0], [1.0], [1.0]], dtype=np.float64)
    rb = np.array([False, True, False], dtype=np.bool_)
    close = np.array([[100.0], [100.0], [100.0]], dtype=np.float64)
    open_ = np.array([[100.0], [50.0], [100.0]], dtype=np.float64)
    m = np.ones_like(close, dtype=bool)
    sim_in = PortfolioSimulationInput(
        dates=dates,
        tickers=("X",),
        weights_target=w,
        close_prices=close,
        close_price_mask=m,
        execution_prices=open_,
        execution_price_mask=m.copy(),
        rebalance_mask=rb,
        execution_timing=EXECUTION_TIMING_NEXT_BAR_OPEN,
        ledger_decision_idx=ledger_decision_idx_next_bar_open_legacy(
            n_rows=w.shape[0], n_assets=w.shape[1]
        ),
    )
    pv0 = 100_000.0
    out = simulate_portfolio(
        sim_in,
        global_config=_global_cfg(pv0),
        scenario_config={"allocation_mode": "reinvestment"},
    )
    exec_ix = 1
    assert _scalar_float(out.positions.iloc[exec_ix, 0]) == pytest.approx(2000.0, rel=0.0, abs=1e-6)
    assert _scalar_float(out.cash.iloc[exec_ix]) == pytest.approx(0.0, abs=1e-5)
    assert _scalar_float(out.portfolio_values.iloc[exec_ix]) == pytest.approx(
        200_000.0, rel=0.0, abs=1e-3
    )
    assert _scalar_float(out.daily_returns.iloc[exec_ix]) == pytest.approx(1.0, rel=0.0, abs=1e-6)


def test_next_bar_open_rebalance_sizes_from_execution_nav_not_same_day_close():
    cal = pd.date_range("2023-01-01", periods=2, freq="D")
    wd = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=cal, columns=["A", "B"])
    close = np.array([[100.0, 50.0], [200.0, 100.0]], dtype=np.float64)
    open_ = np.array([[100.0, 50.0], [50.0, 100.0]], dtype=np.float64)
    mk = np.ones_like(close, dtype=bool)
    rb = np.ones(len(cal), dtype=bool)
    inp = build_portfolio_simulation_input(
        weights_daily=wd,
        price_index=cal,
        valid_cols=["A", "B"],
        close_arr=close,
        close_price_mask_arr=mk,
        open_arr=open_,
        open_price_mask_arr=mk,
        rebalance_mask_arr=rb,
        trade_execution_timing="next_bar_open",
    )
    out = simulate_portfolio(
        inp,
        global_config=_global_cfg(10_000.0),
        scenario_config={"allocation_mode": "reinvestment"},
    )
    exp_sh_b = (100.0 * 50.0) / 100.0
    assert _scalar_float(out.positions.iloc[1, 1]) == pytest.approx(exp_sh_b, rel=0.0, abs=1e-6)


def test_next_bar_open_zero_target_exit_retries_when_open_invalid_then_valid():
    cal = pd.date_range("2023-01-01", periods=3, freq="D")
    wd = pd.DataFrame({"A": [1.0, 0.0, 0.0]}, index=cal)
    close = np.full((3, 1), 100.0)
    close_mk = np.ones_like(close, dtype=bool)
    open_ = np.full((3, 1), 100.0)
    open_mk = np.array([[True], [False], [True]], dtype=bool)
    rb = np.array([True, True, False], dtype=bool)
    inp = build_portfolio_simulation_input(
        weights_daily=wd,
        price_index=cal,
        valid_cols=["A"],
        close_arr=close,
        close_price_mask_arr=close_mk,
        open_arr=open_,
        open_price_mask_arr=open_mk,
        rebalance_mask_arr=rb,
        trade_execution_timing="next_bar_open",
    )
    assert inp.rebalance_mask[1] and inp.rebalance_mask[2]
    out = simulate_portfolio(
        inp,
        global_config=_global_cfg(10_000.0),
        scenario_config={"allocation_mode": "reinvestment"},
    )
    assert _scalar_float(out.positions.iloc[2, 0]) == pytest.approx(0.0, abs=1e-6)


def test_next_bar_open_zero_exit_retries_two_invalid_opens_then_valid():
    cal = pd.date_range("2023-01-01", periods=4, freq="D")
    wd = pd.DataFrame({"A": [1.0, 0.0, 0.0, 0.0]}, index=cal)
    close = np.full((4, 1), 100.0)
    close_mk = np.ones_like(close, dtype=bool)
    open_ = np.full((4, 1), 100.0)
    open_mk = np.array([[True], [False], [False], [True]], dtype=bool)
    rb = np.array([True, True, False, False], dtype=bool)
    inp = build_portfolio_simulation_input(
        weights_daily=wd,
        price_index=cal,
        valid_cols=["A"],
        close_arr=close,
        close_price_mask_arr=close_mk,
        open_arr=open_,
        open_price_mask_arr=open_mk,
        rebalance_mask_arr=rb,
        trade_execution_timing="next_bar_open",
    )
    assert inp.rebalance_mask[1] and inp.rebalance_mask[2] and inp.rebalance_mask[3]
    assert int(inp.ledger_decision_idx[3, 0]) == 0
    out = simulate_portfolio(
        inp,
        global_config=_global_cfg(10_000.0),
        scenario_config={"allocation_mode": "reinvestment"},
    )
    assert _scalar_float(out.positions.iloc[-1, 0]) == pytest.approx(0.0, abs=1e-6)
    exit_rows = out.execution_ledger[out.execution_ledger["quantity"] < 0]
    assert len(exit_rows) == 1
    row = exit_rows.iloc[0]
    assert int(row["execution_date_idx"]) == 3
    assert int(row["decision_date_idx"]) == 0
    assert pd.Timestamp(row["execution_date"]) == cal[3]
    assert pd.Timestamp(row["decision_date"]) == cal[0]
