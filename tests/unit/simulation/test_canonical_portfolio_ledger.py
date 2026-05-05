from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.backtester_logic.portfolio_simulation_input import (
    EXECUTION_TIMING_BAR_CLOSE,
    EXECUTION_TIMING_NEXT_BAR_OPEN,
    PortfolioSimulationInput,
)
from portfolio_backtester.simulation.kernel import simulate_portfolio


def _simple_scenario(transaction_costs_bps: float | None = None) -> tuple[dict, dict]:
    g = {
        "portfolio_value": 10_000.0,
        "commission_per_share": 0.0,
        "commission_min_per_order": 0.0,
        "commission_max_percent_of_trade": 0.0,
        "slippage_bps": 0.0,
    }
    sc: dict = {"allocation_mode": "reinvestment"}
    if transaction_costs_bps is not None:
        sc["costs_config"] = {"transaction_costs_bps": float(transaction_costs_bps)}
    return g, sc


def test_constant_targets_rebalances_only_when_mask_fire_after_drift():
    dates = pd.date_range("2023-01-01", periods=3, freq="D")
    w = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]}, index=dates).to_numpy(
        dtype=np.float64
    )
    rb = np.array([True, False, True], dtype=np.bool_)
    close = np.array(
        [
            [80.0, 120.0],
            [160.0, 120.0],
            [160.0, 120.0],
        ],
        dtype=np.float64,
    )
    m = np.ones_like(close, dtype=bool)
    sim_in = PortfolioSimulationInput(
        dates=dates,
        tickers=("A", "B"),
        weights_target=w,
        close_prices=close,
        close_price_mask=m,
        execution_prices=close.copy(),
        execution_price_mask=m.copy(),
        rebalance_mask=rb,
        execution_timing=EXECUTION_TIMING_BAR_CLOSE,
    )
    g, sc = _simple_scenario(None)
    out = simulate_portfolio(sim_in, global_config=g, scenario_config=sc)
    pv = out.portfolio_values.to_numpy(dtype=np.float64)
    assert pv[1] == pytest.approx(15_000.0, abs=1e-6)

    tgt_d = 7500.0
    tgt_a = tgt_d / 160.0
    tgt_b = tgt_d / 120.0
    assert out.positions.iloc[2, 0] == pytest.approx(tgt_a, abs=1e-6)
    assert out.positions.iloc[2, 1] == pytest.approx(tgt_b, abs=1e-6)


def test_same_targets_no_mask_no_extra_trades_or_costs_vs_hold():
    dates = pd.date_range("2023-01-01", periods=3, freq="D")
    w = np.ones((3, 1), dtype=np.float64)
    rb = np.array([True, False, False], dtype=np.bool_)
    close = np.array([[100.0], [200.0], [200.0]], dtype=np.float64)
    m = np.ones_like(close, dtype=bool)

    sim_in = PortfolioSimulationInput(
        dates=dates,
        tickers=("A",),
        weights_target=w,
        close_prices=close,
        close_price_mask=m,
        execution_prices=close.copy(),
        execution_price_mask=m.copy(),
        rebalance_mask=rb,
        execution_timing=EXECUTION_TIMING_BAR_CLOSE,
    )
    g, sc = _simple_scenario(100.0)
    out = simulate_portfolio(sim_in, global_config=g, scenario_config=sc)
    assert float(
        np.sum(out.per_asset_transaction_cost_frac_of_reference_pv[1:, :])
    ) == pytest.approx(0.0, abs=1e-12)


def test_day_zero_bps_reduces_daily_return_and_nav():
    dates = pd.date_range("2023-01-01", periods=2, freq="D")
    w = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    rb = np.ones(2, dtype=np.bool_)
    close = np.array([[100.0, 40.0], [100.0, 40.0]], dtype=np.float64)
    m = np.ones_like(close, dtype=bool)

    sim_in = PortfolioSimulationInput(
        dates=dates,
        tickers=("A", "B"),
        weights_target=w,
        close_prices=close,
        close_price_mask=m,
        execution_prices=close.copy(),
        execution_price_mask=m.copy(),
        rebalance_mask=rb,
        execution_timing=EXECUTION_TIMING_BAR_CLOSE,
    )
    g, sc = _simple_scenario(10.0)
    ref = float(g["portfolio_value"])
    exp_cost = ref * (10.0 / 10000.0)
    expected_pv0 = ref - exp_cost
    out = simulate_portfolio(sim_in, global_config=g, scenario_config=sc)
    assert out.portfolio_values.iloc[0] == pytest.approx(expected_pv0, abs=1e-6)
    assert out.daily_returns.iloc[0] == pytest.approx(expected_pv0 / ref - 1.0, abs=1e-9)


def test_missing_execution_first_rebalance_skips_then_entries_when_valid():
    dates = pd.date_range("2023-01-01", periods=3, freq="D")
    w = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    rb = np.ones(3, dtype=np.bool_)
    close = np.array(
        [
            [100.0, 50.0],
            [100.0, 50.0],
            [100.0, 50.0],
        ],
        dtype=np.float64,
    )
    m = np.ones_like(close, dtype=bool)
    exec_m = np.array(
        [
            [False, True],
            [True, True],
            [True, True],
        ],
        dtype=bool,
    )
    sim_in = PortfolioSimulationInput(
        dates=dates,
        tickers=("A", "B"),
        weights_target=w,
        close_prices=close,
        close_price_mask=m,
        execution_prices=close.copy(),
        execution_price_mask=exec_m,
        rebalance_mask=rb,
        execution_timing=EXECUTION_TIMING_BAR_CLOSE,
    )
    g, sc = _simple_scenario(None)
    out = simulate_portfolio(sim_in, global_config=g, scenario_config=sc)
    assert out.positions.iloc[0, 0] == 0.0
    assert out.positions.iloc[1, 0] > 0.0


def test_bar_close_vs_next_bar_open_differs_when_open_not_equal_close():
    dates = pd.date_range("2023-01-01", periods=2, freq="D")
    w = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    rb = np.ones(2, dtype=np.bool_)
    close = np.array([[100.0, 50.0], [100.0, 50.0]], dtype=np.float64)
    open_ = np.array([[100.0, 50.0], [90.0, 55.0]], dtype=np.float64)
    m = np.ones_like(close, dtype=bool)

    sim_close = PortfolioSimulationInput(
        dates=dates,
        tickers=("A", "B"),
        weights_target=w,
        close_prices=close,
        close_price_mask=m,
        execution_prices=close.copy(),
        execution_price_mask=m.copy(),
        rebalance_mask=rb,
        execution_timing=EXECUTION_TIMING_BAR_CLOSE,
    )
    sim_open = PortfolioSimulationInput(
        dates=dates,
        tickers=("A", "B"),
        weights_target=w,
        close_prices=close,
        close_price_mask=m,
        execution_prices=open_.copy(),
        execution_price_mask=m.copy(),
        rebalance_mask=rb,
        execution_timing=EXECUTION_TIMING_NEXT_BAR_OPEN,
    )
    g, sc = _simple_scenario(None)
    out_bc = simulate_portfolio(sim_close, global_config=g, scenario_config=sc)
    out_bo = simulate_portfolio(sim_open, global_config=g, scenario_config=sc)
    pv_bar_close_d1 = float(g["portfolio_value"])
    exp_bc_b = pv_bar_close_d1 / 50.0
    np.testing.assert_allclose(
        out_bc.positions.iloc[1].to_numpy(dtype=np.float64),
        np.array([0.0, exp_bc_b], dtype=np.float64),
        rtol=0.0,
        atol=1e-6,
    )
    exp_bo_b = (100.0 * float(open_[1, 0])) / float(open_[1, 1])
    np.testing.assert_allclose(
        out_bo.positions.iloc[1].to_numpy(dtype=np.float64),
        np.array([0.0, exp_bo_b], dtype=np.float64),
        rtol=0.0,
        atol=1e-6,
    )
    assert not np.allclose(
        out_bc.positions.iloc[1].to_numpy(dtype=np.float64),
        out_bo.positions.iloc[1].to_numpy(dtype=np.float64),
    )


def test_detailed_commission_slippage_hand_calculated_row():
    dates = pd.date_range("2023-01-01", periods=2, freq="D")
    w = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    rb = np.array([True, False], dtype=np.bool_)
    close = np.array([[100.0, 20.0], [100.0, 20.0]], dtype=np.float64)
    m = np.ones_like(close, dtype=bool)
    sim_in = PortfolioSimulationInput(
        dates=dates,
        tickers=("A", "B"),
        weights_target=w,
        close_prices=close,
        close_price_mask=m,
        execution_prices=close.copy(),
        execution_price_mask=m.copy(),
        rebalance_mask=rb,
        execution_timing=EXECUTION_TIMING_BAR_CLOSE,
    )
    g = {
        "portfolio_value": 10_000.0,
        "commission_per_share": 0.005,
        "commission_min_per_order": 1.0,
        "commission_max_percent_of_trade": 1.0,
        "slippage_bps": 25.0,
    }
    sc = {"allocation_mode": "reinvestment"}
    out = simulate_portfolio(sim_in, global_config=g, scenario_config=sc)
    td = abs(100.0 * 100.0)
    comm_trade = abs(100.0) * 0.005
    comm_trade = max(comm_trade, 1.0)
    comm_trade = min(comm_trade, td * 1.0)
    slip = td * (25.0 / 10000.0)
    c0 = float(comm_trade + slip)
    assert out.per_asset_transaction_cost_frac_of_reference_pv[0, 0] == pytest.approx(
        c0 / float(g["portfolio_value"]), abs=1e-9
    )
    cash = float(g["portfolio_value"]) - 100.0 * 100.0 - c0
    assert out.cash.iloc[0] == pytest.approx(cash, abs=1e-6)
    nav = cash + 100.0 * 100.0
    assert out.portfolio_values.iloc[0] == pytest.approx(nav, abs=1e-6)
    assert out.positions.iloc[0, 0] == pytest.approx(100.0, abs=1e-9)
    assert out.positions.iloc[0, 1] == pytest.approx(0.0, abs=1e-12)


def test_nav_matches_cash_plus_marked_positions_with_close_values():
    dates = pd.date_range("2023-01-01", periods=2, freq="D")
    w = np.array([[0.25, 0.75], [0.25, 0.75]], dtype=np.float64)
    rb = np.array([True, False], dtype=np.bool_)
    close = np.array([[10.0, 90.0], [20.0, 180.0]], dtype=np.float64)
    m = np.ones_like(close, dtype=bool)
    sim_in = PortfolioSimulationInput(
        dates=dates,
        tickers=("X", "Y"),
        weights_target=w,
        close_prices=close,
        close_price_mask=m,
        execution_prices=close.copy(),
        execution_price_mask=m.copy(),
        rebalance_mask=rb,
        execution_timing=EXECUTION_TIMING_BAR_CLOSE,
    )
    g, sc = _simple_scenario(None)
    out = simulate_portfolio(sim_in, global_config=g, scenario_config=sc)
    holdings = np.sum(np.asarray(out.positions.iloc[0]) * np.asarray(close[0]), dtype=float)
    assert float(out.cash.iloc[0]) + holdings == pytest.approx(
        float(out.portfolio_values.iloc[0]), abs=1e-5
    )
    close1 = np.asarray(close[1], dtype=float)
    pos_prev = np.asarray(out.positions.iloc[0])
    holdings1 = np.sum(pos_prev * close1)
    assert float(out.cash.iloc[0]) + holdings1 == pytest.approx(
        float(out.portfolio_values.iloc[1]), abs=1e-5
    )
