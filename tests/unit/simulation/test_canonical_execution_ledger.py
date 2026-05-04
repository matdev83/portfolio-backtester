from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.backtester_logic.portfolio_simulation_input import (
    EXECUTION_TIMING_BAR_CLOSE,
    EXECUTION_TIMING_NEXT_BAR_OPEN,
    PortfolioSimulationInput,
)
from portfolio_backtester.simulation.kernel import EXECUTION_LEDGER_COLUMNS, simulate_portfolio
from portfolio_backtester.trading.trade_tracker import TradeTracker


def _zero_cost_global(portfolio_value: float) -> dict:
    return {
        "portfolio_value": float(portfolio_value),
        "commission_per_share": 0.0,
        "commission_min_per_order": 0.0,
        "commission_max_percent_of_trade": 0.0,
        "slippage_bps": 0.0,
    }


def test_day_zero_bar_close_buy_emits_ledger_nav_unchanged():
    dates = pd.date_range("2023-01-01", periods=1, freq="D")
    w = np.array([[1.0]], dtype=np.float64)
    rb = np.array([True], dtype=np.bool_)
    close = np.array([[100.0]], dtype=np.float64)
    m = np.ones_like(close, dtype=bool)
    sim_in = PortfolioSimulationInput(
        dates=dates,
        tickers=("X",),
        weights_target=w,
        close_prices=close,
        close_price_mask=m,
        execution_prices=close.copy(),
        execution_price_mask=m.copy(),
        rebalance_mask=rb,
        execution_timing=EXECUTION_TIMING_BAR_CLOSE,
    )
    g = _zero_cost_global(100_000.0)
    sc: dict = {"allocation_mode": "reinvestment"}
    out = simulate_portfolio(sim_in, global_config=g, scenario_config=sc)
    assert list(out.execution_ledger.columns) == list(EXECUTION_LEDGER_COLUMNS)
    assert len(out.execution_ledger) == 1
    row = out.execution_ledger.iloc[0]
    assert int(row["date_idx"]) == 0
    assert pd.Timestamp(row["date"]) == dates[0]
    assert str(row["ticker"]) == "X"
    assert float(row["quantity"]) == pytest.approx(1000.0)
    assert float(row["execution_price"]) == pytest.approx(100.0)
    assert float(row["execution_value"]) == pytest.approx(100_000.0)
    assert float(row["cash_before"]) == pytest.approx(100_000.0)
    assert float(row["cash_after"]) == pytest.approx(0.0)
    assert float(row["position_before"]) == pytest.approx(0.0)
    assert float(row["position_after"]) == pytest.approx(1000.0)
    assert float(row["cost"]) == pytest.approx(0.0)
    assert float(out.portfolio_values.iloc[0]) == pytest.approx(100_000.0)


def test_next_bar_open_buy_at_open_then_close_nav():
    dates = pd.date_range("2023-01-01", periods=2, freq="D")
    w = np.array([[0.0], [1.0]], dtype=np.float64)
    rb = np.array([False, True], dtype=np.bool_)
    close = np.array([[100.0], [100.0]], dtype=np.float64)
    exec_px = np.array([[100.0], [50.0]], dtype=np.float64)
    m = np.ones_like(close, dtype=bool)
    sim_in = PortfolioSimulationInput(
        dates=dates,
        tickers=("X",),
        weights_target=w,
        close_prices=close,
        close_price_mask=m,
        execution_prices=exec_px,
        execution_price_mask=m.copy(),
        rebalance_mask=rb,
        execution_timing=EXECUTION_TIMING_NEXT_BAR_OPEN,
    )
    g = _zero_cost_global(100_000.0)
    sc: dict = {"allocation_mode": "reinvestment"}
    out = simulate_portfolio(sim_in, global_config=g, scenario_config=sc)
    assert list(out.execution_ledger.columns) == list(EXECUTION_LEDGER_COLUMNS)
    assert len(out.execution_ledger) == 1
    row = out.execution_ledger.iloc[0]
    assert int(row["date_idx"]) == 1
    assert float(row["quantity"]) == pytest.approx(2000.0)
    assert float(row["execution_price"]) == pytest.approx(50.0)
    assert float(row["cash_after"]) == pytest.approx(0.0)
    assert float(out.portfolio_values.iloc[1]) == pytest.approx(200_000.0)


def test_repeated_equal_targets_masked_rebalances_emit_day_two_ledger_rows():
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
    g = _zero_cost_global(10_000.0)
    sc: dict = {"allocation_mode": "reinvestment"}
    out = simulate_portfolio(sim_in, global_config=g, scenario_config=sc)
    d2 = out.execution_ledger[out.execution_ledger["date_idx"] == 2]
    assert len(d2) >= 2


def test_partial_liquidation_ledger_sell_same_price_nav_coherent():
    dates = pd.date_range("2023-01-01", periods=2, freq="D")
    w = np.array([[1.0], [0.4]], dtype=np.float64)
    rb = np.array([True, True], dtype=np.bool_)
    close = np.array([[100.0], [100.0]], dtype=np.float64)
    m = np.ones_like(close, dtype=bool)
    sim_in = PortfolioSimulationInput(
        dates=dates,
        tickers=("X",),
        weights_target=w,
        close_prices=close,
        close_price_mask=m,
        execution_prices=close.copy(),
        execution_price_mask=m.copy(),
        rebalance_mask=rb,
        execution_timing=EXECUTION_TIMING_BAR_CLOSE,
    )
    g = _zero_cost_global(100_000.0)
    sc: dict = {"allocation_mode": "reinvestment"}
    out = simulate_portfolio(sim_in, global_config=g, scenario_config=sc)
    assert len(out.execution_ledger) == 2
    sell = out.execution_ledger[out.execution_ledger["date_idx"] == 1].iloc[0]
    assert float(sell["quantity"]) == pytest.approx(-600.0)
    assert float(sell["execution_price"]) == pytest.approx(100.0)
    assert float(sell["execution_value"]) == pytest.approx(-60_000.0)
    assert float(sell["position_before"]) == pytest.approx(1000.0)
    assert float(sell["position_after"]) == pytest.approx(400.0)
    assert float(sell["cash_before"]) == pytest.approx(0.0)
    assert float(sell["cash_after"]) == pytest.approx(60_000.0)
    assert float(sell["cost"]) == pytest.approx(0.0)
    pv1 = float(out.portfolio_values.iloc[1])
    cash1 = float(out.cash.iloc[1])
    pos1 = float(out.positions.to_numpy(dtype=np.float64, copy=False)[1, 0])
    assert pos1 == pytest.approx(400.0)
    assert cash1 + pos1 * 100.0 == pytest.approx(pv1)
    assert pv1 == pytest.approx(100_000.0)


def test_full_liquidation_ledger_closes_to_cash_matches_nav():
    dates = pd.date_range("2023-01-01", periods=2, freq="D")
    w = np.array([[1.0], [0.0]], dtype=np.float64)
    rb = np.array([True, True], dtype=np.bool_)
    close = np.array([[100.0], [100.0]], dtype=np.float64)
    m = np.ones_like(close, dtype=bool)
    sim_in = PortfolioSimulationInput(
        dates=dates,
        tickers=("X",),
        weights_target=w,
        close_prices=close,
        close_price_mask=m,
        execution_prices=close.copy(),
        execution_price_mask=m.copy(),
        rebalance_mask=rb,
        execution_timing=EXECUTION_TIMING_BAR_CLOSE,
    )
    g = _zero_cost_global(50_000.0)
    sc: dict = {"allocation_mode": "reinvestment"}
    out = simulate_portfolio(sim_in, global_config=g, scenario_config=sc)
    sell = out.execution_ledger[out.execution_ledger["date_idx"] == 1].iloc[0]
    assert float(sell["quantity"]) == pytest.approx(-500.0)
    assert float(sell["position_after"]) == pytest.approx(0.0)
    assert float(sell["cash_after"]) == pytest.approx(50_000.0)
    assert float(out.positions.to_numpy(dtype=np.float64, copy=False)[1, 0]) == pytest.approx(0.0)
    pv1 = float(out.portfolio_values.iloc[1])
    assert float(out.cash.iloc[1]) == pytest.approx(pv1)
    assert pv1 == pytest.approx(50_000.0)


def test_long_short_flip_single_row_signed_quantity_nav_coherent():
    dates = pd.date_range("2023-01-01", periods=2, freq="D")
    w = np.array([[1.0], [-1.0]], dtype=np.float64)
    rb = np.array([True, True], dtype=np.bool_)
    close = np.array([[100.0], [100.0]], dtype=np.float64)
    m = np.ones_like(close, dtype=bool)
    sim_in = PortfolioSimulationInput(
        dates=dates,
        tickers=("X",),
        weights_target=w,
        close_prices=close,
        close_price_mask=m,
        execution_prices=close.copy(),
        execution_price_mask=m.copy(),
        rebalance_mask=rb,
        execution_timing=EXECUTION_TIMING_BAR_CLOSE,
    )
    g = _zero_cost_global(100_000.0)
    sc: dict = {"allocation_mode": "reinvestment"}
    out = simulate_portfolio(sim_in, global_config=g, scenario_config=sc)
    flip = out.execution_ledger[out.execution_ledger["date_idx"] == 1].iloc[0]
    assert float(flip["quantity"]) == pytest.approx(-2000.0)
    assert float(flip["position_before"]) == pytest.approx(1000.0)
    assert float(flip["position_after"]) == pytest.approx(-1000.0)
    assert float(flip["execution_value"]) == pytest.approx(-200_000.0)
    pv1 = float(out.portfolio_values.iloc[1])
    cash1 = float(out.cash.iloc[1])
    pos1 = float(out.positions.to_numpy(dtype=np.float64, copy=False)[1, 0])
    assert cash1 + pos1 * 100.0 == pytest.approx(pv1)
    assert pv1 == pytest.approx(100_000.0)


def test_day_zero_simple_bps_cost_ledger_matches_fractions_and_return():
    dates = pd.date_range("2023-01-01", periods=1, freq="D")
    w = np.array([[1.0]], dtype=np.float64)
    rb = np.array([True], dtype=np.bool_)
    close = np.array([[100.0]], dtype=np.float64)
    m = np.ones_like(close, dtype=bool)
    sim_in = PortfolioSimulationInput(
        dates=dates,
        tickers=("X",),
        weights_target=w,
        close_prices=close,
        close_price_mask=m,
        execution_prices=close.copy(),
        execution_price_mask=m.copy(),
        rebalance_mask=rb,
        execution_timing=EXECUTION_TIMING_BAR_CLOSE,
    )
    ref = 100_000.0
    g = _zero_cost_global(ref)
    bps = 10.0
    trade_notional = ref
    expected_cost = trade_notional * (bps / 10_000.0)
    sc: dict = {"allocation_mode": "reinvestment", "costs_config": {"transaction_costs_bps": bps}}
    out = simulate_portfolio(sim_in, global_config=g, scenario_config=sc)
    row = out.execution_ledger.iloc[0]
    assert float(row["cost"]) == pytest.approx(expected_cost)
    assert float(row["cash_after"]) == pytest.approx(-expected_cost)
    assert float(out.portfolio_values.iloc[0]) == pytest.approx(ref - expected_cost)
    assert float(out.total_cost_fraction[0]) == pytest.approx(expected_cost / ref)
    assert float(out.per_asset_cost_fraction[0, 0]) == pytest.approx(expected_cost / ref)
    assert float(out.daily_returns.iloc[0]) == pytest.approx((ref - expected_cost) / ref - 1.0)


def test_day_zero_detailed_cost_ledger_matches_per_asset_cost_fraction():
    dates = pd.date_range("2023-01-01", periods=1, freq="D")
    w = np.array([[1.0]], dtype=np.float64)
    rb = np.array([True], dtype=np.bool_)
    close = np.array([[50.0]], dtype=np.float64)
    m = np.ones_like(close, dtype=bool)
    sim_in = PortfolioSimulationInput(
        dates=dates,
        tickers=("X",),
        weights_target=w,
        close_prices=close,
        close_price_mask=m,
        execution_prices=close.copy(),
        execution_price_mask=m.copy(),
        rebalance_mask=rb,
        execution_timing=EXECUTION_TIMING_BAR_CLOSE,
    )
    ref = 10_000.0
    g = {
        "portfolio_value": ref,
        "commission_per_share": 0.01,
        "commission_min_per_order": 0.0,
        "commission_max_percent_of_trade": 1.0,
        "slippage_bps": 10.0,
    }
    sc: dict = {"allocation_mode": "reinvestment"}
    out = simulate_portfolio(sim_in, global_config=g, scenario_config=sc)
    shares = ref / 50.0
    trade_value = ref
    commission = shares * 0.01
    slip = trade_value * (10.0 / 10_000.0)
    expected_cost = commission + slip
    row = out.execution_ledger.iloc[0]
    assert float(row["cost"]) == pytest.approx(expected_cost)
    assert float(row["cost"]) == pytest.approx(float(out.per_asset_cost_fraction[0, 0]) * ref)


def test_invalid_execution_masked_day_then_valid_open_emits_single_fill_row():
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
    g = _zero_cost_global(10_000.0)
    sc: dict = {"allocation_mode": "reinvestment"}
    out = simulate_portfolio(sim_in, global_config=g, scenario_config=sc)
    a_rows = out.execution_ledger[out.execution_ledger["ticker"] == "A"]
    assert len(a_rows[a_rows["date_idx"] == 0]) == 0
    assert len(a_rows[a_rows["date_idx"] == 1]) >= 1


def test_same_day_multi_asset_ledger_row_shuffle_trade_tracker_replay_invariant():
    """Kernel emits fills in column order; replay sorts by (date_idx, ticker) for determinism."""
    dates = pd.date_range("2023-01-01", periods=1, freq="D")
    w = np.array([[0.5, 0.5]], dtype=np.float64)
    rb = np.array([True], dtype=np.bool_)
    close = np.array([[100.0, 200.0]], dtype=np.float64)
    m = np.ones_like(close, dtype=bool)
    sim_in = PortfolioSimulationInput(
        dates=dates,
        tickers=("Z", "A"),
        weights_target=w,
        close_prices=close,
        close_price_mask=m,
        execution_prices=close.copy(),
        execution_price_mask=m.copy(),
        rebalance_mask=rb,
        execution_timing=EXECUTION_TIMING_BAR_CLOSE,
    )
    g = _zero_cost_global(10_000.0)
    sc: dict = {"allocation_mode": "reinvestment"}
    out = simulate_portfolio(sim_in, global_config=g, scenario_config=sc)
    raw = out.execution_ledger.copy()
    assert len(raw) == 2
    assert list(raw["ticker"]) == ["Z", "A"]
    shuffled = raw.iloc[[1, 0]].reset_index(drop=True)

    idx = out.portfolio_values.index
    close_df = pd.DataFrame(close, index=idx, columns=["Z", "A"])
    pv = out.portfolio_values
    pos = out.positions

    def _open_snapshot(tt: TradeTracker) -> dict[str, tuple[float, float]]:
        return {
            t: (float(tr.quantity), float(tr.entry_price))
            for t, tr in tt.trade_lifecycle_manager.get_open_positions().items()
        }

    tt1 = TradeTracker(10_000.0, "reinvestment")
    tt1.populate_from_execution_ledger(raw, pv, pos, close_df)
    tt2 = TradeTracker(10_000.0, "reinvestment")
    tt2.populate_from_execution_ledger(shuffled, pv, pos, close_df)
    assert _open_snapshot(tt1) == _open_snapshot(tt2)
    assert set(_open_snapshot(tt1).keys()) == {"A", "Z"}
