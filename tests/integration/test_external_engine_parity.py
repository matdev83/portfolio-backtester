"""Optional parity checks: canonical ``simulate_portfolio`` vs a Python reference and ``vectorbt``.

``bt`` is import-tested only; full backtest parity is not wired because ``bt``'s algo stack
does not mirror share/cash execution row-for-row without a bespoke emulation layer.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.backtester_logic.portfolio_simulation_input import (
    EXECUTION_TIMING_BAR_CLOSE,
    PortfolioSimulationInput,
    ledger_decision_idx_bar_close,
)
from portfolio_backtester.simulation.kernel import simulate_portfolio


def _approx_rtol() -> float:
    return 2e-3


def _zero_cost_global(portfolio_value: float) -> dict[str, float]:
    return {
        "portfolio_value": float(portfolio_value),
        "commission_per_share": 0.0,
        "commission_min_per_order": 0.0,
        "commission_max_percent_of_trade": 0.0,
        "slippage_bps": 0.0,
    }


def _per_trade_cost_dollars(
    trade_value: float,
    abs_share_delta: float,
    *,
    use_simple_bps: bool,
    transaction_costs_bps: float,
    commission_per_share: float,
    commission_min_per_order: float,
    commission_max_percent: float,
    slippage_bps: float,
    eps: float,
) -> float:
    if trade_value <= 0.0 or (not np.isfinite(trade_value)):
        return 0.0
    if use_simple_bps:
        return float(trade_value) * (transaction_costs_bps / 10000.0)
    if abs_share_delta <= 0.0:
        return 0.0
    commission_trade = abs_share_delta * commission_per_share
    if commission_trade < commission_min_per_order:
        commission_trade = commission_min_per_order
    max_commission = trade_value * commission_max_percent
    if commission_trade > max_commission:
        commission_trade = max_commission
    slippage_amount = trade_value * (slippage_bps / 10000.0)
    return float(commission_trade + slippage_amount)


def reference_bar_close_reinvest(
    initial_portfolio_value: float,
    weights: np.ndarray,
    close_prices: np.ndarray,
    close_price_mask: np.ndarray,
    execution_prices: np.ndarray,
    execution_price_mask: np.ndarray,
    rebalance_mask: np.ndarray,
    *,
    use_simple_bps: bool,
    transaction_costs_bps: float,
    commission_per_share: float,
    commission_min_per_order: float,
    commission_max_percent: float,
    slippage_bps: float,
    eps: float = 1e-9,
) -> np.ndarray:
    """Python mirror of ``canonical_portfolio_simulation_kernel`` for ``bar_close`` reinvestment only."""

    weights = np.asarray(weights, dtype=np.float64)
    close_prices = np.asarray(close_prices, dtype=np.float64)
    close_price_mask = np.asarray(close_price_mask, dtype=np.bool_)
    execution_prices = np.asarray(execution_prices, dtype=np.float64)
    execution_price_mask = np.asarray(execution_price_mask, dtype=np.bool_)
    rebalance_mask = np.asarray(rebalance_mask, dtype=np.bool_)
    t, n = weights.shape
    if close_prices.shape != (t, n):
        msg = "close_prices must match weights shape"
        raise ValueError(msg)

    portfolio_values = np.zeros(t, dtype=np.float64)
    cash_values = np.zeros(t, dtype=np.float64)
    positions = np.zeros((t, n), dtype=np.float64)

    last_valid_close = np.zeros(n, dtype=np.float64)
    for j in range(n):
        if close_price_mask[0, j]:
            last_valid_close[j] = close_prices[0, j]

    do_rebalance_0 = bool(rebalance_mask[0])
    if not do_rebalance_0:
        for j in range(n):
            if abs(weights[0, j]) > eps:
                do_rebalance_0 = True
                break

    last_positions0 = np.zeros(n, dtype=np.float64)
    positions0 = np.zeros(n, dtype=np.float64)
    day0_cost_dollars = 0.0
    exec_cash_flow_0 = 0.0

    if do_rebalance_0:
        capital_base0 = initial_portfolio_value
        tdv0 = weights[0] * capital_base0
        for j in range(n):
            if execution_price_mask[0, j] and execution_prices[0, j] > 0.0:
                positions0[j] = tdv0[j] / execution_prices[0, j]
            else:
                positions0[j] = last_positions0[j]
        running_cash0 = initial_portfolio_value
        for j in range(n):
            if not execution_price_mask[0, j]:
                continue
            exc_price = float(execution_prices[0, j])
            if exc_price <= 0.0:
                continue
            dsh0 = float(positions0[j] - last_positions0[j])
            exec_cash_flow_0 += dsh0 * exc_price
            td = abs(dsh0) * exc_price
            c0 = _per_trade_cost_dollars(
                float(td),
                float(abs(dsh0)),
                use_simple_bps=use_simple_bps,
                transaction_costs_bps=transaction_costs_bps,
                commission_per_share=commission_per_share,
                commission_min_per_order=commission_min_per_order,
                commission_max_percent=commission_max_percent,
                slippage_bps=slippage_bps,
                eps=eps,
            )
            day0_cost_dollars += c0
            cash_after0 = running_cash0 - (dsh0 * exc_price) - c0
            running_cash0 = cash_after0

        holdings_value0 = float(np.sum(positions0 * last_valid_close))
        cash_values[0] = float(initial_portfolio_value - exec_cash_flow_0 - day0_cost_dollars)
        portfolio_values[0] = float(cash_values[0] + holdings_value0)
        positions[0] = positions0
    else:
        cash_values[0] = initial_portfolio_value
        portfolio_values[0] = initial_portfolio_value
        positions[0] = positions0

    for i in range(1, t):
        close_row = close_prices[i]
        close_row_mask = close_price_mask[i]
        for j in range(n):
            if close_row_mask[j]:
                last_valid_close[j] = close_row[j]

        last_positions = positions[i - 1].copy()
        holdings_mark = float(np.sum(last_positions * last_valid_close))
        portfolio_values[i] = float(cash_values[i - 1] + holdings_mark)
        positions[i] = last_positions.copy()
        cash_values[i] = float(cash_values[i - 1])

        if not rebalance_mask[i]:
            continue

        exec_row = execution_prices[i]
        exec_row_mask = execution_price_mask[i]

        capital_base_use = float(portfolio_values[i])
        target_dollar_vals = weights[i] * capital_base_use
        target_positions = np.empty(n, dtype=np.float64)
        for j in range(n):
            if exec_row_mask[j] and exec_row[j] > 0.0:
                target_positions[j] = target_dollar_vals[j] / exec_row[j]
            else:
                target_positions[j] = last_positions[j]

        exec_cash_flow = 0.0
        day_cost_dollars = 0.0
        running_cash = float(cash_values[i - 1])
        for j in range(n):
            if not exec_row_mask[j]:
                continue
            exc_price = float(exec_row[j])
            if exc_price <= 0.0:
                continue
            dsh = float(target_positions[j] - last_positions[j])
            exec_cash_flow += dsh * exc_price
            td = abs(dsh) * exc_price
            cday = _per_trade_cost_dollars(
                float(td),
                float(abs(dsh)),
                use_simple_bps=use_simple_bps,
                transaction_costs_bps=transaction_costs_bps,
                commission_per_share=commission_per_share,
                commission_min_per_order=commission_min_per_order,
                commission_max_percent=commission_max_percent,
                slippage_bps=slippage_bps,
                eps=eps,
            )
            day_cost_dollars += cday
            cash_after_ev = running_cash - (dsh * exc_price) - cday
            running_cash = cash_after_ev

        positions[i] = target_positions.copy()
        new_holdings = float(np.sum(positions[i] * last_valid_close))
        cash_values[i] = float(cash_values[i - 1] - exec_cash_flow - day_cost_dollars)
        portfolio_values[i] = float(cash_values[i] + new_holdings)

    return portfolio_values


def _run_canonical(
    *,
    dates: pd.DatetimeIndex,
    tickers: tuple[str, ...],
    weights: np.ndarray,
    close: np.ndarray,
    rebalance_mask: np.ndarray,
    global_config: dict[str, float],
    scenario_config: dict[str, Any] | None = None,
) -> pd.Series:
    t, n = close.shape
    if len(tickers) != n:
        msg = "tickers length must match columns"
        raise ValueError(msg)
    m = np.ones_like(close, dtype=bool)
    sim_in = PortfolioSimulationInput(
        dates=dates,
        tickers=tickers,
        weights_target=np.asarray(weights, dtype=np.float64),
        close_prices=np.asarray(close, dtype=np.float64),
        close_price_mask=m,
        execution_prices=np.asarray(close, dtype=np.float64).copy(),
        execution_price_mask=m.copy(),
        rebalance_mask=np.asarray(rebalance_mask, dtype=np.bool_),
        execution_timing=EXECUTION_TIMING_BAR_CLOSE,
        ledger_decision_idx=ledger_decision_idx_bar_close(n_rows=int(t), n_assets=int(n)),
    )
    out = simulate_portfolio(sim_in, global_config=global_config, scenario_config=scenario_config)
    return out.portfolio_values


def _synthetic_close(periods: int) -> tuple[pd.DatetimeIndex, np.ndarray]:
    dates = pd.bdate_range("2020-01-02", periods=int(periods), freq="B")
    t = len(dates)
    i = np.arange(t, dtype=np.float64)
    col0 = 100.0 * np.power(1.0015, i)
    col1 = 50.0 * np.power(1.0010, i)
    close = np.column_stack([col0, col1])
    return dates, close


@pytest.mark.external_parity
def test_buy_hold_matches_numpy_reference() -> None:
    dates, close = _synthetic_close(80)
    t, _n = close.shape
    weights = np.tile(np.array([0.6, 0.4], dtype=np.float64), (t, 1))
    rebalance = np.zeros(t, dtype=bool)
    rebalance[0] = True
    initial = 100_000.0
    g = _zero_cost_global(initial)
    scen: dict[str, Any] = {"allocation_mode": "reinvestment"}
    canon = _run_canonical(
        dates=dates,
        tickers=("EQ0", "EQ1"),
        weights=weights,
        close=close,
        rebalance_mask=rebalance,
        global_config=g,
        scenario_config=scen,
    )
    m = np.ones_like(close, dtype=bool)
    ref = reference_bar_close_reinvest(
        initial,
        weights,
        close,
        m,
        close,
        m,
        rebalance,
        use_simple_bps=False,
        transaction_costs_bps=0.0,
        commission_per_share=0.0,
        commission_min_per_order=0.0,
        commission_max_percent=0.0,
        slippage_bps=0.0,
    )
    np.testing.assert_allclose(canon.to_numpy(dtype=float), ref, rtol=_approx_rtol(), atol=1e-3)


@pytest.mark.external_parity
def test_periodic_60_40_rebalance_matches_numpy_reference() -> None:
    dates, close = _synthetic_close(90)
    t, _n = close.shape
    weights = np.tile(np.array([0.6, 0.4], dtype=np.float64), (t, 1))
    rebalance = np.zeros(t, dtype=bool)
    rebalance[[0, 21, 42, 63]] = True
    initial = 250_000.0
    g = _zero_cost_global(initial)
    scen: dict[str, Any] = {"allocation_mode": "reinvestment"}
    canon = _run_canonical(
        dates=dates,
        tickers=("EQ0", "EQ1"),
        weights=weights,
        close=close,
        rebalance_mask=rebalance,
        global_config=g,
        scenario_config=scen,
    )
    m = np.ones_like(close, dtype=bool)
    ref = reference_bar_close_reinvest(
        initial,
        weights,
        close,
        m,
        close,
        m,
        rebalance,
        use_simple_bps=False,
        transaction_costs_bps=0.0,
        commission_per_share=0.0,
        commission_min_per_order=0.0,
        commission_max_percent=0.0,
        slippage_bps=0.0,
    )
    np.testing.assert_allclose(canon.to_numpy(dtype=float), ref, rtol=_approx_rtol(), atol=1e-3)


@pytest.mark.external_parity
def test_repeated_identical_targets_matches_numpy_reference() -> None:
    dates, close = _synthetic_close(72)
    t, _n = close.shape
    weights = np.tile(np.array([0.55, 0.45], dtype=np.float64), (t, 1))
    rebalance = np.zeros(t, dtype=bool)
    rebalance[0::9] = True
    initial = 80_000.0
    g = _zero_cost_global(initial)
    scen: dict[str, Any] = {"allocation_mode": "reinvestment"}
    canon = _run_canonical(
        dates=dates,
        tickers=("EQ0", "EQ1"),
        weights=weights,
        close=close,
        rebalance_mask=rebalance,
        global_config=g,
        scenario_config=scen,
    )
    m = np.ones_like(close, dtype=bool)
    ref = reference_bar_close_reinvest(
        initial,
        weights,
        close,
        m,
        close,
        m,
        rebalance,
        use_simple_bps=False,
        transaction_costs_bps=0.0,
        commission_per_share=0.0,
        commission_min_per_order=0.0,
        commission_max_percent=0.0,
        slippage_bps=0.0,
    )
    np.testing.assert_allclose(canon.to_numpy(dtype=float), ref, rtol=_approx_rtol(), atol=1e-3)


@pytest.mark.external_parity
def test_fixed_bps_cost_matches_numpy_reference() -> None:
    dates, close = _synthetic_close(55)
    t, _n = close.shape
    weights = np.tile(np.array([0.5, 0.5], dtype=np.float64), (t, 1))
    rebalance = np.zeros(t, dtype=bool)
    rebalance[[0, 18, 36]] = True
    initial = 100_000.0
    g = _zero_cost_global(initial)
    scen: dict[str, Any] = {
        "allocation_mode": "reinvestment",
        "costs_config": {"transaction_costs_bps": 8.0},
    }
    canon = _run_canonical(
        dates=dates,
        tickers=("EQ0", "EQ1"),
        weights=weights,
        close=close,
        rebalance_mask=rebalance,
        global_config=g,
        scenario_config=scen,
    )
    m = np.ones_like(close, dtype=bool)
    ref = reference_bar_close_reinvest(
        initial,
        weights,
        close,
        m,
        close,
        m,
        rebalance,
        use_simple_bps=True,
        transaction_costs_bps=8.0,
        commission_per_share=0.0,
        commission_min_per_order=0.0,
        commission_max_percent=0.0,
        slippage_bps=0.0,
    )
    np.testing.assert_allclose(canon.to_numpy(dtype=float), ref, rtol=_approx_rtol(), atol=2e-2)


@pytest.mark.external_parity
def test_vectorbt_buy_hold_close_execution_when_available() -> None:
    pytest.importorskip("vectorbt")
    vbt = importlib.import_module("vectorbt")

    dates, close = _synthetic_close(75)
    tickers = ("EQ0", "EQ1")
    t = close.shape[0]
    weights = np.tile(np.array([0.6, 0.4], dtype=np.float64), (t, 1))
    rebalance = np.zeros(t, dtype=bool)
    rebalance[0] = True
    initial = 100_000.0

    close_df = pd.DataFrame(close, index=dates, columns=list(tickers))
    size = pd.DataFrame(np.nan, index=dates, columns=list(tickers))
    size.iloc[0] = weights[0].copy()

    fee = 0.0
    pf = vbt.Portfolio.from_orders(
        close_df,
        size,
        size_type="targetpercent",
        price=close_df,
        val_price=close_df,
        init_cash=float(initial),
        cash_sharing=True,
        group_by=True,
        call_seq="auto",
        fees=fee,
        freq="d",
    )
    vbt_values = pf.value().to_numpy(dtype=float)
    g = _zero_cost_global(initial)
    scen: dict[str, Any] = {"allocation_mode": "reinvestment"}
    canon = _run_canonical(
        dates=dates,
        tickers=tickers,
        weights=weights,
        close=close,
        rebalance_mask=rebalance,
        global_config=g,
        scenario_config=scen,
    )
    np.testing.assert_allclose(canon.to_numpy(dtype=float), vbt_values, rtol=1e-2, atol=2.0)


@pytest.mark.external_parity
def test_vectorbt_periodic_rebalance_with_fees_when_available() -> None:
    pytest.importorskip("vectorbt")
    vbt = importlib.import_module("vectorbt")

    dates, close = _synthetic_close(88)
    tickers = ("EQ0", "EQ1")
    t = close.shape[0]
    target = np.array([0.6, 0.4], dtype=np.float64)
    weights = np.tile(target, (t, 1))
    rebalance = np.zeros(t, dtype=bool)
    rebalance[[0, 22, 44, 66]] = True
    initial = 200_000.0
    bps = 6.0
    fee = float(bps / 10000.0)

    close_df = pd.DataFrame(close, index=dates, columns=list(tickers))
    size = pd.DataFrame(np.nan, index=dates, columns=list(tickers))
    for i in np.flatnonzero(rebalance):
        size.iloc[i] = target.copy()

    pf = vbt.Portfolio.from_orders(
        close_df,
        size,
        size_type="targetpercent",
        price=close_df,
        val_price=close_df,
        init_cash=float(initial),
        cash_sharing=True,
        group_by=True,
        call_seq="auto",
        fees=fee,
        freq="d",
    )
    vbt_values = pf.value().to_numpy(dtype=float)
    g = _zero_cost_global(initial)
    scen: dict[str, Any] = {
        "allocation_mode": "reinvestment",
        "costs_config": {"transaction_costs_bps": bps},
    }
    canon = _run_canonical(
        dates=dates,
        tickers=tickers,
        weights=weights,
        close=close,
        rebalance_mask=rebalance,
        global_config=g,
        scenario_config=scen,
    )
    np.testing.assert_allclose(canon.to_numpy(dtype=float), vbt_values, rtol=2e-2, atol=5.0)


@pytest.mark.external_parity
def test_bt_importable_smoke_when_available() -> None:
    pytest.importorskip("bt", reason="optional bt not installed")
    bt = importlib.import_module("bt")

    assert hasattr(bt, "Backtest")
    assert hasattr(bt, "AlgoStack")
