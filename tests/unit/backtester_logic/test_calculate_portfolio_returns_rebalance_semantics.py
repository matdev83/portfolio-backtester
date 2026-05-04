from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.backtester_logic.portfolio_logic import (
    _resolve_trade_execution_timing_for_portfolio,
    _sized_signals_to_weights_daily,
    _sparse_targets_after_time_based_rebalance,
    calculate_portfolio_returns,
)
from portfolio_backtester.backtester_logic.portfolio_simulation_input import (
    build_portfolio_simulation_input,
    extract_open_frame_from_ohlc,
    prepare_close_arrays_for_simulation,
    prepare_open_arrays_for_simulation,
)
from portfolio_backtester.simulation.kernel import SimulationResult, simulate_portfolio
from portfolio_backtester.timing.trade_execution_timing import (
    map_sparse_target_weights_to_execution_dates,
)


def _base_global(portfolio_value: float = 10_000.0) -> dict:
    return {
        "portfolio_value": portfolio_value,
        "commission_per_share": 0.0,
        "commission_min_per_order": 0.0,
        "commission_max_percent_of_trade": 0.0,
        "slippage_bps": 0.0,
    }


def _simulate_result(
    sized_signals: pd.DataFrame,
    scenario: dict,
    price_data_daily_ohlc: pd.DataFrame,
    universe_tickers: list[str],
    global_config: dict,
    *,
    strategy: object | None = None,
) -> SimulationResult:
    sized_for_timing = _sparse_targets_after_time_based_rebalance(sized_signals, scenario)
    tet = _resolve_trade_execution_timing_for_portfolio(scenario, strategy)
    execution_calendar = pd.DatetimeIndex(price_data_daily_ohlc.index)
    sized_for_daily = map_sparse_target_weights_to_execution_dates(
        sized_for_timing,
        trade_execution_timing=tet,
        calendar=execution_calendar,
    )
    weights_daily = _sized_signals_to_weights_daily(
        sized_for_daily, universe_tickers, price_data_daily_ohlc.index
    )
    if isinstance(price_data_daily_ohlc.columns, pd.MultiIndex) and (
        "Close" in price_data_daily_ohlc.columns.get_level_values(-1)
    ):
        close_prices_df = price_data_daily_ohlc.xs("Close", level="Field", axis=1)
        price_index = close_prices_df.index
    else:
        close_prices_df = price_data_daily_ohlc
        price_index = price_data_daily_ohlc.index

    valid_cols = [t for t in universe_tickers if t in close_prices_df.columns]
    price_ix = pd.DatetimeIndex(price_index)
    close_arr, close_mask_arr = prepare_close_arrays_for_simulation(
        market_data_panel=None,
        close_prices_df=close_prices_df,
        price_index=price_ix,
        valid_cols=valid_cols,
    )
    open_arr_np = None
    open_mask_np = None
    if tet == "next_bar_open":
        open_frame = extract_open_frame_from_ohlc(price_data_daily_ohlc)
        assert open_frame is not None
        open_arr_np, open_mask_np = prepare_open_arrays_for_simulation(
            market_data_panel=None,
            open_prices_df=open_frame,
            price_index=price_ix,
            valid_cols=valid_cols,
        )

    sparse_exec_targets = sized_for_daily.reindex(columns=valid_cols)
    sim_input = build_portfolio_simulation_input(
        weights_daily=weights_daily,
        price_index=price_ix,
        valid_cols=valid_cols,
        close_arr=close_arr,
        close_price_mask_arr=close_mask_arr,
        open_arr=open_arr_np,
        open_price_mask_arr=open_mask_np,
        sparse_execution_targets=sparse_exec_targets,
        trade_execution_timing=tet,
    )
    return simulate_portfolio(
        sim_input,
        global_config=global_config,
        scenario_config=scenario,
    )


def test_monthly_sparse_identical_targets_rebalance_after_drift_incurring_costs():
    dates = pd.date_range("2023-01-03", "2023-03-10", freq="B")
    daily = pd.DataFrame(index=dates)
    daily["A"] = np.linspace(100.0, 250.0, len(dates))
    daily["B"] = 50.0
    ev1 = pd.Timestamp("2023-01-31")
    ev2 = pd.Timestamp("2023-02-28")
    sized = pd.DataFrame(
        {"A": [0.5, 0.5], "B": [0.5, 0.5]},
        index=pd.DatetimeIndex([ev1, ev2]),
    )
    scenario = {"timing_config": {"mode": "time_based", "rebalance_frequency": "ME"}}
    g = _base_global(50_000.0)
    scenario_cost = {
        **scenario,
        "costs_config": {"transaction_costs_bps": 25.0},
    }
    out = _simulate_result(sized, scenario_cost, daily, ["A", "B"], g)
    ix2 = int(dates.get_loc(ev2))
    assert float(out.total_cost_fraction[ix2]) > 1e-12


def test_ffilled_daily_weights_without_sparse_events_do_not_rebalance_mid_window():
    dates = pd.date_range("2023-01-03", periods=15, freq="B")
    daily = pd.DataFrame(
        {
            "A": [100.0 + float(i) for i in range(len(dates))],
            "B": [50.0] * len(dates),
        },
        index=dates,
    )
    ev_idx = [dates[0], dates[7], dates[14]]
    sized = pd.DataFrame(
        {"A": [0.6, 0.55, 0.5], "B": [0.4, 0.45, 0.5]},
        index=pd.DatetimeIndex(ev_idx),
    )
    scenario = {"timing_config": {"mode": "signal_based", "trade_execution_timing": "bar_close"}}
    g = _base_global(25_000.0)
    scenario_cost = {**scenario, "costs_config": {"transaction_costs_bps": 40.0}}
    out = _simulate_result(sized, scenario_cost, daily, ["A", "B"], g)
    mid = 3
    assert float(out.total_cost_fraction[mid]) == pytest.approx(0.0, abs=1e-15)


def test_signal_based_all_nan_skipped_rows_hold_prior_targets():
    dates = pd.date_range("2023-01-03", periods=7, freq="B")
    daily = pd.DataFrame(
        {
            "A": [100.0 + i for i in range(len(dates))],
            "B": [40.0 + 0.5 * i for i in range(len(dates))],
        },
        index=dates,
    )
    rets = daily.pct_change(fill_method=None).fillna(0.0)
    sized = pd.DataFrame(index=dates, columns=["A", "B"], dtype=float)
    sized.iloc[0] = [0.7, 0.3]
    sized.iloc[3] = [np.nan, np.nan]
    sized.iloc[6] = [0.5, 0.5]
    scenario = {"timing_config": {"mode": "signal_based", "trade_execution_timing": "bar_close"}}
    g = _base_global(30_000.0)
    _, _, signed = calculate_portfolio_returns(
        sized,
        scenario,
        daily,
        rets,
        ["A", "B"],
        g,
        track_trades=False,
        include_signed_weights=True,
    )
    assert signed is not None
    mid = dates[4]
    assert float(signed.loc[mid, "A"]) > 0.55


def test_next_bar_open_executes_at_open_not_close():
    dates = pd.date_range("2023-01-03", periods=3, freq="B")
    ticker = "X"
    ohlc = pd.DataFrame(
        {
            (ticker, "Open"): [100.0, 50.0, 100.0],
            (ticker, "Close"): [100.0, 100.0, 100.0],
        },
        index=dates,
    )
    ohlc.columns = pd.MultiIndex.from_tuples(ohlc.columns, names=["Ticker", "Field"])
    sized = pd.DataFrame({ticker: [1.0]}, index=pd.DatetimeIndex([dates[0]]))
    scenario = {
        "timing_config": {"mode": "signal_based", "trade_execution_timing": "next_bar_open"},
    }
    g = _base_global(100_000.0)
    out = _simulate_result(sized, scenario, ohlc, [ticker], g)
    exec_ix = 1
    shares_exec_open = float(g["portfolio_value"]) / 50.0
    assert float(out.positions.iloc[exec_ix][ticker]) == pytest.approx(shares_exec_open, rel=1e-6)
    shares_hyp_close_exec = float(g["portfolio_value"]) / 100.0
    assert float(out.positions.iloc[exec_ix][ticker]) != pytest.approx(
        shares_hyp_close_exec, rel=1e-3
    )


def test_next_bar_open_via_calculate_portfolio_returns_nav_return_uses_exec_open():
    dates = pd.date_range("2023-01-03", periods=3, freq="B")
    ticker = "X"
    ohlc = pd.DataFrame(
        {
            (ticker, "Open"): [100.0, 50.0, 100.0],
            (ticker, "Close"): [100.0, 100.0, 100.0],
        },
        index=dates,
    )
    ohlc.columns = pd.MultiIndex.from_tuples(ohlc.columns, names=["Ticker", "Field"])
    close_df = ohlc.xs("Close", level="Field", axis=1)
    rets = close_df.pct_change(fill_method=None).fillna(0.0)
    sized = pd.DataFrame({ticker: [1.0]}, index=pd.DatetimeIndex([dates[0]]))
    scenario = {
        "timing_config": {"mode": "signal_based", "trade_execution_timing": "next_bar_open"},
    }
    g = _base_global(100_000.0)
    rets_net, _ = calculate_portfolio_returns(
        sized,
        scenario,
        ohlc,
        rets,
        [ticker],
        g,
        track_trades=False,
    )
    exec_ix = dates[1]
    assert float(rets_net.loc[exec_ix]) == pytest.approx(1.0, rel=0.0, abs=1e-5)
