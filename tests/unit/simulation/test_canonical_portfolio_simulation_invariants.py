from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.backtester_logic.portfolio_logic import calculate_portfolio_returns


def _base_global(portfolio_value: float = 10_000.0) -> dict:
    return {
        "portfolio_value": portfolio_value,
        "commission_per_share": 0.0,
        "commission_min_per_order": 0.0,
        "commission_max_percent_of_trade": 0.0,
        "slippage_bps": 0.0,
    }


def test_calculate_portfolio_returns_track_trades_does_not_change_return_series():
    dates = pd.date_range("2023-01-01", periods=5, freq="B")
    daily = pd.DataFrame(
        {"A": [100.0, 102.0, 98.0, 101.0, 105.0], "B": [50.0, 51.0, 52.0, 48.0, 50.0]},
        index=dates,
    )
    rets = daily.pct_change(fill_method=None).fillna(0.0)
    sized = pd.DataFrame(
        {
            "A": [0.6, 0.4, 0.8, 0.2, 0.0],
            "B": [0.4, 0.6, 0.2, 0.8, 1.0],
        },
        index=dates,
    )
    scenario = {
        "timing_config": {"rebalance_frequency": "D"},
        "costs_config": {"transaction_costs_bps": 12.0},
        "allocation_mode": "reinvestment",
    }
    g = _base_global(50_000.0)

    r_off, tt_off = calculate_portfolio_returns(
        sized, scenario, daily, rets, ["A", "B"], g, track_trades=False
    )
    r_on, tt_on = calculate_portfolio_returns(
        sized, scenario, daily, rets, ["A", "B"], g, track_trades=True
    )

    assert tt_off is None
    assert tt_on is not None
    pd.testing.assert_series_equal(
        r_off.astype(float),
        r_on.astype(float),
        check_names=False,
        rtol=0.0,
        atol=1e-9,
    )


def test_calculate_portfolio_returns_track_trades_parity_with_missing_prices():
    dates = pd.date_range("2023-01-01", periods=4, freq="B")
    daily = pd.DataFrame(
        {"A": [100.0, np.nan, 110.0, 115.0], "B": [20.0, 21.0, 22.0, 23.0]},
        index=dates,
    )
    rets = daily.pct_change(fill_method=None)
    rets = rets.fillna(0.0)
    sized = pd.DataFrame({"A": [0.5, 0.5, 0.5, 0.0], "B": [0.5, 0.5, 0.5, 1.0]}, index=dates)
    scenario = {
        "timing_config": {"rebalance_frequency": "D"},
        "costs_config": {"transaction_costs_bps": 5.0},
    }
    g = _base_global(20_000.0)

    r_off, _ = calculate_portfolio_returns(
        sized, scenario, daily, rets, ["A", "B"], g, track_trades=False
    )
    r_on, _ = calculate_portfolio_returns(
        sized, scenario, daily, rets, ["A", "B"], g, track_trades=True
    )
    pd.testing.assert_series_equal(r_off.astype(float), r_on.astype(float), atol=1e-9, rtol=0.0)


def test_share_delta_costs_no_trade_when_target_row_repeats_after_price_jump():
    dates = pd.date_range("2023-01-01", periods=3, freq="D")
    daily = pd.DataFrame({"A": [100.0, 200.0, 200.0]}, index=dates)
    rets = daily.pct_change(fill_method=None).fillna(0.0)
    sized = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=dates)
    scenario = {
        "timing_config": {"rebalance_frequency": "D"},
        "costs_config": {"transaction_costs_bps": 100.0},
    }
    g = _base_global(10_000.0)

    r, _ = calculate_portfolio_returns(sized, scenario, daily, rets, ["A"], g, track_trades=False)
    assert r.iloc[1] > 0.0
    assert r.iloc[2] == pytest.approx(0.0)

    from portfolio_backtester.simulation.kernel import simulate_portfolio
    from portfolio_backtester.backtester_logic.portfolio_simulation_input import (
        build_close_and_mask_from_dataframe,
        build_portfolio_simulation_input,
    )

    valid_cols = ["A"]
    close_arr, mask_arr = build_close_and_mask_from_dataframe(daily, dates, valid_cols)
    sim_in = build_portfolio_simulation_input(
        weights_daily=sized.reindex(columns=valid_cols).ffill().fillna(0.0),
        price_index=dates,
        valid_cols=valid_cols,
        close_arr=close_arr,
        price_mask_arr=mask_arr,
    )
    out = simulate_portfolio(
        sim_in,
        global_config=g,
        scenario_config=scenario,
        materialize_trades=False,
    )
    assert out.per_asset_cost_fraction.shape == (len(dates), 1)
    assert float(out.per_asset_cost_fraction[1, 0]) == pytest.approx(0.0, abs=1e-12)
    assert float(out.per_asset_cost_fraction[2, 0]) == pytest.approx(0.0, abs=1e-12)


def test_share_delta_costs_on_rebalance_not_weight_drift_proxy():
    dates = pd.date_range("2023-01-01", periods=3, freq="D")
    daily = pd.DataFrame({"A": [100.0, 100.0, 100.0]}, index=dates)
    sized = pd.DataFrame({"A": [1.0, 0.5, 1.0]}, index=dates)
    scenario = {
        "timing_config": {"rebalance_frequency": "D"},
    }
    g = {
        **_base_global(10_000.0),
        "commission_per_share": 0.01,
        "commission_min_per_order": 0.0,
        "commission_max_percent_of_trade": 1.0,
        "slippage_bps": 0.0,
    }

    from portfolio_backtester.simulation.kernel import simulate_portfolio
    from portfolio_backtester.backtester_logic.portfolio_simulation_input import (
        build_close_and_mask_from_dataframe,
        build_portfolio_simulation_input,
    )

    valid_cols = ["A"]
    close_arr, mask_arr = build_close_and_mask_from_dataframe(daily, dates, valid_cols)
    wdf = sized.reindex(columns=valid_cols).ffill().fillna(0.0)
    sim_in = build_portfolio_simulation_input(
        weights_daily=wdf,
        price_index=dates,
        valid_cols=valid_cols,
        close_arr=close_arr,
        price_mask_arr=mask_arr,
    )
    out = simulate_portfolio(
        sim_in,
        global_config=g,
        scenario_config=scenario,
        materialize_trades=False,
    )
    assert float(np.sum(out.per_asset_cost_fraction[1, :])) > 0.0
    assert float(np.sum(out.per_asset_cost_fraction[2, :])) > 0.0
