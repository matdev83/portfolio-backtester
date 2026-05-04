"""Parity between DataFrame close prep and MarketDataPanel-backed prep (canonical selection)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from portfolio_backtester.backtester_logic.portfolio_logic import calculate_portfolio_returns
from portfolio_backtester.optimization.market_data_panel import MarketDataPanel


def _base_global(pv: float = 10000.0) -> dict:
    return {
        "portfolio_value": pv,
        "commission_per_share": 0.005,
        "commission_min_per_order": 1.0,
        "commission_max_percent_of_trade": 0.005,
        "slippage_bps": 2.5,
    }


@pytest.mark.parametrize("tet", ["bar_close", "next_bar_open"])
def test_array_prep_matches_dataframe_timing_modes(tet: str) -> None:
    dates = pd.date_range("2023-01-03", periods=8, freq="B")
    daily = pd.DataFrame(
        {"A": np.linspace(100.0, 108.0, len(dates)), "B": np.linspace(50.0, 52.0, len(dates))},
        index=dates,
    )
    rets = daily.pct_change(fill_method=None).fillna(0.0)
    sized = pd.DataFrame({"A": [0.6, 0.4], "B": [0.4, 0.6]}, index=[dates[0], dates[3]])
    sized = sized.reindex(dates).ffill()
    scenario = {
        "timing_config": {
            "mode": "signal_based",
            "scan_frequency": "D",
            "trade_execution_timing": tet,
        },
        "costs_config": {"transaction_costs_bps": 5.0},
    }
    panel = MarketDataPanel.from_daily_ohlc_and_returns(daily, rets)
    g = _base_global()
    r_panel, _ = calculate_portfolio_returns(
        sized, scenario, daily, rets, ["A", "B"], g, market_data_panel=panel
    )
    r_df, _ = calculate_portfolio_returns(sized, scenario, daily, rets, ["A", "B"], g)
    pd.testing.assert_series_equal(r_panel, r_df, rtol=1e-5, atol=1e-12)


def test_array_prep_detailed_commission_path() -> None:
    dates = pd.date_range("2023-01-03", periods=6, freq="B")
    daily = pd.DataFrame(
        {"A": [100.0, 101.0, 99.0, 102.0, 101.0, 103.0], "B": [20.0, 21.0, 20.5, 22.0, 21.5, 22.5]},
        index=dates,
    )
    rets = daily.pct_change(fill_method=None).fillna(0.0)
    sized = pd.DataFrame(0.5, index=dates, columns=["A", "B"])
    scenario = {"timing_config": {"rebalance_frequency": "D"}}
    panel = MarketDataPanel.from_daily_ohlc_and_returns(daily, rets)
    g = _base_global()
    r_panel, _ = calculate_portfolio_returns(
        sized, scenario, daily, rets, ["A", "B"], g, market_data_panel=panel
    )
    r_df, _ = calculate_portfolio_returns(sized, scenario, daily, rets, ["A", "B"], g)
    pd.testing.assert_series_equal(r_panel, r_df, rtol=1e-5, atol=1e-12)


def test_array_prep_missing_prices_multiindex_ohlc() -> None:
    dates = pd.date_range("2023-01-03", periods=5, freq="B")
    tuples: list[tuple[str, str]] = []
    for t in ["A", "B"]:
        tuples.extend([(t, "Open"), (t, "High"), (t, "Low"), (t, "Close")])
    cols = pd.MultiIndex.from_tuples(tuples, names=["Ticker", "Field"])
    ohlc = pd.DataFrame(100.0, index=dates, columns=cols)
    ohlc.loc[dates[2], ("B", "Close")] = np.nan
    _daily_close = ohlc.xs("Close", level="Field", axis=1)
    daily_close = (
        _daily_close if isinstance(_daily_close, pd.DataFrame) else _daily_close.to_frame()
    )
    rets = daily_close.pct_change(fill_method=None).fillna(0.0)
    sized = pd.DataFrame({"A": 0.7, "B": 0.3}, index=dates)
    scenario = {
        "timing_config": {"rebalance_frequency": "D"},
        "costs_config": {"transaction_costs_bps": 0.0},
    }
    panel = MarketDataPanel.from_daily_ohlc_and_returns(ohlc, rets)
    g = _base_global()
    r_panel, _ = calculate_portfolio_returns(
        sized, scenario, ohlc, rets, ["A", "B"], g, market_data_panel=panel
    )
    r_df, _ = calculate_portfolio_returns(sized, scenario, ohlc, rets, ["A", "B"], g)
    pd.testing.assert_series_equal(r_panel, r_df, rtol=1e-5, atol=1e-12)


def test_array_prep_sparse_signal_rows() -> None:
    thu = pd.Timestamp("2023-09-28")
    fri = pd.Timestamp("2023-09-29")
    mon = pd.Timestamp("2023-10-02")
    idx = pd.DatetimeIndex([thu, fri, mon])
    daily = pd.DataFrame(100.0, index=idx, columns=["A"])
    rets = pd.DataFrame(0.0, index=idx, columns=["A"])
    rets.loc[mon, "A"] = 0.1
    sized = pd.DataFrame({"A": [1.0]}, index=[fri])
    scenario = {
        "timing_config": {"mode": "time_based", "rebalance_frequency": "ME"},
        "costs_config": {"transaction_costs_bps": 0.0},
    }
    panel = MarketDataPanel.from_daily_ohlc_and_returns(daily, rets)
    g = _base_global()
    r_panel, _ = calculate_portfolio_returns(
        sized, scenario, daily, rets, ["A"], g, market_data_panel=panel
    )
    r_df, _ = calculate_portfolio_returns(sized, scenario, daily, rets, ["A"], g)
    pd.testing.assert_series_equal(r_panel, r_df, rtol=1e-5, atol=1e-12)


def test_no_trade_tracking_panel_matches_dataframe() -> None:
    dates = pd.date_range("2023-01-03", periods=5, freq="B")
    daily = pd.DataFrame(100.0, index=dates, columns=["A", "B"])
    rets = daily.pct_change(fill_method=None).fillna(0.0)
    sized = pd.DataFrame(
        {"A": [0.0, 0.0, 1.0, 1.0, 0.0], "B": [1.0, 1.0, 0.0, 0.0, 1.0]}, index=dates
    )
    scenario = {
        "timing_config": {"rebalance_frequency": "D"},
        "costs_config": {"transaction_costs_bps": 0.0},
        "allocation_mode": "fixed",
    }
    panel = MarketDataPanel.from_daily_ohlc_and_returns(daily, rets)
    g = _base_global()
    r_a, _ = calculate_portfolio_returns(
        sized,
        scenario,
        daily,
        rets,
        ["A", "B"],
        g,
        track_trades=False,
        market_data_panel=panel,
    )
    r_b, _ = calculate_portfolio_returns(
        sized, scenario, daily, rets, ["A", "B"], g, track_trades=False
    )
    pd.testing.assert_series_equal(r_a, r_b, rtol=1e-5, atol=1e-12)


@patch("portfolio_backtester.backtester_logic.portfolio_logic.TradeTracker")
def test_trade_tracking_falls_back_close_prep(mock_tt_cls) -> None:
    dates = pd.date_range("2023-01-03", periods=5, freq="B")
    daily = pd.DataFrame(100.0, index=dates, columns=["A", "B"])
    rets = daily.pct_change(fill_method=None).fillna(0.0)
    sized = pd.DataFrame(0.5, index=dates, columns=["A", "B"])
    scenario = {
        "timing_config": {"rebalance_frequency": "D"},
        "costs_config": {"transaction_costs_bps": 0.0},
        "allocation_mode": "fixed",
    }
    mock_tt = MagicMock()
    mock_tt.initial_portfolio_value = 1000.0
    mock_tt.allocation_mode = "fixed"
    mock_tt.portfolio_value_tracker.daily_portfolio_value = pd.Series(1000.0, index=dates)
    mock_tt_cls.return_value = mock_tt
    panel = MarketDataPanel.from_daily_ohlc_and_returns(daily, rets)
    g = {**_base_global(1000.0), "portfolio_value": 1000.0}
    calculate_portfolio_returns(
        sized,
        scenario,
        daily,
        rets,
        ["A", "B"],
        g,
        track_trades=True,
        market_data_panel=panel,
    )
    calculate_portfolio_returns(sized, scenario, daily, rets, ["A", "B"], g, track_trades=True)
    assert mock_tt.populate_from_kernel_results.call_count == 2


def test_misaligned_panel_falls_back_to_dataframe() -> None:
    dates = pd.date_range("2023-01-03", periods=4, freq="B")
    daily = pd.DataFrame(100.0, index=dates, columns=["A"])
    rets = daily.pct_change(fill_method=None).fillna(0.0)
    sized = pd.DataFrame(1.0, index=dates, columns=["A"])
    scenario = {
        "timing_config": {"rebalance_frequency": "D"},
        "costs_config": {"transaction_costs_bps": 0.0},
    }
    wrong_panel = MarketDataPanel.from_daily_ohlc_and_returns(daily.iloc[:-1], rets.iloc[:-1])
    g = _base_global()
    r_wrong, _ = calculate_portfolio_returns(
        sized, scenario, daily, rets, ["A"], g, market_data_panel=wrong_panel
    )
    r_df, _ = calculate_portfolio_returns(sized, scenario, daily, rets, ["A"], g)
    pd.testing.assert_series_equal(r_wrong, r_df, rtol=1e-5, atol=1e-12)
