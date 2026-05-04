"""Regression tests for meta timing parity with standard signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_backtester.backtester_logic.portfolio_simulation_input import (
    propagate_rebalance_mask_for_invalid_next_bar_opens,
)
from portfolio_backtester.backtester_logic.strategy_logic import _resolve_signal_scan_window
from portfolio_backtester.timing.time_based_timing import TimeBasedTiming


def test_resolve_signal_scan_window_clips_to_wfo_bounds():
    ix = pd.date_range("2020-01-01", periods=400, freq="D")
    scenario_like = {
        "start_date": "2020-06-01",
        "end_date": None,
        "wfo_start_date": "2020-07-01",
        "wfo_end_date": "2020-09-30",
    }
    start_date, end_date, *_ = _resolve_signal_scan_window(scenario_like, ix)
    assert start_date == pd.Timestamp("2020-07-01")
    assert end_date == pd.Timestamp("2020-09-30")


def test_resolve_signal_scan_window_scenario_start_end_without_wfo():
    ix = pd.date_range("2021-01-01", periods=100, freq="D")
    scenario_like = {
        "start_date": "2021-02-01",
        "end_date": "2021-03-15",
        "wfo_start_date": None,
        "wfo_end_date": None,
    }
    start_date, end_date, *_ = _resolve_signal_scan_window(scenario_like, ix)
    assert start_date == pd.Timestamp("2021-02-01")
    assert end_date == pd.Timestamp("2021-03-15")


def test_propagate_rebalance_mask_numpy_contract_three_day_chain():
    rb = np.array([True, False, False, False, False], dtype=bool)
    w = np.ones((5, 1), dtype=np.float64)
    ex = np.array([[False], [False], [True], [True], [True]], dtype=bool)
    out = propagate_rebalance_mask_for_invalid_next_bar_opens(rb, w, ex)
    expected = np.array([True, True, True, False, False], dtype=bool)
    np.testing.assert_array_equal(out, expected)


def test_month_end_rolls_to_prior_session_when_calendar_me_missing_from_index():
    """Calendar month-end can be a non-session day; timing rolls to the prior session."""

    available = pd.DatetimeIndex(pd.bdate_range("2023-04-01", "2023-04-28"))
    assert pd.Timestamp("2023-04-30") not in available  # Sunday in 2023; index ends Friday 28th

    tc = TimeBasedTiming({"rebalance_frequency": "M"})
    rd = tc.get_rebalance_dates(
        start_date=available.min(),
        end_date=pd.Timestamp("2023-04-30"),
        available_dates=available,
        strategy_context=object(),
    )
    assert pd.Timestamp("2023-04-28") in rd

    legacy_calendar_me = pd.date_range("2023-04-01", "2023-04-30", freq="ME")
    dropped_by_isin = legacy_calendar_me[legacy_calendar_me.isin(available)]
    assert len(dropped_by_isin) == 0
