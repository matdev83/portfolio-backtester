"""Tests for dummy signal strategy historical slicing (empty non-universe panel)."""

from __future__ import annotations

import pandas as pd

from portfolio_backtester.testing.strategies.dummy_signal_strategy import (
    _slice_historical_to_date,
)


def test_slice_historical_empty_non_universe_no_typeerror() -> None:
    cal = pd.date_range("2020-01-01", periods=3, freq="D")
    empty = pd.DataFrame()
    out = _slice_historical_to_date(
        empty,
        pd.Timestamp("2020-01-02"),
        calendar_index=cal,
    )
    assert out.empty


def test_slice_historical_datetime_index() -> None:
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    out = _slice_historical_to_date(
        df,
        pd.Timestamp("2020-01-02"),
        calendar_index=idx,
    )
    assert len(out) == 2
    assert out.index.max() == pd.Timestamp("2020-01-02")
