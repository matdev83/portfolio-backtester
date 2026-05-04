"""Tests for sparse target-weight execution-date mapping (TDD)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.timing.trade_execution_timing import (
    map_sparse_target_weights_to_execution_dates,
)


@pytest.fixture
def calendar_week() -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-02", periods=5)


def test_bar_close_is_no_op_on_execution_index(calendar_week: pd.DatetimeIndex) -> None:
    decision = calendar_week[2]
    weights = pd.DataFrame({"AAA": [1.0]}, index=[decision])
    out = map_sparse_target_weights_to_execution_dates(
        weights,
        trade_execution_timing="bar_close",
        calendar=calendar_week,
    )
    pd.testing.assert_frame_equal(out.sort_index(), weights.sort_index())


def test_next_bar_open_moves_each_row_to_next_calendar_session(
    calendar_week: pd.DatetimeIndex,
) -> None:
    decision = calendar_week[2]
    weights = pd.DataFrame({"AAA": [1.0]}, index=[decision])
    out = map_sparse_target_weights_to_execution_dates(
        weights,
        trade_execution_timing="next_bar_open",
        calendar=calendar_week,
    )
    expected = pd.DataFrame({"AAA": [1.0]}, index=[calendar_week[3]])
    pd.testing.assert_frame_equal(out.sort_index(), expected.sort_index())


def test_next_bar_open_skips_weekend_using_supplied_calendar() -> None:
    calendar = pd.DatetimeIndex([pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-08")])
    weights = pd.DataFrame({"AAA": [1.0]}, index=[pd.Timestamp("2024-01-05")])
    out = map_sparse_target_weights_to_execution_dates(
        weights,
        trade_execution_timing="next_bar_open",
        calendar=calendar,
    )
    expected = pd.DataFrame({"AAA": [1.0]}, index=[pd.Timestamp("2024-01-08")])
    pd.testing.assert_frame_equal(out.sort_index(), expected.sort_index())


def test_duplicate_execution_dates_keep_last_event_row(calendar_week: pd.DatetimeIndex) -> None:
    weights = pd.concat(
        [
            pd.DataFrame({"AAA": [0.25]}, index=[calendar_week[0]]),
            pd.DataFrame({"AAA": [0.75]}, index=[calendar_week[0]]),
        ]
    )
    out = map_sparse_target_weights_to_execution_dates(
        weights,
        trade_execution_timing="next_bar_open",
        calendar=calendar_week,
    )
    expected = pd.DataFrame({"AAA": [0.75]}, index=[calendar_week[1]])
    pd.testing.assert_frame_equal(out.sort_index(), expected.sort_index())


def test_next_bar_open_final_calendar_date_overflow_drops_and_warns(
    calendar_week: pd.DatetimeIndex,
    caplog: pytest.LogCaptureFixture,
) -> None:
    weights = pd.DataFrame({"AAA": [1.0]}, index=[calendar_week[-1]])
    caplog.set_level(logging.WARNING)
    out = map_sparse_target_weights_to_execution_dates(
        weights,
        trade_execution_timing="next_bar_open",
        calendar=calendar_week,
        logger=logging.getLogger("portfolio_backtester.timing.trade_execution_timing"),
    )
    assert out.empty
    assert any(
        "trade_execution_timing" in rec.message.lower() or "drop" in rec.message.lower()
        for rec in caplog.records
    )


def test_all_nan_no_op_rows_are_dropped_and_do_not_emit_flattening_row(
    calendar_week: pd.DatetimeIndex,
) -> None:
    signal = pd.DataFrame({"AAA": [1.0]}, index=[calendar_week[0]])
    noop = pd.DataFrame([[np.nan]], columns=["AAA"], index=[calendar_week[1]])
    combined = pd.concat([signal, noop])
    out = map_sparse_target_weights_to_execution_dates(
        combined,
        trade_execution_timing="bar_close",
        calendar=calendar_week,
    )
    assert calendar_week[1] not in out.index
    pd.testing.assert_frame_equal(out.sort_index(), signal.sort_index())


def test_next_bar_open_empty_weights_frame_returns_empty_with_columns(
    calendar_week: pd.DatetimeIndex,
) -> None:
    weights = pd.DataFrame(columns=["AAA"])
    out = map_sparse_target_weights_to_execution_dates(
        weights,
        trade_execution_timing="next_bar_open",
        calendar=calendar_week,
    )
    assert out.empty
    assert list(out.columns) == ["AAA"]


def test_next_bar_open_all_nan_active_row_yields_empty(calendar_week: pd.DatetimeIndex) -> None:
    weights = pd.DataFrame({"AAA": [float("nan")]}, index=[calendar_week[0]])
    out = map_sparse_target_weights_to_execution_dates(
        weights,
        trade_execution_timing="next_bar_open",
        calendar=calendar_week,
    )
    assert out.empty


def test_invalid_trade_execution_timing_raises(calendar_week: pd.DatetimeIndex) -> None:
    weights = pd.DataFrame({"AAA": [1.0]}, index=[calendar_week[0]])
    with pytest.raises(ValueError, match="Invalid trade_execution_timing"):
        map_sparse_target_weights_to_execution_dates(
            weights,
            trade_execution_timing="not_a_mode",
            calendar=calendar_week,
        )
