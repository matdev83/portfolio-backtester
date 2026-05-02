"""Positional WFOWindow bounds parity with evaluator-style pandas slicing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_backtester.optimization.market_data_panel import (
    drop_tz_like_evaluator,
    evaluation_naive_datetimes_like_evaluator,
)
from portfolio_backtester.optimization.window_bounds import build_window_bounds
from portfolio_backtester.optimization.wfo_window import WFOWindow


def _daily_like_after_eval_strip(daily_tz: pd.DataFrame) -> pd.DataFrame:
    ix = pd.DatetimeIndex(pd.to_datetime(daily_tz.index))
    if ix.tz is not None:
        ix = ix.tz_localize(None)
    out = daily_tz.copy()
    out.index = ix
    return out


def test_bounds_match_loc_tz_and_fractional_calendar_timestamps() -> None:
    naive_idx = pd.date_range("2024-06-03", periods=20, freq="B")
    daily_tz = pd.DataFrame(
        np.arange(len(naive_idx)),
        index=naive_idx.tz_localize("US/Eastern"),
        columns=["p"],
    )
    normalized = _daily_like_after_eval_strip(daily_tz)

    window = WFOWindow(
        train_start=pd.Timestamp("2024-06-05 07:41:51", tz="US/Eastern"),
        train_end=pd.Timestamp("2024-06-12 02:51:43", tz="UTC"),
        test_start=pd.Timestamp("2024-06-07 06:06:59", tz="Europe/Berlin"),
        test_end=pd.Timestamp("2024-06-17 03:58:58"),
    )

    b = build_window_bounds(normalized.index, window)

    tr_s = drop_tz_like_evaluator(window.train_start)
    tr_e = drop_tz_like_evaluator(window.train_end)
    te_s = drop_tz_like_evaluator(window.test_start)
    te_e = drop_tz_like_evaluator(window.test_end)

    want_train = normalized.loc[tr_s:tr_e]
    got_train = normalized.iloc[b.train_start_ix : b.train_end_ix_exclusive]
    pd.testing.assert_frame_equal(want_train, got_train)

    want_extent = normalized.loc[tr_s:te_e]
    got_extent = normalized.iloc[b.extent_start_ix : b.extent_end_ix_exclusive]
    pd.testing.assert_frame_equal(want_extent, got_extent)

    want_test = normalized.loc[te_s:te_e]
    got_test = normalized.iloc[b.test_start_ix : b.test_end_ix_exclusive]
    pd.testing.assert_frame_equal(want_test, got_test)


def test_off_calendar_start_before_first_bar() -> None:
    naive_idx = pd.date_range("2024-08-05", periods=10, freq="B")
    normalized = pd.DataFrame({"p": np.arange(len(naive_idx))}, index=naive_idx)

    window = WFOWindow(
        train_start=pd.Timestamp("2024-06-02"),
        train_end=naive_idx[4],
        test_start=naive_idx[5],
        test_end=naive_idx[-1],
    )
    b = build_window_bounds(normalized.index, window)
    want_train = normalized.loc[
        drop_tz_like_evaluator(window.train_start) : drop_tz_like_evaluator(window.train_end)
    ]
    got_train = normalized.iloc[b.train_start_ix : b.train_end_ix_exclusive]
    pd.testing.assert_frame_equal(want_train, got_train)


def test_evaluation_naive_index_strips_tz() -> None:
    tz_idx = pd.date_range("2024-06-03", periods=5, freq="B", tz="Asia/Tokyo")
    got = evaluation_naive_datetimes_like_evaluator(tz_idx)
    assert getattr(got, "tz", None) is None
    assert len(got) == len(tz_idx)
