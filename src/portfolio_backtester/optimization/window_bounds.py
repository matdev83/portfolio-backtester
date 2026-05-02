"""Positional bounds for walk-forward windows on evaluation-naive daily indexes."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .market_data_panel import drop_tz_like_evaluator
from .wfo_window import WFOWindow


def _normalize_daily_index_tz_naive(daily_index: pd.DatetimeIndex | pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(daily_index))
    if idx.tz is not None:
        return idx.tz_localize(None)
    return idx


def _iloc_half_open(
    eval_index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[int, int]:
    """Returns ``slice_locs``: ``iloc[start_ix:end_ix]`` aligns with ``.loc[start:end]``."""

    raw_start, raw_end_exclusive = eval_index.slice_locs(start, end)
    return int(raw_start), int(raw_end_exclusive)


@dataclass(frozen=True)
class WindowBounds:
    """Half-open row intervals into an evaluation ``daily_index_naive`` (``.iloc``-style).

    ``end_ix_exclusive`` boundaries come directly from pandas ``DatetimeIndex.slice_locs``,
    after dropping timezone awareness using the same normalization as ``evaluate_window``.
    Labels remain inclusive on ``.loc``; integer ranges use half-open slicing for ``iloc``.
    """

    train_start_ix: int
    train_end_ix_exclusive: int
    extent_start_ix: int
    extent_end_ix_exclusive: int
    test_start_ix: int
    test_end_ix_exclusive: int


def build_window_bounds(
    eval_daily_index_naive: pd.DatetimeIndex | pd.Index, window: WFOWindow
) -> WindowBounds:
    """Map WFOWindow calendar bounds to positional windows."""

    ix = _normalize_daily_index_tz_naive(eval_daily_index_naive)

    ts_tr_s = drop_tz_like_evaluator(window.train_start)
    ts_tr_e = drop_tz_like_evaluator(window.train_end)
    ts_te_s = drop_tz_like_evaluator(window.test_start)
    ts_te_e = drop_tz_like_evaluator(window.test_end)

    tr_s, tr_e_ex = _iloc_half_open(ix, ts_tr_s, ts_tr_e)
    te_s, te_e_ex = _iloc_half_open(ix, ts_te_s, ts_te_e)
    ex_s, ex_e_ex = _iloc_half_open(ix, ts_tr_s, ts_te_e)

    return WindowBounds(
        train_start_ix=tr_s,
        train_end_ix_exclusive=tr_e_ex,
        extent_start_ix=ex_s,
        extent_end_ix_exclusive=ex_e_ex,
        test_start_ix=te_s,
        test_end_ix_exclusive=te_e_ex,
    )
