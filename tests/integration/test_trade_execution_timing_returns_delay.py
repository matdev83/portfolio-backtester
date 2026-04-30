"""Regression-style check: ``next_bar_open`` delays exposure vs ``bar_close`` through return kernel."""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_backtester.numba_kernels import drifting_weights_returns_kernel
from portfolio_backtester.timing.trade_execution_timing import (
    map_sparse_target_weights_to_execution_dates,
)


def _dense_weights_matrix(
    mapped: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    aligned = mapped.reindex(calendar).ffill()
    aligned = aligned.fillna(0.0)
    aligned = aligned.shift(1).fillna(0.0)
    values = aligned[columns].to_numpy(dtype=np.float64)
    mask = np.ones_like(values, dtype=np.bool_)
    return values, mask


def test_next_bar_open_delays_exposure_relative_to_bar_close() -> None:
    calendar = pd.bdate_range("2024-06-03", periods=8)
    decision = calendar[3]
    sparse = pd.DataFrame({"X": [1.0]}, index=[decision])

    mapped_close = map_sparse_target_weights_to_execution_dates(
        sparse,
        trade_execution_timing="bar_close",
        calendar=calendar,
    )
    mapped_open = map_sparse_target_weights_to_execution_dates(
        sparse,
        trade_execution_timing="next_bar_open",
        calendar=calendar,
    )

    cols = ["X"]
    rets = np.zeros((len(calendar), len(cols)), dtype=np.float64)
    rets[4, 0] = 0.02

    w_close, mask_close = _dense_weights_matrix(mapped_close, calendar, cols)
    w_open, mask_open = _dense_weights_matrix(mapped_open, calendar, cols)

    r_close = drifting_weights_returns_kernel(w_close, rets, mask_close)
    r_open = drifting_weights_returns_kernel(w_open, rets, mask_open)

    assert not np.allclose(r_close, r_open)
