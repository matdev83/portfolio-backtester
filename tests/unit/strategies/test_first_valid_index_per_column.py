"""Parity tests for ``_first_valid_index_per_column`` vs pandas ``apply``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.portfolio_backtester.strategies._core.base.base.base_strategy import (
    _first_valid_index_per_column,
)


def _reference(close_df: pd.DataFrame) -> pd.Series:
    return close_df.apply(lambda s: s.first_valid_index())


def test_first_valid_matches_apply_mixed_nan_leadin() -> None:
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    close_df = pd.DataFrame(
        {
            "A": [np.nan, np.nan, 1.0, 2.0, 3.0],
            "B": [10.0, np.nan, np.nan, 20.0, 30.0],
        },
        index=idx,
    )
    got = _first_valid_index_per_column(close_df)
    ref = _reference(close_df)
    pd.testing.assert_series_equal(got, ref, check_names=False, check_dtype=False)


def test_first_valid_all_nan_column_is_none() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    close_df = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0], "B": [np.nan, np.nan, np.nan]},
        index=idx,
    )
    got = _first_valid_index_per_column(close_df)
    ref = _reference(close_df)
    pd.testing.assert_series_equal(got, ref, check_names=False, check_dtype=False)


def test_first_valid_single_column() -> None:
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    close_df = pd.DataFrame({"X": [np.nan, 1.0, 2.0, 3.0]}, index=idx)
    got = _first_valid_index_per_column(close_df)
    ref = _reference(close_df)
    pd.testing.assert_series_equal(got, ref, check_names=False, check_dtype=False)


def test_first_valid_empty_frame() -> None:
    close_df = pd.DataFrame()
    got = _first_valid_index_per_column(close_df)
    ref = _reference(close_df)
    pd.testing.assert_series_equal(got, ref, check_names=False, check_dtype=False)
