"""Unit tests for PreparedArrays cache wrapper.

`portfolio_backtester.prepared_arrays.get_or_prepare` is used to hoist expensive
DataFrame->ndarray conversion out of optimization loops. These tests guard the
shape/dtype invariants and that the cache keying is stable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester import prepared_arrays


def _make_inputs() -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DatetimeIndex]:
    dates = pd.date_range("2022-01-01", periods=3, freq="D")
    universe = ["A", "B"]

    weights = pd.DataFrame(
        {
            "A": [0.5, 0.6, 0.7],
            "B": [0.5, 0.4, 0.3],
        },
        index=dates,
    )

    rets = pd.DataFrame(
        {
            "A": [0.01, np.nan, 0.03],
            "B": [0.02, 0.01, 0.00],
        },
        index=dates,
    )

    return weights, rets, universe, dates


def test_get_or_prepare_returns_cached_object_for_same_key() -> None:
    # Clear internal cache to keep this test deterministic.
    prepared_arrays._PREPARED_CACHE._cache.clear()

    weights, rets, universe, dates = _make_inputs()

    pa1 = prepared_arrays.get_or_prepare(
        weights_daily=weights,
        rets_daily=rets,
        universe_tickers=universe,
        price_index=dates,
        universe_name="u1",
        scenario_name="s1",
        feature_flags={"a": 1},
        use_float32=True,
    )
    pa2 = prepared_arrays.get_or_prepare(
        weights_daily=weights,
        rets_daily=rets,
        universe_tickers=universe,
        price_index=dates,
        universe_name="u1",
        scenario_name="s1",
        feature_flags={"a": 1},
        use_float32=True,
    )

    assert pa1 is pa2
    assert pa1.weights.shape == (len(dates), len(universe))
    assert pa1.rets.shape == (len(dates), len(universe))
    assert pa1.mask.shape == (len(dates), len(universe))
    assert pa1.dtype == np.dtype(np.float32)


def test_get_or_prepare_cache_key_changes_with_columns() -> None:
    prepared_arrays._PREPARED_CACHE._cache.clear()

    weights, rets, universe, dates = _make_inputs()

    # Make a second weights frame with a different column set -> should produce a different cache key
    weights2 = weights.copy()
    weights2["C"] = 0.0

    pa1 = prepared_arrays.get_or_prepare(
        weights_daily=weights,
        rets_daily=rets,
        universe_tickers=universe,
        price_index=dates,
        universe_name="u2",
        scenario_name="s2",
        use_float32=True,
    )
    pa2 = prepared_arrays.get_or_prepare(
        weights_daily=weights2,
        rets_daily=rets,
        universe_tickers=universe,
        price_index=dates,
        universe_name="u2",
        scenario_name="s2",
        use_float32=True,
    )

    assert pa1 is not pa2


def test_get_or_prepare_rejects_non_dataframes() -> None:
    prepared_arrays._PREPARED_CACHE._cache.clear()

    _, rets, universe, dates = _make_inputs()

    with pytest.raises(ValueError, match="must be DataFrames"):
        prepared_arrays.get_or_prepare(
            weights_daily=123,  # type: ignore[arg-type]
            rets_daily=rets,
            universe_tickers=universe,
            price_index=dates,
        )
