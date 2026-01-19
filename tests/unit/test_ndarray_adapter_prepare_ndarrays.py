"""Unit tests for ndarray preparation utilities.

These tests focus on critical data-prep invariants:
- Proper [T, N] shapes
- Deterministic ticker ordering and column filtering
- Correct mask semantics
- Contiguous arrays suitable for fast/Numba kernels
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_backtester import ndarray_adapter


def _make_frames() -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]
):
    dates = pd.date_range("2023-01-01", periods=4, freq="D")

    # weights use a subset of dates and shuffled columns; will be aligned/reindexed
    weights_for_returns = pd.DataFrame(
        {"B": [0.2, 0.3], "A": [0.8, 0.7]},
        index=dates[:2],
    )
    weights_current = pd.DataFrame(
        {"A": [0.9, 0.6], "B": [0.1, 0.4]},
        index=dates[:2],
    )

    # returns include NaNs and an extra ticker that is not in the requested universe
    rets_daily = pd.DataFrame(
        {
            "A": [0.01, np.nan, 0.03, 0.04],
            "B": [0.02, 0.01, np.nan, 0.00],
            "C": [0.99, 0.99, 0.99, 0.99],
        },
        index=dates,
    )

    prices_close = pd.DataFrame(
        {
            "A": [100.0, 101.0, np.nan, 103.0],
            "B": [50.0, 0.0, 52.0, 53.0],  # 0 should be treated as invalid in prices_mask
            "C": [1.0, 1.0, 1.0, 1.0],
        },
        index=dates,
    )

    return weights_for_returns, weights_current, rets_daily, prices_close, dates


def test_prepare_ndarrays_alignment_masks_dtype_and_contiguity() -> None:
    ndarray_adapter._PREPARED_CACHE.clear()

    wret, wcur, rets, prices, dates = _make_frames()

    # Universe includes a missing ticker "D"; adapter should filter to those present in returns.
    universe = ["B", "A", "D"]

    pa = ndarray_adapter.prepare_ndarrays(
        weights_for_returns=wret,
        weights_current=wcur,
        rets_daily=rets,
        universe_tickers=universe,
        price_index=dates,
        use_float32=True,
        prices_close_df=prices,
    )

    # Column filtering/order: only tickers present in rets_daily, in the requested universe order.
    assert pa.tickers == ["B", "A"]

    # Shape invariants
    assert pa.weights_for_returns.shape == (len(dates), 2)
    assert pa.weights_current.shape == (len(dates), 2)
    assert pa.rets.shape == (len(dates), 2)
    assert pa.rets_mask.shape == (len(dates), 2)
    assert pa.prices_close is not None
    assert pa.prices_close.shape == (len(dates), 2)
    assert pa.prices_mask is not None
    assert pa.prices_mask.shape == (len(dates), 2)

    # dtype invariants
    assert pa.dtype == np.dtype(np.float32)
    assert pa.weights_for_returns.dtype == np.float32
    assert pa.rets.dtype == np.float32

    # contiguity invariants (important for Numba fastpaths)
    assert pa.weights_for_returns.flags["C_CONTIGUOUS"]
    assert pa.weights_current.flags["C_CONTIGUOUS"]
    assert pa.rets.flags["C_CONTIGUOUS"]
    assert pa.rets_mask.flags["C_CONTIGUOUS"]
    assert pa.prices_close.flags["C_CONTIGUOUS"]
    assert pa.prices_mask.flags["C_CONTIGUOUS"]

    # Mask semantics: rets_mask indicates where original rets_daily had non-NaN for tickers.
    # For ticker B: NaN at dates[2]
    assert bool(pa.rets_mask[2, 0]) is False
    assert bool(pa.rets_mask[0, 0]) is True
    # For ticker A: NaN at dates[1]
    assert bool(pa.rets_mask[1, 1]) is False

    # prices_mask is True where price is notna and > 0.0
    # For B: price at dates[1] is 0 -> invalid
    assert bool(pa.prices_mask[1, 0]) is False
    # For A: price at dates[2] is NaN -> invalid
    assert bool(pa.prices_mask[2, 1]) is False


def test_get_or_prepare_cached_returns_same_object_for_same_key() -> None:
    ndarray_adapter._PREPARED_CACHE.clear()

    wret, wcur, rets, prices, dates = _make_frames()
    universe = ["A", "B"]
    cache_key = ("universe", str(dates[0]), str(dates[-1]), "float32", True, "daily")

    pa1 = ndarray_adapter.get_or_prepare_cached(
        cache_key,
        weights_for_returns=wret,
        weights_current=wcur,
        rets_daily=rets,
        universe_tickers=universe,
        price_index=dates,
        use_float32=True,
        prices_close_df=prices,
    )
    pa2 = ndarray_adapter.get_or_prepare_cached(
        cache_key,
        weights_for_returns=wret,
        weights_current=wcur,
        rets_daily=rets,
        universe_tickers=universe,
        price_index=dates,
        use_float32=True,
        prices_close_df=prices,
    )

    assert pa1 is pa2


def test_to_ndarrays_legacy_adapter_preserves_shapes_and_mask() -> None:
    wret, _, rets, _, dates = _make_frames()

    adapter = ndarray_adapter.to_ndarrays(
        weights_daily=wret,
        rets_daily=rets,
        universe_tickers=["A", "B"],
        price_index=dates,
        use_float32=False,
    )

    assert adapter["weights"].shape == (len(dates), 2)
    assert adapter["rets"].shape == (len(dates), 2)
    assert adapter["mask"].shape == (len(dates), 2)
    assert adapter["dtype"] == np.dtype(np.float64)
    assert adapter["weights"].flags["C_CONTIGUOUS"]
    assert adapter["rets"].flags["C_CONTIGUOUS"]

    # The mask should reflect NaNs in the original returns.
    # rets['A'] has NaN at dates[1]
    assert bool(adapter["mask"][1, 0]) is False
