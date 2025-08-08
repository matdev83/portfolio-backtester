import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, List

import numpy as np
import pandas as pd

from .ndarray_adapter import to_ndarrays

logger = logging.getLogger(__name__)


@dataclass
class PreparedArrays:
    weights: np.ndarray  # [T, N] float32/float64
    rets: np.ndarray  # [T, N] float32/float64
    mask: np.ndarray  # [T, N] bool
    dates: pd.DatetimeIndex
    tickers: List[str]
    dtype: np.dtype
    shapes: Dict[str, Tuple[int, ...]]
    bytes_estimate: int


class _PreparedArraysCache:
    def __init__(self) -> None:
        self._cache: Dict[Tuple[Any, ...], PreparedArrays] = {}

    def make_key(
        self,
        universe_name: Optional[str],
        date_start: Optional[pd.Timestamp],
        date_end: Optional[pd.Timestamp],
        scenario_name: Optional[str],
        feature_flags: Optional[Dict[str, Any]],
        rets_columns: Optional[Tuple[str, ...]],
        weights_columns: Optional[Tuple[str, ...]],
    ) -> Tuple[Any, ...]:
        flags_tuple = tuple(sorted((feature_flags or {}).items()))
        return (
            universe_name,
            str(date_start) if date_start is not None else None,
            str(date_end) if date_end is not None else None,
            scenario_name,
            flags_tuple,
            rets_columns,
            weights_columns,
        )

    def get(self, key: Tuple[Any, ...]) -> Optional[PreparedArrays]:
        return self._cache.get(key)

    def set(self, key: Tuple[Any, ...], value: PreparedArrays) -> None:
        self._cache[key] = value

    def size(self) -> int:
        return len(self._cache)


_PREPARED_CACHE = _PreparedArraysCache()


def _estimate_bytes(arr: np.ndarray) -> int:
    try:
        return int(arr.nbytes)
    except Exception:
        return 0


def get_or_prepare(
    weights_daily: pd.DataFrame,
    rets_daily: pd.DataFrame,
    universe_tickers: List[str],
    price_index: pd.DatetimeIndex,
    *,
    universe_name: Optional[str] = None,
    date_start: Optional[pd.Timestamp] = None,
    date_end: Optional[pd.Timestamp] = None,
    scenario_name: Optional[str] = None,
    feature_flags: Optional[Dict[str, Any]] = None,
    use_float32: bool = True,
) -> PreparedArrays:
    """
    Return PreparedArrays from cache or prepare once and cache.

    This hoists DataFrame->ndarray conversion out of per-trial loops.
    """
    if not isinstance(weights_daily, pd.DataFrame) or not isinstance(rets_daily, pd.DataFrame):
        raise ValueError("weights_daily and rets_daily must be DataFrames")

    rets_cols = tuple(rets_daily.columns.tolist())
    weights_cols = tuple(weights_daily.columns.tolist())
    key = _PREPARED_CACHE.make_key(
        universe_name=universe_name,
        date_start=date_start,
        date_end=date_end,
        scenario_name=scenario_name,
        feature_flags=feature_flags,
        rets_columns=rets_cols,
        weights_columns=weights_cols,
    )

    cached = _PREPARED_CACHE.get(key)
    if cached is not None:
        return cached

    adapter = to_ndarrays(
        weights_daily=weights_daily,
        rets_daily=rets_daily,
        universe_tickers=universe_tickers,
        price_index=price_index,
        use_float32=use_float32,
    )

    weights_arr: np.ndarray = adapter["weights"]
    rets_arr: np.ndarray = adapter["rets"]
    mask_arr: np.ndarray = adapter["mask"]
    dates = adapter["dates"]
    tickers = adapter["tickers"]
    dtype: np.dtype = adapter["dtype"]

    shapes = {
        "weights": tuple(weights_arr.shape),
        "rets": tuple(rets_arr.shape),
        "mask": tuple(mask_arr.shape),
    }
    total_bytes = (
        _estimate_bytes(weights_arr) + _estimate_bytes(rets_arr) + _estimate_bytes(mask_arr)
    )

    prepared = PreparedArrays(
        weights=weights_arr,
        rets=rets_arr,
        mask=mask_arr,
        dates=dates,
        tickers=tickers,
        dtype=dtype,
        shapes=shapes,
        bytes_estimate=total_bytes,
    )
    _PREPARED_CACHE.set(key, prepared)

    if logger.isEnabledFor(logging.INFO):
        mb = total_bytes / (1024 * 1024) if total_bytes else 0.0
        logger.info(
            "Prepared arrays (dtype=%s) shapes=%s approx=%.2f MB [cache_size=%d]",
            str(dtype),
            shapes,
            mb,
            _PREPARED_CACHE.size(),
        )

    return prepared
