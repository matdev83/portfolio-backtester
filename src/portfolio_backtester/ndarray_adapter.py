import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Sequence, TypedDict


@dataclass(frozen=True)
class PreparedArrays:
    """
    Reusable, aligned, contiguous ndarray views for fast kernels.
    """

    dates: pd.DatetimeIndex
    tickers: List[str]
    dtype: np.dtype

    # Core arrays [T, N]
    weights_for_returns: np.ndarray
    weights_current: np.ndarray
    rets: np.ndarray
    prices_close: Optional[np.ndarray]  # may be None when not provided

    # Masks [T, N]
    rets_mask: np.ndarray
    prices_mask: Optional[np.ndarray]


# Simple in-process cache keyed by (universe_key, start, end, dtype, float32_flag, rebalance_sig)
_PREPARED_CACHE: Dict[Tuple[Any, Any, Any, Any, Any, Any], PreparedArrays] = {}


def _normalize_dtype(use_float32: bool) -> "np.dtype[Any]":
    # np.float32 is a class; wrap with np.dtype for a concrete dtype instance
    return np.dtype(np.float32 if use_float32 else np.float64)


class _AlignedFrames(TypedDict):
    weights_for_returns: pd.DataFrame
    weights_current: pd.DataFrame
    rets: pd.DataFrame
    prices_close: Optional[pd.DataFrame]
    tickers: List[str]


def _ensure_aligned_frames(
    weights_for_returns: pd.DataFrame,
    weights_current: pd.DataFrame,
    rets_daily: pd.DataFrame,
    universe_tickers: Sequence[str],
    price_index: pd.DatetimeIndex,
    prices_close_df: Optional[pd.DataFrame] = None,
) -> _AlignedFrames:
    """
    Align inputs by date and columns in the given order, ffill weights_for_returns/current.
    """
    valid_cols: List[str] = [str(t) for t in universe_tickers if t in rets_daily.columns]
    # Reindex weights and ffill, fillna zeros
    wret = weights_for_returns.reindex(index=price_index, columns=valid_cols).fillna(0.0)
    wcur = weights_current.reindex(index=price_index, columns=valid_cols).fillna(0.0)
    rets = rets_daily.reindex(index=price_index, columns=valid_cols).fillna(0.0)
    prices_close: Optional[pd.DataFrame] = None
    if prices_close_df is not None:
        prices_close = prices_close_df.reindex(index=price_index, columns=valid_cols)

    return {
        "weights_for_returns": wret,
        "weights_current": wcur,
        "rets": rets,
        "prices_close": prices_close,
        "tickers": valid_cols,
    }


def prepare_ndarrays(
    weights_for_returns: pd.DataFrame,
    weights_current: pd.DataFrame,
    rets_daily: pd.DataFrame,
    universe_tickers: Sequence[str],
    price_index: pd.DatetimeIndex,
    use_float32: bool = True,
    prices_close_df: Optional[pd.DataFrame] = None,
) -> PreparedArrays:
    """
    Prepare contiguous ndarrays aligned for Numba kernels and return a PreparedArrays container.
    """
    dtype = _normalize_dtype(use_float32)
    aligned = _ensure_aligned_frames(
        weights_for_returns=weights_for_returns,
        weights_current=weights_current,
        rets_daily=rets_daily,
        universe_tickers=universe_tickers,
        price_index=price_index,
        prices_close_df=prices_close_df,
    )
    tickers: List[str] = aligned["tickers"]
    wret_arr = np.ascontiguousarray(aligned["weights_for_returns"].to_numpy(dtype=dtype))
    wcur_arr = np.ascontiguousarray(aligned["weights_current"].to_numpy(dtype=dtype))
    rets_arr = np.ascontiguousarray(aligned["rets"].to_numpy(dtype=dtype))
    rets_mask = np.ascontiguousarray(
        ~rets_daily.reindex(index=price_index, columns=tickers).isna().to_numpy()
    )

    prices_arr = None
    prices_mask = None
    prices_aligned = aligned["prices_close"]
    if prices_aligned is not None:
        prices_arr = np.ascontiguousarray(prices_aligned.fillna(0.0).to_numpy(dtype=dtype))
        prices_mask = np.ascontiguousarray(
            (prices_aligned.notna() & (prices_aligned > 0.0)).to_numpy()
        )

    return PreparedArrays(
        dates=price_index,
        tickers=list(tickers),
        dtype=dtype,
        weights_for_returns=wret_arr,
        weights_current=wcur_arr,
        rets=rets_arr,
        prices_close=prices_arr,
        rets_mask=rets_mask,
        prices_mask=prices_mask,
    )


def get_or_prepare_cached(
    cache_key: Tuple[Any, Any, Any, Any, Any, Any],
    weights_for_returns: pd.DataFrame,
    weights_current: pd.DataFrame,
    rets_daily: pd.DataFrame,
    universe_tickers: Sequence[str],
    price_index: pd.DatetimeIndex,
    use_float32: bool = True,
    prices_close_df: Optional[pd.DataFrame] = None,
) -> PreparedArrays:
    """
    Get PreparedArrays from cache or build and cache it.
    Caller is responsible for constructing a stable cache_key that represents
    universe and date span identity (e.g., (tuple(universe), start, end, use_float32, rebalance_freq, version_tag)).
    """
    pa = _PREPARED_CACHE.get(cache_key)
    if pa is not None:
        return pa
    pa = prepare_ndarrays(
        weights_for_returns=weights_for_returns,
        weights_current=weights_current,
        rets_daily=rets_daily,
        universe_tickers=universe_tickers,
        price_index=price_index,
        use_float32=use_float32,
        prices_close_df=prices_close_df,
    )
    _PREPARED_CACHE[cache_key] = pa
    return pa


def to_ndarrays(
    weights_daily: pd.DataFrame,
    rets_daily: pd.DataFrame,
    universe_tickers: List[str],
    price_index: pd.DatetimeIndex,
    use_float32: bool = True,
) -> Dict[str, Any]:
    """
    Legacy adapter preserved for existing call sites.
    Convert aligned DataFrames to contiguous ndarrays suitable for Numba kernels.

    Returns:
      - weights: ndarray [T, N]  (interpreted as weights_for_returns)
      - rets: ndarray [T, N]
      - mask: boolean ndarray [T, N] (True if valid return)
      - dates: DatetimeIndex (aligned)
      - tickers: List[str] (aligned)
      - dtype: np.dtype used
    """
    dtype = _normalize_dtype(use_float32)
    # Reindex to desired ticker order and timeline
    tickers: List[str] = [str(t) for t in universe_tickers]
    weights = weights_daily.reindex(columns=tickers).fillna(0.0)
    weights = weights.reindex(price_index, method="ffill").fillna(0.0)
    rets = rets_daily.reindex(index=price_index, columns=tickers).fillna(0.0)
    # Build mask for original NaNs (true valid where original non-NaN)
    orig_rets = rets_daily.reindex(index=price_index, columns=tickers)
    mask = ~orig_rets.isna().to_numpy()
    # Convert to contiguous arrays
    weights_arr = np.ascontiguousarray(weights.to_numpy(dtype=dtype))
    rets_arr = np.ascontiguousarray(rets.to_numpy(dtype=dtype))
    mask_arr = np.ascontiguousarray(mask)
    return {
        "weights": weights_arr,
        "rets": rets_arr,
        "mask": mask_arr,
        "dates": price_index,
        "tickers": list(tickers),
        "dtype": dtype,
    }
