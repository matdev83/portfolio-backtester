"""
Grab the longest public history of S&P-500 ETF holdings (default: SPY).

This module routes SPY holdings through ``portfolio_backtester.data_sources.mdmp_facade``,
which wraps ``market_data_multi_provider.sp500``. Persisted history lives in MDMP; PB does
not ship repo-local holdings parquet/csv warehouses.

MDMP may aggregate holdings from (priority order inside provider; not PB disk paths):
1. Kaggle S&P 500 Historical Data
2. SSGA daily basket XLSX (≈2011-present, 1-day lag)
3. SEC N-PORT-P XML (monthly, 2019-present) - *Planned for future MDMP update*
4. SEC N-Q HTML (quarterly, 2004-2018) - *Planned for future MDMP update*
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path
from typing import List, Tuple, Union, cast

import numpy as np
import pandas as pd

from portfolio_backtester.data_sources.mdmp_facade import (
    import_sp500_module,
    sp500_build_history as mdmp_build_history,
    sp500_get_holdings as mdmp_get_holdings,
    sp500_reset_cache_if_loaded,
)


logger = logging.getLogger(__name__)


# --- Fast in-process index for holdings-by-date lookups ---
# We avoid repeatedly scanning/filtering millions of rows for each requested date.
# Instead we load history once and perform a binary search over unique dates.

_HISTORY_DF_LOCAL: pd.DataFrame | None = None
_HISTORY_DATES: np.ndarray | None = None  # np.datetime64[ns] sorted unique dates
_HISTORY_SLICES: dict[np.datetime64, tuple[int, int]] | None = None  # date -> (start,end)


# Constants preserved for compatibility


def _normalize_holdings_date(date: Union[str, dt.date, pd.Timestamp]) -> pd.Timestamp:
    # Normalize to date boundary and strip timezone (if present)
    ts = pd.Timestamp(date)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


def _ensure_history_index_built() -> None:
    """Load and index holdings history once for fast date lookups."""
    global _HISTORY_DF_LOCAL, _HISTORY_DATES, _HISTORY_SLICES

    if _HISTORY_DF_LOCAL is not None and _HISTORY_DATES is not None and _HISTORY_SLICES is not None:
        return

    try:
        sp500_mod = import_sp500_module()
    except ImportError as e:
        raise ImportError("market-data-multi-provider package is required") from e

    hist = sp500_mod.get_holdings_history()
    if hist.empty:
        raise ValueError("SPY holdings history is not available (empty history)")

    df = hist.copy()

    if "date" not in df.columns or "ticker" not in df.columns:
        raise ValueError(f"Unexpected holdings history schema: columns={list(df.columns)}")

    # Normalize dates to tz-naive midnight for stable matching

    # Ensure stable ordering by date then weight (weight may not exist in some schemas)
    sort_cols = ["date"]
    if "weight_pct" in df.columns:
        sort_cols.append("weight_pct")
        df = df.sort_values(sort_cols, ascending=[True, False], kind="mergesort")
    else:
        df = df.sort_values(sort_cols, ascending=[True], kind="mergesort")

    # Build slice boundaries for each unique date (dates are contiguous after sort)
    dates_arr = df["date"].to_numpy(dtype="datetime64[ns]")
    if dates_arr.size == 0:
        raise ValueError("SPY holdings history is not available (no dates)")

    # unique dates (sorted) and slice boundaries
    unique_dates, first_idx = np.unique(dates_arr, return_index=True)
    # Compute end indices by shifting start indices; last ends at len(df)
    starts = first_idx
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = dates_arr.size

    slices: dict[np.datetime64, tuple[int, int]] = {}
    for i in range(unique_dates.size):
        slices[unique_dates[i]] = (int(starts[i]), int(ends[i]))

    _HISTORY_DF_LOCAL = df
    _HISTORY_DATES = unique_dates
    _HISTORY_SLICES = slices


def get_spy_holdings(
    date: Union[str, dt.date, pd.Timestamp], *, exact: bool = False
) -> pd.DataFrame:
    """Return a DataFrame of SPY constituent holdings for the supplied *date*.

    The DataFrame is sorted by descending weight (``weight_pct``). The *date*
    can be provided as ``datetime.date``, ``pandas.Timestamp`` or a ``YYYY-MM-DD``
    string.

    Parameters
    ----------
    date : str | datetime.date | pandas.Timestamp
        Target date for which to fetch the holdings.
    exact : bool, default ``False``
        If ``True``, require an exact match for *date*. If ``False`` (default)
        and the exact date is missing, the most recent previous trading day
        available (within the dataset) will be used.

    Returns
    -------
    pandas.DataFrame
        Holdings on the requested date ordered by descending index weight.
    """
    if mdmp_get_holdings is None:
        raise ImportError("market-data-multi-provider package is required")

    # Fast path: use the pre-indexed history for O(log N) lookup.
    # This avoids repeatedly scanning/filtering the full history for each date.
    try:
        _ensure_history_index_built()
        assert _HISTORY_DF_LOCAL is not None
        assert _HISTORY_DATES is not None
        assert _HISTORY_SLICES is not None

        target = np.datetime64(_normalize_holdings_date(date).to_datetime64(), "ns")

        # Exact match required
        if exact:
            sl = _HISTORY_SLICES.get(target)
            if sl is None:
                raise ValueError(f"No holdings for {pd.Timestamp(target).date()} (exact=True)")
            s, e = sl
            out = _HISTORY_DF_LOCAL.iloc[s:e]
            # Ensure expected output ordering (desc weight)
            if "weight_pct" in out.columns:
                out = out.sort_values("weight_pct", ascending=False)
            return out.reset_index(drop=True)

        # Non-exact: choose the most recent previous available date (never future)
        # Fast-fail if before earliest date
        if target < _HISTORY_DATES[0]:
            raise ValueError(
                f"No holdings for {pd.Timestamp(target).date()} (exact=False). "
                f"Data range: {pd.Timestamp(_HISTORY_DATES[0]).date()} to {pd.Timestamp(_HISTORY_DATES[-1]).date()}"
            )

        # idx = rightmost date <= target
        pos = int(np.searchsorted(_HISTORY_DATES, target, side="right") - 1)
        use_date = _HISTORY_DATES[pos]
        s, e = _HISTORY_SLICES[use_date]
        out = _HISTORY_DF_LOCAL.iloc[s:e]
        if "weight_pct" in out.columns:
            out = out.sort_values("weight_pct", ascending=False)
        return out.reset_index(drop=True)

    except Exception as fast_exc:
        # If anything about the fast path fails, fall back to MDMP's direct lookup.
        # This preserves behavior while allowing the fast path to handle the normal case.
        try:
            return cast(pd.DataFrame, mdmp_get_holdings(date, exact=exact))
        except Exception:
            raise fast_exc


def get_top_weight_sp500_components(
    date: Union[str, dt.date, pd.Timestamp], top_n: int = 30, *, exact: bool = False
) -> List[Tuple[str, float]]:
    """Return the *top_n* ticker symbols by weight in the S&P 500 (via SPY) for *date*.

    Parameters
    ----------
    date : "YYYY-MM-DD" | datetime.date | pandas.Timestamp
        Date for which to retrieve the universe constituents.
    top_n : int, default 30
        The number of top components to return.
    exact : bool, default ``False``
        If ``True``, only return components if we have data for that specific day.
        If ``False``, fallback to the most recent previous day if *date* is missing.

    Returns
    -------
    List[Tuple[str, float]]
        A list of (ticker, weight_pct) tuples sorted by descending weight.
    """
    if mdmp_get_holdings is None:
        raise ImportError("market-data-multi-provider package is required")

    df = get_spy_holdings(date, exact=exact)

    # Return as list of tuples (ticker, weight_pct) matching original signature
    # (Note: Original implementation returned list of tuples via _get_top_weight_sp500_components_cached
    #  MDMP's get_top_components return list[str], so we use get_holdings directly)

    top_df = df.head(top_n)
    return list(zip(top_df["ticker"], top_df["weight_pct"]))


def build_history(
    start: pd.Timestamp, end: pd.Timestamp, *, ignore_cache: bool = False
) -> pd.DataFrame:
    """Download & assemble SPY holdings between *start* and *end* via MDMP.

    Parameters
    ----------
    ignore_cache : bool, default ``False``
        If ``True``, bypass MDMP aggregated cache and rebuild from sources.
    """
    if mdmp_build_history is None:
        raise ImportError("market-data-multi-provider package is required")

    return cast(pd.DataFrame, mdmp_build_history(start, end, ignore_cache=ignore_cache))


def update_full_history(
    out_path: Path | None,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    *,
    rebuild: bool = False,
) -> None:
    """Rebuild SPY holdings history inside MDMP (no local parquet writes from PB).

    Portfolio Backtester does not persist holdings artifacts; MDMP owns storage.
    If ``out_path`` is set, it is ignored with a warning (script compatibility).
    """
    logger.info("Rebuilding SPY holdings history from %s to %s via MDMP...", start_date, end_date)
    build_history(start_date, end_date, ignore_cache=rebuild)
    if out_path:
        logger.warning(
            "Ignoring out_path=%s: PB does not write holdings parquet; use MDMP for persisted history.",
            out_path,
        )


def reset_history_cache() -> None:
    """Clear the in-memory holdings DataFrame *and* the LRU caches."""
    global _HISTORY_DF_LOCAL, _HISTORY_DATES, _HISTORY_SLICES
    _HISTORY_DF_LOCAL = None
    _HISTORY_DATES = None
    _HISTORY_SLICES = None
    sp500_reset_cache_if_loaded()


def get_all_historical_tickers() -> List[str]:
    """Get all unique tickers that have ever been in the S&P 500 history."""
    try:
        sp500_mod = import_sp500_module()
        hist = sp500_mod.get_holdings_history()
    except ImportError as e:
        raise ImportError("market-data-multi-provider package is required") from e

    if hist.empty:
        return []

    unique_tickers = hist["ticker"].unique()
    return sorted([str(t) for t in unique_tickers if t is not None and str(t).strip()])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download/Update S&P 500 (SPY) holdings history via MDMP"
    )
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument(
        "--rebuild", action="store_true", help="Ignore cache and rebuild from scratch"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Deprecated: ignored. MDMP persists holdings; PB does not write parquet.",
    )

    args = parser.parse_args()

    start = pd.Timestamp(args.start) if args.start else pd.Timestamp("2009-01-30")
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()

    out_path = Path(args.output) if args.output else None

    update_full_history(out_path, start, end, rebuild=args.rebuild)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
