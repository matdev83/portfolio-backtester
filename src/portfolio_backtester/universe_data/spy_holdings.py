"""
Grab the longest public history of S&P-500 ETF holdings (default: SPY).

This module has been migrated to use `market-data-multi-provider` as the backend.
It now acts as a compatibility wrapper around `market_data_multi_provider.sp500`.

Original Data sources – in priority order:
1. Kaggle S&P 500 Historical Data
2. SSGA daily basket XLSX (≈2011-present, 1-day lag)
3. SEC N-PORT-P XML (monthly, 2019-present) - *Planned for future MDMP update*
4. SEC N-Q HTML (quarterly, 2004-2018) - *Planned for future MDMP update*
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

# Import from MDMP
try:
    from market_data_multi_provider.sp500 import (
        build_history as mdmp_build_history,
        get_holdings as mdmp_get_holdings,
        get_top_components as mdmp_get_top_components,
        reset_cache as mdmp_reset_cache,
    )
except ImportError:
    # Fallback for when mdmp is not installed (e.g. CI without dependencies)
    # This ensures the module is at least importable
    mdmp_build_history = None  # type: ignore
    mdmp_get_holdings = None  # type: ignore
    mdmp_get_top_components = None  # type: ignore
    mdmp_reset_cache = None  # type: ignore


logger = logging.getLogger(__name__)

# Constants preserved for compatibility
TICKER = "SPY"
CIK = "0000884394"


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
        
    try:
        return mdmp_get_holdings(date, exact=exact)
    except ValueError as e:
        # Re-raise with same message if possible, or let it bubble up
        # MDMP raises ValueError if data not found, which matches original behavior
        raise e


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
    """Download & assemble SPY holdings between *start* and *end*.

    Parameters
    ----------
    ignore_cache : bool, default ``False``
        If ``True`` the function bypasses any aggregated parquet in
        cache and rebuilds the DataFrame from scratch.
    """
    if mdmp_build_history is None:
        raise ImportError("market-data-multi-provider package is required")
        
    return mdmp_build_history(start, end, ignore_cache=ignore_cache)


def update_full_history(
    out_path: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    *,
    rebuild: bool = False,
) -> None:
    """Update or rebuild the SPY holdings parquet.
    
    This function is maintained for script compatibility.
    """
    logger.info(f"Updating history from {start_date} to {end_date} via MDMP...")
    df = build_history(start_date, end_date, ignore_cache=rebuild)
    
    # Save to specified path if provided
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path)
        logger.info(f"History saved to {out_path}")


def reset_history_cache() -> None:
    """Clear the in-memory holdings DataFrame *and* the LRU caches."""
    if mdmp_reset_cache:
        mdmp_reset_cache()


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
        "--output", type=str, help="Output parquet file path"
    )
    
    args = parser.parse_args()
    
    start = pd.Timestamp(args.start) if args.start else pd.Timestamp("2009-01-30")
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    
    out_path = Path(args.output) if args.output else Path("data/spy_holdings_full.parquet")
    
    update_full_history(out_path, start, end, rebuild=args.rebuild)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
