from __future__ import annotations

"""Utility helpers for universe creation.

This module provides a single convenience wrapper
``get_top_weight_sp500_components`` that exposes the top-weighted
constituents of the S&P 500 (approximated by SPY ETF holdings) for any
point in time.

The heavy lifting of downloading, cleaning and caching the full holdings
history is implemented in :pymod:`portfolio_backtester.spy_holdings` – we
simply reuse that functionality and add:

1. A lightweight in-process cache keyed by ``(date, n, exact)`` so that
   repeated optimiser calls are instantaneous.
2. A thin API that returns a plain :class:`list` of ticker symbols which
   is what the strategy layer needs.

The helper **does not** introduce any look-ahead bias: if an exact file
for *date* is missing we fall back **only** to the most recent **earlier**
trading day, never to the future.
"""

from functools import lru_cache
import datetime as dt
from typing import List, Union

import pandas as pd

from .universe_data.spy_holdings import get_spy_holdings

# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _normalize_date(date: Union[str, dt.date, pd.Timestamp]) -> pd.Timestamp:
    """Convert a variety of date representations to *normalised* Timestamp.

    We normalise to 00:00 so that different intraday timestamps for the
    same calendar day map to the same cache key.
    """
    if isinstance(date, pd.Timestamp):
        ts = date
    elif isinstance(date, str):
        ts = pd.Timestamp(date)
    elif isinstance(date, dt.date):
        ts = pd.Timestamp(date)
    else:
        raise TypeError("date must be str, datetime.date or pandas.Timestamp")

    return ts.normalize()


@lru_cache(maxsize=4096)
def _cached_top(date_str: str, n: int, exact: bool) -> tuple[str, ...]:
    """Cached backend returning *tuple* for hashability."""
    df = get_spy_holdings(date_str, exact=exact)
    return tuple(df.head(n)["ticker"].tolist())


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def get_top_weight_sp500_components(
    date: Union[str, dt.date, pd.Timestamp],
    n: int = 50,
    *,
    exact: bool = False,
) -> List[str]:
    """Return tickers of the *n* heaviest S&P 500 constituents on *date*.

    Parameters
    ----------
    date
        Target date (any of ``str`` *YYYY-MM-DD*, ``datetime.date`` or
        :class:`pandas.Timestamp`).
    n
        Number of top holdings to return (must be > 0).
    exact
        If *True* require an *exact* match for ``date``.  If *False*
        (default) and the requested trading day is missing, the most
        recent **previous** date in the historical dataset is used.  No
        forward-looking substitution is ever performed.

    Returns
    -------
    list[str]
        List of ticker symbols ordered by descending index weight.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")

    # Ensure we have a deterministic cache key
    date_norm = _normalize_date(date)
    date_key = date_norm.strftime("%Y-%m-%d")

    top = _cached_top(date_key, n, exact)
    # Could be shorter than *n* if the holdings file itself is shorter.
    return list(top)


__all__ = ["get_top_weight_sp500_components"] 