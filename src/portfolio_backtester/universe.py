from __future__ import annotations

import datetime as dt
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

from portfolio_backtester.data_sources.mdmp_facade import russell_get_constituents

from .universe_data.spy_holdings import get_spy_holdings
from .interfaces.date_normalizer_interface import normalize_date_polymorphic

"""Utility helpers for universe creation.

This module provides a single convenience wrapper
``get_top_weight_sp500_components`` that exposes the top-weighted
constituents of the S&P 500 (approximated by SPY ETF holdings) for any
point in time.

The heavy lifting of downloading, cleaning and caching the full holdings
history is delegated to MDMP via :pymod:`portfolio_backtester.universe_data.spy_holdings`; we
simply reuse that functionality and add:

1. A lightweight in-process cache keyed by ``(date, n, exact)`` so that
   repeated optimiser calls are instantaneous.
2. A thin API that returns a plain :class:`list` of ticker symbols which
   is what the strategy layer needs.

The helper **does not** introduce any look-ahead bias: if an exact file
for *date* is missing we fall back **only** to the most recent **earlier**
trading day, never to the future.
"""

# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _normalize_date(date: Union[str, dt.date, pd.Timestamp]) -> pd.Timestamp:
    """Convert a variety of date representations to *normalised* Timestamp.

    We normalise to 00:00 so that different intraday timestamps for the
    same calendar day map to the same cache key.

    This function uses polymorphic date normalization instead of isinstance
    checks, following the Open/Closed Principle.
    """
    return normalize_date_polymorphic(date)


_SPY_HOLDINGS_ALIAS_PATH = Path(__file__).resolve().parent / "spy_holdings_ticker_aliases.json"
_SPY_HOLDINGS_ALIASES_CACHE: Dict[str, str] | None = None


def _spy_holdings_aliases() -> Dict[str, str]:
    """Load JSON map of legacy SPY basket symbols to MDMP/yfinance-friendly tickers."""
    global _SPY_HOLDINGS_ALIASES_CACHE
    if _SPY_HOLDINGS_ALIASES_CACHE is not None:
        return _SPY_HOLDINGS_ALIASES_CACHE
    merged: Dict[str, str] = {}
    if _SPY_HOLDINGS_ALIAS_PATH.is_file():
        with _SPY_HOLDINGS_ALIAS_PATH.open(encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            merged = {
                str(k).strip().upper(): str(v).strip().upper()
                for k, v in loaded.items()
                if isinstance(k, str) and isinstance(v, str)
            }
    _SPY_HOLDINGS_ALIASES_CACHE = merged
    return _SPY_HOLDINGS_ALIASES_CACHE


def normalize_spy_holding_ticker(symbol: str) -> str:
    """Map a raw SPY holdings label to a ticker data providers typically resolve.

    Extend mappings by editing ``spy_holdings_ticker_aliases.json`` beside this module.

    Args:
        symbol: Raw ticker from SPY holdings history.

    Returns:
        Mapped symbol when an alias exists; otherwise the trimmed uppercase input.
    """
    key = symbol.strip().upper()
    return _spy_holdings_aliases().get(key, key)


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
    return [normalize_spy_holding_ticker(str(t)) for t in top]


def get_all_historical_sp500_components() -> List[str]:
    """Return a list of ALL unique tickers that have ever been in the S&P 500 history."""
    from .universe_data.spy_holdings import get_all_historical_tickers

    return get_all_historical_tickers()


@lru_cache(maxsize=64)
def _cached_current_sp500_components(date_key: str, exact: bool) -> tuple[str, ...]:
    from .universe_data.spy_holdings import get_spy_holdings

    df = get_spy_holdings(date_key, exact=exact)
    if "ticker" not in df.columns:
        raise ValueError("SPY holdings missing 'ticker' column")
    tickers = df["ticker"].dropna().astype(str).tolist()
    return tuple(tickers)


def get_current_sp500_components(
    as_of_date: Union[str, dt.date, pd.Timestamp, None] = None,
    *,
    exact: bool = False,
) -> List[str]:
    """Return current S&P 500 constituents as of a given date (default: latest available).

    This mirrors the paper's survivorship-biased universe (current constituents).
    """
    if as_of_date is None:
        as_of_date = pd.Timestamp.today().normalize()

    date_key = _normalize_date(as_of_date).strftime("%Y-%m-%d")
    return [
        normalize_spy_holding_ticker(str(t))
        for t in _cached_current_sp500_components(date_key, exact)
    ]


def get_all_historical_russell_1000_components() -> List[str]:
    """Return a list of ALL unique tickers that have ever been in the Russell 1000 history."""
    try:
        if russell_get_constituents is None:
            return []
        result = russell_get_constituents("russell_1000")
        return list(result) if result is not None else []
    except ImportError:
        return []


def get_all_historical_russell_2000_components() -> List[str]:
    """Return a list of ALL unique tickers that have ever been in the Russell 2000 history."""
    try:
        if russell_get_constituents is None:
            return []
        result = russell_get_constituents("russell_2000")
        return list(result) if result is not None else []
    except ImportError:
        return []


__all__ = [
    "get_top_weight_sp500_components",
    "get_all_historical_sp500_components",
    "get_current_sp500_components",
    "get_all_historical_russell_1000_components",
    "get_all_historical_russell_2000_components",
    "normalize_spy_holding_ticker",
]
