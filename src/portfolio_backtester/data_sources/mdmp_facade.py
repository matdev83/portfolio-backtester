"""Central import boundary for ``market_data_multi_provider`` within portfolio_backtester."""

from __future__ import annotations

import importlib
import logging
import sys
import types
from typing import Any

logger = logging.getLogger(__name__)

MarketDataClient: Any

try:
    from market_data_multi_provider import MarketDataClient as _MarketDataClient  # type: ignore[attr-defined]
    from market_data_multi_provider import get_symbol as mdmp_get_symbol  # type: ignore[attr-defined]

    MarketDataClient = _MarketDataClient
except ImportError:
    MarketDataClient = None
    mdmp_get_symbol = None

try:
    from market_data_multi_provider.sp500 import (  # type: ignore[import-untyped]
        build_history as sp500_build_history,
        get_holdings as sp500_get_holdings,
        reset_cache as sp500_reset_cache,
    )
except ImportError:
    sp500_build_history = None
    sp500_get_holdings = None
    sp500_reset_cache = None

try:
    from market_data_multi_provider.russell import (  # type: ignore[import-untyped]
        get_constituents as russell_get_constituents,
    )
except ImportError:
    russell_get_constituents = None


def import_sp500_module() -> types.ModuleType:
    """Import and return ``market_data_multi_provider.sp500``."""
    return importlib.import_module("market_data_multi_provider.sp500")


def lookup_mdmp_symbol(ticker_upper: str) -> Any:
    """Resolve ticker via MDMP registry; returns spec or ``None`` if unavailable."""
    if mdmp_get_symbol is None:
        return None
    try:
        return mdmp_get_symbol(ticker_upper)
    except Exception as e:
        logger.debug("MDMP lookup failed for %s: %s", ticker_upper, e)
        return None


def sp500_reset_cache_if_loaded() -> None:
    """Clear SP500 LRU caches when the MDMP ``sp500`` module is already imported."""
    if sp500_reset_cache is None:
        return
    module = sys.modules.get("market_data_multi_provider.sp500")
    if isinstance(module, types.ModuleType):
        sp500_reset_cache()


__all__ = [
    "MarketDataClient",
    "import_sp500_module",
    "lookup_mdmp_symbol",
    "russell_get_constituents",
    "sp500_build_history",
    "sp500_get_holdings",
    "sp500_reset_cache",
    "sp500_reset_cache_if_loaded",
]
