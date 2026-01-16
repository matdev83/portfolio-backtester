"""Symbol mapping between portfolio-backtester and market-data-multi-provider formats.

This module provides bidirectional symbol mapping:
- Local format: Simple tickers like "SPY", "^GSPC", "AAPL"
- Canonical format: EXCHANGE:SYMBOL like "AMEX:SPY", "SP:SPX", "NASDAQ:AAPL"

The mapping prioritizes resolution via the market-data-multi-provider registry
when available, with fallback to static mappings for common symbols.
"""

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# Direct mapping for special symbols (indices, non-standard tickers)
_SPECIAL_SYMBOL_MAP: dict[str, str] = {
    # S&P 500 Index
    "^GSPC": "SP:SPX",
    "SPX": "SP:SPX",
    "^SPX": "SP:SPX",
    # VIX family
    "^VIX": "CBOE:VIX",
    "VIX": "CBOE:VIX",
    "^VVIX": "CBOE:VVIX",
    "VVIX": "CBOE:VVIX",
    "^SKEW": "CBOE:SKEW",
    "SKEW": "CBOE:SKEW",
    "^VIX9D": "CBOE:VIX9D",
    "^VIX3M": "CBOE:VIX3M",
    "^VIX6M": "CBOE:VIX6M",
    # Treasury yields
    "^TNX": "TVC:TNX",
    "TNX": "TVC:TNX",
    "^TYX": "TVC:TYX",
    "TYX": "TVC:TYX",
    "^IRX": "TVC:IRX",
    "IRX": "TVC:IRX",
    "^FVX": "TVC:FVX",
    "FVX": "TVC:FVX",
    # Russell 2000 Index
    "^RUT": "TVC:RUT",
    "RUT": "TVC:RUT",
    # NASDAQ Composite
    "^IXIC": "NASDAQ:IXIC",
    # Dow Jones
    "^DJI": "DJ:DJI",
}

# Known NASDAQ-listed stocks (for exchange prefix assignment)
_NASDAQ_STOCKS: set[str] = {
    "AAPL", "ADBE", "AMGN", "AMZN", "AVGO", "CMCSA", "COST", "CSCO",
    "GOOG", "GOOGL", "HON", "INTC", "INTU", "META", "MSFT", "NFLX",
    "NVDA", "PEP", "QCOM", "TSLA", "TXN", "WBA",
    # Add QQQ and NASDAQ ETFs
    "QQQ", "QQQE", "TQQQ", "SQQQ",
}

# Known AMEX-listed ETFs
_AMEX_ETFS: set[str] = {
    "SPY", "IWM", "DIA", "RSP", "GLD", "SLV", "VTI", "VEA", "VWO",
    "XLB", "XLE", "XLF", "XLI", "XLP", "XLU", "XLV", "XLK", "XLY", "XLRE",
    "UVXY", "SVXY", "VXX", "VIXY",
    "TLT", "IEF", "SHY", "BND", "AGG",
    "EEM", "EFA", "IEMG",
}

# Reverse mapping cache for canonical -> local
_REVERSE_MAP: dict[str, str] = {}


def _build_reverse_map() -> None:
    """Build reverse mapping from canonical IDs to local tickers."""
    global _REVERSE_MAP
    if _REVERSE_MAP:
        return
    for local, canonical in _SPECIAL_SYMBOL_MAP.items():
        # Prefer simpler forms (without ^) for reverse mapping
        if canonical not in _REVERSE_MAP or not local.startswith("^"):
            _REVERSE_MAP[canonical] = local


@lru_cache(maxsize=4096)
def to_canonical_id(ticker: str) -> str:
    """Convert local ticker to market-data-multi-provider canonical ID.

    Resolution order:
    1. Check special symbol map (indices, yield symbols)
    2. Try MDMP registry lookup via alias
    3. Determine exchange prefix based on known lists
    4. Default to NYSE for unknown symbols

    Args:
        ticker: Local ticker symbol (e.g., "SPY", "^GSPC", "AAPL")

    Returns:
        Canonical ID (e.g., "AMEX:SPY", "SP:SPX", "NASDAQ:AAPL")

    Examples:
        >>> to_canonical_id("SPY")
        'AMEX:SPY'
        >>> to_canonical_id("^GSPC")
        'SP:SPX'
        >>> to_canonical_id("AAPL")
        'NASDAQ:AAPL'
    """
    ticker_upper = ticker.strip().upper()

    # 1. Check special symbol map first
    if ticker_upper in _SPECIAL_SYMBOL_MAP:
        return _SPECIAL_SYMBOL_MAP[ticker_upper]

    # 2. Already in canonical format?
    if ":" in ticker:
        return ticker

    # 3. Try MDMP registry lookup (if available)
    try:
        from market_data_multi_provider import get_symbol

        spec = get_symbol(ticker_upper)
        if spec is not None:
            return spec.symbol_id
    except ImportError:
        pass  # MDMP not installed, continue with static mapping
    except Exception as e:
        logger.debug(f"MDMP lookup failed for {ticker_upper}: {e}")

    # 4. Determine exchange prefix based on known lists
    if ticker_upper in _NASDAQ_STOCKS:
        return f"NASDAQ:{ticker_upper}"

    if ticker_upper in _AMEX_ETFS:
        return f"AMEX:{ticker_upper}"

    # 5. Default to NYSE for unknown stock symbols
    # Most individual stocks not explicitly listed are NYSE
    return f"NYSE:{ticker_upper}"


@lru_cache(maxsize=4096)
def from_canonical_id(canonical_id: str) -> str:
    """Convert canonical ID back to local ticker format.

    Args:
        canonical_id: Canonical ID (e.g., "AMEX:SPY", "SP:SPX")

    Returns:
        Local ticker (e.g., "SPY", "^GSPC")

    Examples:
        >>> from_canonical_id("AMEX:SPY")
        'SPY'
        >>> from_canonical_id("SP:SPX")
        '^GSPC'
    """
    _build_reverse_map()

    # Check reverse mapping
    if canonical_id in _REVERSE_MAP:
        return _REVERSE_MAP[canonical_id]

    # Extract ticker from canonical format
    if ":" in canonical_id:
        return canonical_id.split(":", 1)[1]

    return canonical_id


def get_exchange_prefix(ticker: str) -> str:
    """Get the exchange prefix for a ticker symbol.

    Args:
        ticker: Ticker symbol

    Returns:
        Exchange prefix (e.g., "NASDAQ", "NYSE", "AMEX")
    """
    ticker_upper = ticker.strip().upper()

    if ticker_upper in _NASDAQ_STOCKS:
        return "NASDAQ"
    if ticker_upper in _AMEX_ETFS:
        return "AMEX"
    if ticker_upper.startswith("^"):
        # Index symbols have various prefixes
        canonical = to_canonical_id(ticker_upper)
        if ":" in canonical:
            return canonical.split(":")[0]
    return "NYSE"  # Default


def is_special_symbol(ticker: str) -> bool:
    """Check if a ticker is a special symbol (index, yield, etc.).

    Args:
        ticker: Ticker symbol

    Returns:
        True if the ticker is a special symbol
    """
    return ticker.strip().upper() in _SPECIAL_SYMBOL_MAP


def clear_cache() -> None:
    """Clear the mapping caches."""
    to_canonical_id.cache_clear()
    from_canonical_id.cache_clear()
    global _REVERSE_MAP
    _REVERSE_MAP = {}


def get_stooq_symbol(ticker: str) -> str:
    """Get the Stooq-formatted symbol for a ticker.

    Args:
        ticker: Ticker symbol (e.g., "AAPL", "SPY")

    Returns:
        Stooq symbol format (e.g., "aapl.us", "spy.us")
    """
    ticker_upper = ticker.strip().upper()

    # Special cases for indices
    if ticker_upper in ("^GSPC", "SPX", "^SPX"):
        return "^spx"
    if ticker_upper.startswith("^"):
        # Other indices - remove ^ and lowercase
        return ticker_upper[1:].lower()

    # Standard US stocks/ETFs
    return f"{ticker_upper.lower()}.us"


__all__ = [
    "to_canonical_id",
    "from_canonical_id",
    "get_exchange_prefix",
    "is_special_symbol",
    "clear_cache",
    "get_stooq_symbol",
]
