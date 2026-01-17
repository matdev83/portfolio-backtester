"""EDGAR-based helpers for mapping tickers to identifiers.

These utilities are intentionally lightweight and designed to be mockable in tests.
"""

from __future__ import annotations

from typing import Any

import requests


def _company_tickers() -> dict[str, dict[str, Any]]:
    """Fetch the SEC company tickers mapping.

    Returns:
        A dict mapping string IDs to dicts containing (at least) `ticker` and `cik`.
    """
    resp = requests.get("https://www.sec.gov/files/company_tickers.json", timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if isinstance(payload, dict):
        return payload
    return {}


def _cik_for_ticker(ticker: str) -> str | None:
    """Resolve EDGAR CIK for a ticker.

    Args:
        ticker: The equity ticker (e.g., "AAPL").

    Returns:
        10-digit zero-padded CIK string, or None if not found.
    """
    t = ticker.strip().upper()

    try:
        mapping = _company_tickers()
        for _k, v in mapping.items():
            if isinstance(v, dict) and str(v.get("ticker", "")).upper() == t:
                cik_raw = str(v.get("cik", "")).strip()
                if cik_raw.isdigit():
                    return cik_raw.zfill(10)
    except Exception:
        # Allow callers/tests to handle network failures explicitly if they call _company_tickers()
        pass

    # Fallback: attempt a simple API lookup if present.
    resp = requests.get(f"https://data.sec.gov/submissions/CIK{t}.json", timeout=30)
    resp.raise_for_status()
    data = resp.json()
    cik_raw = str(data.get("cik", "")).strip()
    if cik_raw.isdigit():
        return cik_raw.zfill(10)
    return None


def lookup_edgar(ticker: str) -> tuple[str | None, str | None]:
    """Lookup (CUSIP, name) via EDGAR.

    Note: EDGAR does not directly provide CUSIP for all securities without further parsing.
    This function is intentionally conservative and may return (None, None).
    """
    cik = _cik_for_ticker(ticker)
    if cik is None:
        return None, None
    # Placeholder: parsing CUSIP from EDGAR would require additional endpoints/data.
    # Keep this minimal and mock-friendly.
    return None, None


__all__ = ["_company_tickers", "_cik_for_ticker", "lookup_edgar"]
