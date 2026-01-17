"""OpenFIGI lookup helpers for mapping tickers to CUSIP.

Network calls are centralized in `_query_openfigi_api` so tests can mock it easily.
"""

from __future__ import annotations

from typing import Any

import requests


def _query_openfigi_api(api_key: str, url: str, payload: list[dict[str, Any]]) -> list[Any] | None:
    """Query OpenFIGI.

    Returns:
        Parsed JSON response as a list, or None on request errors.
    """
    try:
        resp = requests.post(
            url,
            headers={"X-OPENFIGI-APIKEY": api_key},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else None
    except requests.exceptions.RequestException:
        return None


def lookup_openfigi(api_key: str, ticker: str) -> tuple[str | None, str | None]:
    """Lookup (CUSIP, name) from OpenFIGI for a ticker."""
    if not api_key:
        return None, None

    t = ticker.strip().upper()
    url = "https://api.openfigi.com/v3/mapping"
    payload: list[dict[str, Any]] = [{"idType": "TICKER", "idValue": t, "exchCode": "US"}]

    result = _query_openfigi_api(api_key, url, payload)
    if not result:
        return None, None

    # Expected schema: list of dicts with optional `data` list
    first = result[0] if isinstance(result, list) and len(result) > 0 else None
    if not isinstance(first, dict):
        return None, None

    data_list = first.get("data")
    if not isinstance(data_list, list) or not data_list:
        return None, None

    row = data_list[0]
    if not isinstance(row, dict):
        return None, None

    cusip = row.get("cusip")
    name = row.get("securityDescription")
    return (str(cusip) if cusip else None, str(name) if name else None)


__all__ = ["lookup_openfigi", "_query_openfigi_api"]
