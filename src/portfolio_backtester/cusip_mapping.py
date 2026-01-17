"""CUSIP mapping facade.

This module provides `CusipMappingDB`, a small in-memory cache that can be
seeded from local files and optionally refreshed from a live DB file. It is
implemented with Dependency Inversion (DIP) so loaders can be mocked in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from portfolio_backtester.cusip_mapping_logic.edgar_lookup import lookup_edgar
from portfolio_backtester.cusip_mapping_logic.openfigi_lookup import lookup_openfigi
from portfolio_backtester.interfaces.database_loader_interface import (
    ILiveDBLoader,
    ISeedLoader,
    create_live_db_loader,
    create_seed_loader,
)

# Optional API key (may be set by user environment/config in the future)
OPENFIGI_API_KEY: str | None = None


def lookup_duckduckgo(_ticker: str) -> tuple[str | None, str | None]:
    """Fallback lookup stub.

    This project intentionally avoids ad-hoc web scraping. Kept as a mock target for tests.
    """
    return None, None


@dataclass(frozen=True)
class _DefaultPaths:
    seed_files: list[Path]
    live_db_file: Path


def _default_paths() -> _DefaultPaths:
    # Keep paths stable and relative to this module; files may or may not exist.
    base = Path(__file__).resolve().parent
    return _DefaultPaths(
        seed_files=[base / "data" / "cusip_seeds.csv"],
        live_db_file=base / "data" / "cusip_live_db.csv",
    )


class CusipMappingDB:
    """Resolve tickers to (CUSIP, name) using a cached mapping."""

    def __init__(
        self,
        *,
        seed_loader: ISeedLoader | None = None,
        live_db_loader: ILiveDBLoader | None = None,
    ) -> None:
        self._cache: Dict[str, Tuple[str, str]] = {}

        paths = _default_paths()
        self._seed_files = paths.seed_files
        self._live_db_file = paths.live_db_file

        self._seed_loader = seed_loader or create_seed_loader()
        self._live_db_loader = live_db_loader or create_live_db_loader()

        # Populate cache
        self._seed_loader.load_seeds(self._seed_files, self._cache)
        self._live_db_loader.load_live_db(self._live_db_file, self._cache)

    def resolve(self, ticker: str) -> tuple[str, str]:
        """Resolve a ticker into (CUSIP, company name).

        Raises:
            KeyError: If the ticker cannot be resolved.
        """
        t = ticker.strip().upper()
        if not t:
            raise KeyError("CUSIP not found for ticker ")

        if t in self._cache:
            cusip_cached, name_cached = self._cache[t]
            return str(cusip_cached), str(name_cached)

        # External lookups (mock-friendly)
        if OPENFIGI_API_KEY:
            cusip_opt, name_opt = lookup_openfigi(OPENFIGI_API_KEY, t)
            if cusip_opt:
                self._cache[t] = (cusip_opt, name_opt or "")
                return cusip_opt, name_opt or ""

        cusip_opt, name_opt = lookup_edgar(t)
        if cusip_opt:
            self._cache[t] = (cusip_opt, name_opt or "")
            return cusip_opt, name_opt or ""

        cusip_opt, name_opt = lookup_duckduckgo(t)
        if cusip_opt:
            self._cache[t] = (cusip_opt, name_opt or "")
            return cusip_opt, name_opt or ""

        raise KeyError(f"CUSIP not found for ticker {t}")


__all__ = [
    "CusipMappingDB",
    "OPENFIGI_API_KEY",
    "lookup_openfigi",
    "lookup_edgar",
    "lookup_duckduckgo",
]
