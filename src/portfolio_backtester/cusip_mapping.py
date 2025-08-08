from __future__ import annotations
from pathlib import Path
import os
import logging
from typing import Tuple, List, Callable, Optional
import csv

from .cusip_mapping_logic.openfigi_lookup import lookup_openfigi
from .cusip_mapping_logic.edgar_lookup import lookup_edgar
from .cusip_mapping_logic.duckduckgo_lookup import lookup_duckduckgo
from .cusip_mapping_logic.cli import main as cli_main
from .interfaces.database_loader_interface import (
    ISeedLoader,
    ILiveDBLoader,
    ILiveDBWriter,
    create_seed_loader,
    create_live_db_loader,
    create_live_db_writer,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"
if ENV_FILE.exists():
    try:
        with ENV_FILE.open() as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                if key not in os.environ:
                    os.environ[key] = val
    except Exception as e:
        logger.debug("Could not parse .env file: %s", e)

OPENFIGI_API_KEY = os.getenv("OPENFIGI_API_KEY")

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
SEED_FILES = [
    DATA_DIR / "cusip_tickers_seed.csv",
    DATA_DIR / "cusip_tickers_secondary.csv",
]
LIVE_DB_FILE = DATA_DIR / "cusip_mappings.csv"

if not LIVE_DB_FILE.exists():
    with LIVE_DB_FILE.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ticker", "cusip", "name", "source"])


class CusipMappingDB:
    def __init__(
        self,
        seed_loader: Optional[ISeedLoader] = None,
        live_db_loader: Optional[ILiveDBLoader] = None,
        live_db_writer: Optional[ILiveDBWriter] = None,
    ) -> None:
        self._cache: dict[str, Tuple[str, str]] = {}

        # Use dependency injection with default factories if not provided
        if seed_loader is None:
            seed_loader = create_seed_loader()
        if live_db_loader is None:
            live_db_loader = create_live_db_loader()
        if live_db_writer is None:
            live_db_writer = create_live_db_writer()

        # Store the writer for later use
        self._live_db_writer = live_db_writer

        # Load data using the injected dependencies
        seed_loader.load_seeds(SEED_FILES, self._cache)
        live_db_loader.load_live_db(LIVE_DB_FILE, self._cache)

    def resolve(self, ticker: str, *, throttle: float = 1.0) -> Tuple[str, str]:
        ticker = ticker.upper()
        if ticker in self._cache:
            return self._cache[ticker]

        lookup_methods: List[Tuple[str, Callable[..., Tuple[Optional[str], Optional[str]]]]] = []
        api_key = OPENFIGI_API_KEY
        if api_key:
            lookup_methods.append(
                (
                    "openfigi",
                    lambda t, th: lookup_openfigi(api_key, t, throttle=th),
                )
            )
        lookup_methods.extend(
            [
                ("edgar", lookup_edgar),
                ("duckduckgo", lookup_duckduckgo),
            ]
        )

        for source, lookup_function in lookup_methods:
            cusip, name = lookup_function(ticker, throttle=throttle)
            if cusip:
                return self._cache_and_store_mapping(ticker, cusip, name or "", source)

        logger.warning(f"CUSIP not found for ticker {ticker} after trying all lookup methods.")
        raise KeyError(f"CUSIP not found for ticker {ticker}")

    def _cache_and_store_mapping(
        self, ticker: str, cusip: str, name: str, source: str
    ) -> Tuple[str, str]:
        self._live_db_writer.append_to_db(LIVE_DB_FILE, ticker, cusip, name, source)
        self._cache[ticker.upper()] = (cusip, name)
        logger.info(f"Resolved and cached {ticker} -> {cusip} (Name: '{name}') via {source}.")
        return cusip, name


if __name__ == "__main__":
    cli_main()
