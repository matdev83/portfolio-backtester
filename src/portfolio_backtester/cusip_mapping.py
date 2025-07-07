from __future__ import annotations
from pathlib import Path
import os
import logging
from typing import Tuple
import csv

from .cusip_mapping_logic.openfigi_lookup import lookup_openfigi
from .cusip_mapping_logic.edgar_lookup import lookup_edgar
from .cusip_mapping_logic.duckduckgo_lookup import lookup_duckduckgo
from .cusip_mapping_logic.persistence import append_to_db, load_seeds, load_live_db
from .cusip_mapping_logic.cli import main as cli_main

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
    def __init__(self) -> None:
        self._cache: dict[str, Tuple[str, str]] = {}
        load_seeds(SEED_FILES, self._cache)
        load_live_db(LIVE_DB_FILE, self._cache)

    def resolve(self, ticker: str, *, throttle: float = 1.0) -> Tuple[str, str]:
        ticker = ticker.upper()
        if ticker in self._cache:
            return self._cache[ticker]

        lookup_methods = []
        api_key = OPENFIGI_API_KEY
        if api_key:
            lookup_methods.append(("openfigi", lambda t, th: lookup_openfigi(api_key, t, throttle=th)))
        lookup_methods.extend([
            ("edgar", lookup_edgar),
            ("duckduckgo", lookup_duckduckgo),
        ])

        for source, lookup_function in lookup_methods:
            cusip, name = lookup_function(ticker, throttle=throttle)
            if cusip:
                return self._cache_and_store_mapping(ticker, cusip, name or "", source)

        logger.warning(f"CUSIP not found for ticker {ticker} after trying all lookup methods.")
        raise KeyError(f"CUSIP not found for ticker {ticker}")

    def _cache_and_store_mapping(self, ticker: str, cusip: str, name: str, source: str) -> Tuple[str, str]:
        append_to_db(LIVE_DB_FILE, ticker, cusip, name, source=source)
        self._cache[ticker.upper()] = (cusip, name)
        logger.info(f"Resolved and cached {ticker} -> {cusip} (Name: '{name}') via {source}.")
        return cusip, name

if __name__ == "__main__":
    cli_main()