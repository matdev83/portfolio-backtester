"""CUSIP ↔ Ticker mapping utilities.

This module maintains a local CSV database that is gradually extended
whenever we successfully resolve a ticker via the OpenFIGI REST API.

* Seed CSVs live in the data/ directory and are never modified.
* The live DB is data/cusip_mappings.csv – appended to atomically.
* The OpenFIGI key is expected in the environment variable
  OPENFIGI_API_KEY – it can be loaded automatically from a .env file
  in the project root (the file is listed in .gitignore so secrets do
  not leak into version control).
"""
from __future__ import annotations

from pathlib import Path
import csv
import json
import os
import time
import logging
from typing import Optional, Tuple
import re
import functools
import datetime as _dt

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper – load .env so that OPENFIGI_API_KEY appears in os.environ
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # project root
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
    except Exception as e:  # noqa: BLE001
        logger.debug("Could not parse .env file: %s", e)

OPENFIGI_API_KEY = os.getenv("OPENFIGI_API_KEY")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
# Immutable seeds shipped with the repo
SEED_FILES = [
    DATA_DIR / "cusip_tickers_seed.csv",
    DATA_DIR / "cusip_tickers_secondary.csv",
]
# Mutable DB that we append to as we discover new mappings
LIVE_DB_FILE = DATA_DIR / "cusip_mappings.csv"

# Ensure live DB exists with header row
if not LIVE_DB_FILE.exists():
    with LIVE_DB_FILE.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ticker", "cusip", "name", "source"])


class CusipMappingDB:
    """Main class used elsewhere in the code-base."""

    def __init__(self) -> None:
        # Load seeds once into dict – ticker → (cusip, name)
        self._cache: dict[str, Tuple[str, str]] = {}
        self._load_seeds()
        self._load_live_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def resolve(self, ticker: str, *, throttle: float = 1.0) -> Tuple[str, str]:
        """Return *(cusip, name)* for *ticker*.

        If the mapping is not yet cached we call OpenFIGI and persist the
        result so that subsequent runs are instant.
        """
        ticker = ticker.upper()
        if ticker in self._cache:
            return self._cache[ticker]

        # Attempt live lookup – only OpenFIGI for now
        cusip, name = self._lookup_openfigi(ticker, throttle=throttle)
        if cusip:
            self._append_to_db(ticker, cusip, name or "", source="openfigi")
            self._cache[ticker] = (cusip, name or "")
            return cusip, name or ""

        # EDGAR fallback – scrape latest filing HTML for CUSIP
        cusip, name = self._lookup_edgar(ticker, throttle=throttle)
        if cusip:
            self._append_to_db(ticker, cusip, name or "", source="edgar")
            self._cache[ticker] = (cusip, name or "")
            return cusip, name or ""

        # Web search fallback – DuckDuckGo HTML scrape
        cusip, name = self._lookup_duckduckgo(ticker, throttle=throttle)
        if cusip:
            self._append_to_db(ticker, cusip, name or "", source="duckduckgo")
            self._cache[ticker] = (cusip, name or "")
            return cusip, name or ""

        raise KeyError(f"CUSIP not found for ticker {ticker}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_seeds(self) -> None:
        for path in SEED_FILES:
            if not path.exists():
                continue
            try:
                with path.open() as fh:
                    reader = csv.reader(fh)
                    for row in reader:
                        if len(row) < 2:
                            continue  # need at least CUSIP + ticker
                        cusip = row[0]
                        ticker = row[1]
                        name = row[2] if len(row) >= 3 else ""
                        ticker = ticker.strip().upper()
                        cusip = cusip.strip()
                        name = name.strip()
                        if ticker and cusip and 8 <= len(cusip) <= 9:
                            # Prefer the first occurrence (seed order reflects priority)
                            self._cache.setdefault(ticker, (cusip, name))
            except Exception as e:  # noqa: BLE001
                logger.debug("Could not read seed %s: %s", path, e)

    def _load_live_db(self) -> None:
        try:
            with LIVE_DB_FILE.open() as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    ticker = row["ticker"].upper()
                    cusip = row["cusip"]
                    name = row.get("name", "")
                    if ticker and cusip:
                        self._cache[ticker] = (cusip, name)
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not read live DB: %s", e)

    # --------------------------------------------------------------
    # Network look-ups
    # --------------------------------------------------------------
    def _lookup_openfigi(self, ticker: str, *, throttle: float = 1.0) -> Tuple[Optional[str], Optional[str]]:
        if not OPENFIGI_API_KEY:
            logger.debug("OPENFIGI_API_KEY missing – skipping lookup")
            return None, None

        url = "https://api.openfigi.com/v3/mapping"
        headers = {
            "Content-Type": "application/json",
            "X-OPENFIGI-APIKEY": OPENFIGI_API_KEY,
        }

        def _query(payload):
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
            resp.raise_for_status()
            return resp.json()

        payload_us = [{"idType": "TICKER", "idValue": ticker, "exchCode": "US"}]
        payload_any = [{"idType": "TICKER", "idValue": ticker}]
        responses = []
        for pl in (payload_us, payload_any):
            try:
                responses.append(_query(pl))
            except Exception:
                continue
        for data in responses:
            if isinstance(data, list) and data and "data" in data[0] and data[0]["data"]:
                rec = data[0]["data"][0]
                cusip = rec.get("cusip")
                name = rec.get("securityDescription") or rec.get("securityName") or rec.get("securityDescription") or rec.get("securityName") or rec.get("name")
                if cusip and 8 <= len(cusip) <= 9:
                    time.sleep(throttle)
                    return cusip, name
        return None, None

    # --------------------------------------------------------------
    # DuckDuckGo last-resort lookup
    # --------------------------------------------------------------
    _CUSIP_RE = re.compile(r"\b(?=[0-9A-Z]*[A-Z])(?=[0-9A-Z]*[0-9])[0-9A-Z]{8,9}\b")

    def _lookup_duckduckgo(self, ticker: str, *, throttle: float = 1.0):
        """Scrape first search-result page for a 8-9-char alnum CUSIP."""
        try:
            time.sleep(throttle)  # polite delay before external call
            url = f"https://duckduckgo.com/html/?q={ticker}+cusip"
            headers = {"User-Agent": "portfolio-backtester/1.0 (+https://github.com/)"}
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            html = resp.text
            # DuckDuckGo puts each result snippet in <a class="result__a"> … </a>
            hits = set()
            for m in self._CUSIP_RE.finditer(html):
                hits.add(m.group(0))
            # Filter obvious false positives (all digits or alnum ok)
            hits = {h for h in hits if len(h) in (8, 9)}
            if len(hits) == 1:
                cusip = hits.pop()
                return cusip, ""  # name unknown from snippet
        except Exception as e:  # noqa: BLE001
            logger.debug("DuckDuckGo lookup failed for %s: %s", ticker, e)
        return None, None

    # --------------------------------------------------------------
    # EDGAR lookup
    # --------------------------------------------------------------

    _COMPANY_TICKERS_CACHE: dict[str, dict] | None = None

    def _company_tickers(self):
        if CusipMappingDB._COMPANY_TICKERS_CACHE is None:
            url = "https://www.sec.gov/files/company_tickers.json"
            hdr = {"User-Agent": "portfolio-backtester/1.0"}
            CusipMappingDB._COMPANY_TICKERS_CACHE = requests.get(url, headers=hdr, timeout=30).json()
        return CusipMappingDB._COMPANY_TICKERS_CACHE

    _CIK_SEARCH_CACHE: dict[str, str] = {}

    def _cik_for_ticker(self, ticker: str) -> Optional[str]:
        """Return a 10-char CIK string for *ticker* using cache or SEC search."""
        ticker = ticker.upper()
        # first, try in current/historic map
        for v in self._company_tickers().values():
            if v.get("ticker", "").upper() == ticker:
                return str(v["cik"]).zfill(10)

        # cache
        if ticker in CusipMappingDB._CIK_SEARCH_CACHE:
            return CusipMappingDB._CIK_SEARCH_CACHE[ticker]

        # fall back to live EDGAR company search (works for delisted tickers)
        try:
            hdr = {"User-Agent": "portfolio-backtester/1.0"}
            url = f"https://www.sec.gov/cgi-bin/browse-edgar?CIK={ticker}&owner=exclude&action=getcompany&output=json"
            js = requests.get(url, headers=hdr, timeout=20).json()
            cik = js.get("CIK") or js.get("cik")
            if cik:
                cik = str(cik).zfill(10)
                CusipMappingDB._CIK_SEARCH_CACHE[ticker] = cik
                return cik
        except Exception as e:
            logger.debug("CIK search failed for %s: %s", ticker, e)
        return None

    def _lookup_edgar(self, ticker: str, *, throttle: float = 1.0):
        try:
            # Map ticker→CIK using SEC company_tickers.json
            cik = self._cik_for_ticker(ticker)
            if not cik:
                return None, None

            sub_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            hdr = {"User-Agent": "portfolio-backtester/1.0"}
            sj = requests.get(sub_url, headers=hdr, timeout=30).json()
            recent = sj.get("filings", {}).get("recent", {})
            acc = recent.get("accessionNumber", [])
            prim = recent.get("primaryDocument", [])
            if not acc:
                return None, None
            # iterate first 5 filings
            for accession, doc in list(zip(acc, prim))[:5]:
                adash = accession.replace("-", "")
                doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{adash}/{doc}"
                try:
                    html = requests.get(doc_url, headers=hdr, timeout=30).text
                except Exception:
                    continue
                # search for CUSIP pattern following the word CUSIP
                m = re.search(r"CUSIP[^0-9A-Z]{0,20}([0-9]{3}[0-9A-Z]{5,6})", html, re.IGNORECASE)
                if m:
                    cusip = m.group(1)
                    if 8 <= len(cusip) <= 9:
                        return cusip, v.get("title", "") if cik else ""
            return None, None
        except Exception as e:
            logger.debug("EDGAR lookup failed for %s: %s", ticker, e)
            return None, None

    # --------------------------------------------------------------
    # Persistence
    # --------------------------------------------------------------
    def _append_to_db(self, ticker: str, cusip: str, name: str, *, source: str) -> None:
        try:
            with LIVE_DB_FILE.open("a", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow([ticker, cusip, name, source])
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not write to DB: %s", e)

if __name__ == "__main__":
    import argparse
    import sys
    import pandas as pd
    import requests

    parser = argparse.ArgumentParser(description="Update local CUSIP DB for all S&P 500 constituents")
    parser.add_argument("--update-missing", action="store_true", help="Fetch any still-unmapped tickers via OpenFIGI")
    parser.add_argument("--throttle", type=float, default=1.0, help="Seconds to wait between API calls (default 1.0)")
    args = parser.parse_args()

    if args.update_missing:
        # Scrape Wikipedia to obtain the master universe (same logic as tests)
        print("Downloading S&P 500 constituent list from Wikipedia…", file=sys.stderr)
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        symbols = set(tables[0].iloc[:, 0].tolist())
        symbols.update(tables[1].iloc[:, 1].dropna().tolist())
        symbols = {s.upper() for s in symbols if isinstance(s, str) and s.isalpha() and len(s) <= 5}

        db = CusipMappingDB()
        unresolved = [s for s in sorted(symbols) if s not in db._cache]
        print(f"Total unresolved tickers: {len(unresolved)}", file=sys.stderr)
        for i, sym in enumerate(unresolved, 1):
            try:
                cusip, _ = db.resolve(sym, throttle=args.throttle)
                print(f"[{i}/{len(unresolved)}] {sym} => {cusip}")
            except Exception as e:  # noqa: BLE001
                print(f"[{i}/{len(unresolved)}] {sym} FAILED: {e}", file=sys.stderr) 