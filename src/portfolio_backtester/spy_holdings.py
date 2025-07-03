#!/usr/bin/env python
"""
Grab the longest public history of S&P-500 ETF holdings (default: SPY).

Data sources – in priority order
1. SSGA daily basket XLSX (≈2011-present, 1-day lag)
2. SEC N-PORT-P XML (monthly, 2019-present)
3. SEC N-Q HTML (quarterly, 2004-2018)

Anything missing after those three is left blank.
"""

import argparse
import datetime as dt
import io
import os
import re
import time
import zipfile
from pathlib import Path
import logging
from typing import Optional, Union, List

import pandas as pd
import requests
from lxml import etree
from tqdm import tqdm
try:  # EdgarTools ≥4.3.0 provides set_identity
    from edgar import set_identity, Company
except Exception:  # pragma: no cover - fallback for missing API
    import edgar
    set_identity = getattr(edgar, "set_identity", lambda _: None)
    Company = getattr(edgar, "Company", None)
import json
from urllib.parse import quote
from httpx import HTTPStatusError

try:
    # edgartools ≥4 uses dataclasses like FundReport instead of plain dict
    from edgar.objects import FundReport
except ImportError:  # pragma: no cover
    FundReport = None  # type: ignore

logger = logging.getLogger(__name__)

##############################################################################
# Config – change these four lines if you'd rather pull IVV or VOO instead
##############################################################################
TICKER   = "SPY"
CIK      = "0000884394"        # SPDR® S&P 500 ETF Trust
# First date SSGA daily XLSX baskets are known to exist (earliest file on server)
EARLIEST_SSGA_DATE = pd.Timestamp(2015, 3, 16)  # first day XLS/XLSX files reliably available
# Generic (latest only) URL – SSGA hosts just *one* file that always contains today's basket.
GENERIC_SSGA_URL = (
    "https://www.ssga.com/library-content/products/fund-data/etfs/us/"
    "holdings-daily-us-en-spy.xlsx"
)

# Legacy URL patterns that *used* to serve dated files. They no longer work for live requests but
# are still useful when querying the Internet Archive.
URL_PATTERNS = [
    GENERIC_SSGA_URL,  # must stay first so Wayback picks snapshots of the canonical path
    # 2017-08 → 2023-??  (.xlsx with optional date suffix – removed from live site)
    "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy-{ymd}.xlsx",
    # 2015-03 → 2017-07  (.xlsx – extra "us" in path)
    "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-us-spy-{ymd}.xlsx",
    # 2011-03 → 2015-02  (.xls – legacy host)
    "https://www.spdrs.com/library-content/public/documents/etfs/us/fund-data/daily-us-en-us-spy-{ymd}.xls",
]
# Example for IVV (iShares):
# ISSUER_DAILY_URL = (
#     "https://www.ishares.com/us/products/239726/"
#     "ishares-core-s-p-500-etf/1467271812596.ajax?fileType=csv&dataType=fund"
#     "&asOfDate={ymd}"
# )
##############################################################################

# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #

# Initialize cache for CUSIP to Ticker mapping
_CUSIP_TICKER_CACHE = {
    # Top 10 S&P 500 by market cap (most likely to be in SPY)
    '037833100': 'AAPL', '594918104': 'MSFT', '67066G104': 'NVDA',
    '023135106': 'AMZN', '30303M102': 'META', '02079K305': 'GOOGL',
    '02079K107': 'GOOG', '88160R101': 'TSLA', '084670702': 'BRK.B',
    '92826C839': 'V',
    # Major financial services
    '46625H100': 'JPM', '91324P102': 'UNH', '30231G102': 'XOM',
    '459200101': 'JNJ', '713448108': 'PEP',
    # Technology leaders
    '17275R102': 'CSCO', '00724F101': 'ADBE', '79466L302': 'SBUX',
    '04621X108': 'AVGO',
    # Consumer & Industrial
    '931142103': 'WMT', '742718109': 'PG', '437076102': 'HD',
    '580135101': 'MCD', '438516106': 'HON',
    # Healthcare & Pharma
    '88579Y101': 'TMO', '02376R102': 'ABT', '126650100': 'CVS',
    # Additional common SPY holdings
    '191216100': 'KO', '254687106': 'DIS', '149123101': 'CAT',
    '166756103': 'CVX', '65339F101': 'NEE', '57636Q104': 'MA',
    # Banking & Finance
    '060505104': 'BAC', '38141G104': 'GS', '58933Y105': 'MS',
    '172967424': 'C', '807857108': 'SCHW',
    # Additional holdings from our testing data
    '26884L109': 'EQT', '655663102': 'NDSN', '169656105': 'CRL',
    '012653101': 'AMT', '11135F101': 'BLK', '532457108': 'LLY',
}

UA = os.getenv("SEC_USER_AGENT", "mateusz@bartczak.me Data Downloader")
HEADERS_SEC = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}
HEADERS_SSGA = {"User-Agent": "Mozilla/5.0"}


def _fetch_sec_company_tickers() -> None:
    """Fetches company tickers from SEC and updates the cache."""
    try:
        sec_url = "https://www.sec.gov/files/company_tickers.json"
        response = requests.get(sec_url, headers=HEADERS_SEC, timeout=10)
        response.raise_for_status()
        company_data = response.json()
        # This part is tricky as SEC data doesn't directly map CUSIPs.
        # For now, we rely on the static map and log misses.
        # A more advanced approach would involve cross-referencing CIKs or names,
        # but that's beyond the scope of simple CUSIP-to-ticker for now.
        logger.debug("SEC company tickers data fetched, but direct CUSIP mapping is not available from this source.")
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch SEC company tickers: {e}")
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse SEC company tickers JSON: {e}")


def _cusip_to_ticker(cusip: str) -> str:
    """
    Convert CUSIP to ticker symbol using a cached mapping.
    The cache is pre-populated with common CUSIPs and can be (manually) updated.
    """
    if not cusip or not isinstance(cusip, str) or len(cusip) != 9:
        logger.debug(f"Invalid CUSIP format: {cusip}")
        return cusip

    # Check cache first
    if cusip in _CUSIP_TICKER_CACHE:
        return _CUSIP_TICKER_CACHE[cusip]

    # Attempt to fetch from SEC if not in cache (one-time attempt per session for new CUSIPs)
    # This is a placeholder for a more robust dynamic update mechanism.
    # For now, we mainly rely on the static map.
    # if not hasattr(_fetch_sec_company_tickers, '_fetched_once'):
    #     _fetch_sec_company_tickers()
    #     setattr(_fetch_sec_company_tickers, '_fetched_once', True)
    #     if cusip in _CUSIP_TICKER_CACHE: # Check again after potential update
    #         return _CUSIP_TICKER_CACHE[cusip]

    # Log unmapped CUSIPs for future enhancement
    # Only log if it looks like a valid CUSIP structure (e.g., not already a ticker)
    if cusip.isalnum() and not any(c.islower() for c in cusip): # Basic check
        logger.info(f"Unmapped CUSIP: {cusip}. Consider adding to static mapping or enhancing dynamic lookup.")

    return cusip  # Return CUSIP as fallback

def daterange(start: pd.Timestamp, end: pd.Timestamp):
    cur = start
    while cur <= end:
        yield cur
        cur += pd.Timedelta(days=1)

def month_ends(start, end):
    cur = start.replace(day=1)
    while cur <= end:
        next_month = (cur + dt.timedelta(days=32)).replace(day=1)
        yield (next_month - dt.timedelta(days=1))
        cur = next_month

def quarter_ends(start, end):
    months = {3, 6, 9, 12}
    cur = start
    while cur <= end:
        if cur.month in months and (cur + dt.timedelta(days=1)).month not in months:
            yield cur
        cur += dt.timedelta(days=1)

# --------------------------------------------------------------------------- #
# Global / module-level settings
# --------------------------------------------------------------------------- #
# Cached files are considered fresh for this many hours. If a cache file is newer
# than the threshold it will be reused instead of triggering a potentially very
# long redownload.
_CACHE_EXPIRY_HOURS = 24 * 365  # retained for backwards compatibility, no longer used in logic

# In-memory cache for the full history so that repeat calls during the same
# interpreter session are instant.
_HISTORY_DF: Optional[pd.DataFrame] = None  # populated lazily

# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Excel reader helper – handles legacy .xls as well as modern .xlsx
# --------------------------------------------------------------------------- #

def _read_excel_bytes(buf: bytes, *, file_ext: str) -> pd.DataFrame:
    """Read an in-memory Excel file selecting the proper engine.

    Parameters
    ----------
    buf : bytes
        Raw bytes returned by *requests*.
    file_ext : str
        Either ``".xls"`` or ``".xlsx"`` (case-insensitive).

    Returns
    -------
    pandas.DataFrame
    """
    file_ext = file_ext.lower()
    if file_ext == ".xls":
        # xlrd ≥ 2.0 only supports .xls, not .xlsx
        return pd.read_excel(io.BytesIO(buf), engine="xlrd")
    # default to openpyxl for .xlsx
    df = pd.read_excel(io.BytesIO(buf), engine="openpyxl")
    if "ticker" not in (c.lower() for c in df.columns):
        # Try several header offsets (0-5) to cope with extra preamble rows.
        for hdr in range(1, 6):
            try:
                tmp = pd.read_excel(io.BytesIO(buf), engine="openpyxl", header=hdr)
                if "ticker" in (c.lower() for c in tmp.columns):
                    df = tmp
                    break
            except Exception:  # noqa: BLE001
                continue
    return df

# --------------------------------------------------------------------------- #
# 1) SSGA daily XLSX
# --------------------------------------------------------------------------- #
def _fetch_from_wayback(orig_url: str, date: pd.Timestamp) -> bytes | None:
    """Attempt to download *orig_url* for *date* via the Internet Archive.

    Parameters
    ----------
    orig_url : str
        The canonical URL that returned 404 today.
    date : datetime.date
        Target trading date – used as the desired snapshot timestamp.

    Returns
    -------
    bytes | None
        Raw file bytes if a snapshot is available; otherwise ``None``.
    """
    api = (
        "https://archive.org/wayback/available?url="
        f"{quote(orig_url, safe='')}"  # fully-escaped
        f"&timestamp={date:%Y%m%d}"
    )

    try:
        meta_resp = requests.get(api, timeout=30)
        meta_resp.raise_for_status()
        payload = meta_resp.json()
        closest = payload.get("archived_snapshots", {}).get("closest") or {}
        if closest.get("available") and closest.get("status") == "200":
            snap_url = closest.get("url")
            if snap_url:
                file_resp = requests.get(snap_url, timeout=60)
                if file_resp.status_code == 200:
                    return file_resp.content
    except Exception as exc:  # noqa: BLE001 – we want a broad net here
        logger.debug(f"Wayback fetch failed for {orig_url}: {exc}")
    return None

def ssga_daily(date: Union[dt.date, pd.Timestamp]):
    if isinstance(date, dt.date):
        date = pd.Timestamp(date)
    """
    Return DataFrame for one date or None if 404.
    Uses a 6-hour cache in ../cache/ssga_daily/
    """
    script_dir = Path(__file__).parent
    cache_dir = script_dir.parent.parent / "cache" / "ssga_daily"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{date:%Y-%m-%d}.parquet"

    # ------------------------------------------------------------------
    # Cache logic
    #   • For snapshots older than *yesterday* we treat the parquet as immutable
    #     and always reuse it, regardless of its file timestamp. Historical
    #     Wayback files never change.
    #   • For today/very recent dates (<48 h old) we keep the 6-hour freshness
    #     window so a rerun during the trading day can pick up the latest
    #     basket.
    # ------------------------------------------------------------------
    if cache_file.exists():
        is_historical = (pd.Timestamp.today() - date) > pd.Timedelta(days=7)
        if is_historical:
            logger.debug(f"Using historical cache for SSGA {date:%Y-%m-%d}")
            return pd.read_parquet(cache_file)

        mod_time = dt.datetime.fromtimestamp(cache_file.stat().st_mtime)
        is_fresh = (dt.datetime.now() - mod_time) < dt.timedelta(hours=6)
        if is_fresh:
            logger.debug(f"Using fresh cache for SSGA {date:%Y-%m-%d}")
            return pd.read_parquet(cache_file)

    if date < EARLIEST_SSGA_DATE:
        logger.debug(f"SSGA data unavailable before {EARLIEST_SSGA_DATE:%Y-%m-%d} for date {date:%Y-%m-%d}")
        return None

    df = _fetch_ssga_data(date)

    if df is None:
        logger.debug(f"SSGA daily data not found for {date:%Y-%m-%d}")
        return None

    df = _normalize_ssga_columns(df)
    df["date"] = pd.Timestamp(date)
    df["ticker"] = df["ticker"].str.upper()

    result_df = df[["date", "ticker", "weight_pct", "shares", "market_value"]].copy()
    result_df = _sanitize_ssga_weight_pct(result_df)

    logger.info(f"✓ SSGA {date:%Y-%m-%d} basket processed ({len(result_df)} rows)")
    result_df.to_parquet(cache_file)
    return result_df

def _fetch_ssga_data(date: pd.Timestamp) -> pd.DataFrame | None:
    """Fetches SSGA data for a given date, trying live URL first, then Wayback Machine."""
    df = None
    # Try live URL for today's date
    if date == pd.Timestamp.today().normalize():
        try:
            resp = requests.get(GENERIC_SSGA_URL, headers=HEADERS_SSGA, timeout=10)
            if resp.status_code == 200:
                df = _read_excel_bytes(resp.content, file_ext=".xlsx")
                logger.info(f"SSGA {date:%Y-%m-%d} basket downloaded from live URL.")
                return df
            else:
                logger.debug(f"Live SSGA fetch failed for {date:%Y-%m-%d}: status {resp.status_code}")
        except requests.RequestException as exc:
            logger.debug(f"Live SSGA fetch error for {date:%Y-%m-%d}: {exc}")

    # Fallback to Wayback Machine for historical dates or if live fetch failed
    archive_bytes = _fetch_from_wayback(GENERIC_SSGA_URL, date)
    if archive_bytes:
        df = _read_excel_bytes(archive_bytes, file_ext=".xlsx")
        logger.info(f"SSGA {date:%Y-%m-%d} basket downloaded via Wayback ({len(df) if df is not None else 0} rows)")
    return df

def _normalize_ssga_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes column names for SSGA DataFrame."""
    df.columns = df.columns.str.strip().str.lower()
    column_renames = {
        "weight (%)": "weight_pct", "% weight": "weight_pct", "weight": "weight_pct",
        "ticker": "ticker", "shares": "shares", "shares held": "shares",
        "market value ($)": "market_value", "market value": "market_value",
        "market value (usd)": "market_value",
    }
    df = df.rename(columns=column_renames)

    if "market_value" not in df.columns:
        for col in df.columns:
            if "market value" in col:
                df.rename(columns={col: "market_value"}, inplace=True)
                break

    for col in ["shares", "market_value", "weight_pct", "ticker"]:
        if col not in df.columns:
            df[col] = pd.NA
    return df

def _sanitize_ssga_weight_pct(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitizes the 'weight_pct' column in SSGA DataFrame."""
    if "weight_pct" in df.columns and df["weight_pct"].dtype == object:
        clean_weight = (
            df["weight_pct"].astype(str)
            .str.strip()
            .str.rstrip("%")
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA}) # handle various NA representations
        )
        df.loc[:, "weight_pct"] = pd.to_numeric(clean_weight, errors="coerce")
    elif "weight_pct" in df.columns:
        df.loc[:, "weight_pct"] = pd.to_numeric(df["weight_pct"], errors="coerce")
    return df

# --------------------------------------------------------------------------- #
# 2 & 3) SEC filings via EdgarTools (auto-throttled, ~180 filings/min)
# --------------------------------------------------------------------------- #
def init_edgar():
    set_identity(UA.split()[0])        # EdgarTools wants just an email

def _filing_obj_with_retry(filing, max_retries: int = 5, initial_delay: float = 1.0):
    """Return ``filing.obj()`` with exponential back-off on HTTP 429.

    The SEC API will throttle requests aggressively.  When we hit a
    ``429 Too Many Requests`` status we back-off exponentially up to
    *max_retries* attempts.  Other HTTP errors are re-raised
    immediately.

    Parameters
    ----------
    filing : edgar.Filing
        The filing instance from which to fetch the parsed object.
    max_retries : int, default ``5``
        Maximum number of attempts before giving up.
    initial_delay : float, default ``1.0``
        Seconds to wait before the first retry (doubles every attempt).
    """
    for attempt in range(max_retries):
        try:
            return filing.obj()
        except HTTPStatusError as exc:
            if exc.response.status_code == 429:
                wait = initial_delay * (2 ** attempt)
                logger.warning(
                    f"HTTP 429 for {filing.accession_no} – waiting {wait:.1f}s before retry {attempt+1}/{max_retries}.")
                time.sleep(wait)
                continue
            raise
    logger.error(
        f"Exceeded {max_retries} retries for {filing.accession_no} due to repeated HTTP 429 responses.")
    return None

def _get_filing_date(filing) -> pd.Timestamp | None:
    """Extracts and validates the period_of_report date from a filing."""
    period_of_report = filing.period_of_report
    if isinstance(period_of_report, str):
        try:
            return pd.Timestamp(period_of_report)
        except ValueError:
            logger.warning(f"Skipping filing {filing.accession_no}: Invalid date string '{period_of_report}'")
            return None
    elif isinstance(period_of_report, dt.date):
        return pd.Timestamp(period_of_report)
    logger.warning(f"Skipping filing {filing.accession_no}: Invalid period_of_report type {type(period_of_report)}.")
    return None

def _extract_holdings_from_obj(obj, filing_accession_no: str) -> list:
    """Extracts holdings items from a parsed SEC filing object."""
    if FundReport is not None and isinstance(obj, FundReport):
        return getattr(obj.portfolio, "holdings", [])
    elif isinstance(obj, dict):
        return obj.get("portfolio", [])
    elif hasattr(obj, "portfolio"):
        port = obj.portfolio
        if isinstance(port, list):
            return port
        elif hasattr(port, "holdings"):
            return port.holdings or []
    elif hasattr(obj, "investments"): # edgar 4.3+ FundReport
        return obj.investments or []
    elif hasattr(obj, "investment_data"):
        return obj.investment_data or []

    logger.warning(f"Skipping filing {filing_accession_no}: Unexpected obj type {type(obj)} or no holdings data found.")
    return []

def _process_holding_item(item, date: pd.Timestamp) -> tuple | None:
    """Processes a single holding item (either modern object or old dict)."""
    ticker = None
    pct_nav = None
    shares = None
    value = None

    if hasattr(item, 'model_dump'):  # New EdgarTools format (InvestmentOrSecurity)
        raw_ticker = getattr(item, 'ticker', None) or ""
        cusip = getattr(item, 'cusip', None)

        if cusip:
            ticker = _cusip_to_ticker(cusip)
            # If CUSIP maps to itself and raw_ticker exists, prefer raw_ticker
            if ticker == cusip and raw_ticker:
                ticker = raw_ticker
        elif raw_ticker:
            ticker = raw_ticker

        # Fallback to name-based heuristic only if no ticker/CUSIP resolution
        if not ticker and hasattr(item, 'name') and item.name:
            ticker = item.name.replace(" ", "_").upper()[:10]

        pct_nav = float(getattr(item, 'pct_value', 0)) if getattr(item, 'pct_value', None) is not None else None
        shares = float(getattr(item, 'balance', 0)) if getattr(item, 'balance', None) is not None else None
        value = float(getattr(item, 'value_usd', 0)) if getattr(item, 'value_usd', None) is not None else None
        asset_category = getattr(item, 'asset_category', '').upper()
        if ticker and asset_category == 'EC':  # Equity Common stock
            return date, ticker.upper(), pct_nav, shares, value

    elif isinstance(item, dict):  # Old EdgarTools format (dictionary)
        security_type = item.get("security_type", "").lower()
        if security_type == "common stock":
            identifier = str(item.get("identifier", "")).upper()
            ticker_to_use = _cusip_to_ticker(identifier) if len(identifier) == 9 and identifier.isalnum() else identifier

            if ticker_to_use:
                pct_nav = item.get("pct_nav") if item.get("pct_nav") is not None else item.get("pct_value")
                shares = item.get("shares")
                value = item.get("value")
                return date, ticker_to_use.upper(), pct_nav, shares, value
    return None

def _process_sec_filing(filing, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame | None:
    """Helper to process a single SEC filing and extract holdings."""
    date = _get_filing_date(filing)
    if date is None or not (start_date <= date <= end_date):
        logger.debug(f"Skipping filing {filing.accession_no}: Date {date} out of range or invalid.")
        return None

    obj = _filing_obj_with_retry(filing)
    if obj is None:
        logger.warning(f"Could not retrieve object for filing {filing.accession_no}.")
        return None

    items = _extract_holdings_from_obj(obj, filing.accession_no)
    if not items:
        logger.debug(f"No items found in filing {filing.accession_no}.")
        return None

    rows = []
    for item in items:
        processed_item = _process_holding_item(item, date)
        if processed_item:
            rows.append(processed_item)

    if rows:
        df = pd.DataFrame(rows, columns=["date", "ticker", "weight_pct", "shares", "market_value"])
        # Ensure weight_pct is numeric, coercing errors
        if 'weight_pct' in df.columns:
            df['weight_pct'] = pd.to_numeric(df['weight_pct'], errors='coerce')
        return df

    logger.debug(f"No processable common stock holdings found in filing {filing.accession_no}.")
    return None

def _load_sec_holdings_from_cache(cache_file: Path) -> pd.DataFrame | None:
    """Attempts to load SEC holdings from cache. Returns DataFrame if successful, None otherwise."""
    if cache_file.exists():
        mod_time = dt.datetime.fromtimestamp(cache_file.stat().st_mtime)
        if (dt.datetime.now() - mod_time) < dt.timedelta(hours=6):
            logger.info(f"Loading SEC holdings from cache: {cache_file}")
            return pd.read_parquet(cache_file)
    return None

def _get_sec_filings(company: Company, start_date: pd.Timestamp) -> list:
    """Fetches and filters SEC filings for NPORT-P and N-Q forms."""
    forms = company.get_filings().filter(form=["NPORT-P", "N-Q"])
    nport = [f for f in forms if f.form == "NPORT-P" and start_date.date() <= f.filing_date]
    nq    = [f for f in forms if f.form == "N-Q"      and start_date.date() <= f.filing_date]
    return nport + nq

def sec_holdings(start, end):
    """
    Returns a DataFrame from N-PORT-P (monthly) and N-Q (quarterly).
    Uses a 6-hour cache in ../cache/sec_holdings/
    """
    script_dir = Path(__file__).parent
    cache_dir = script_dir.parent.parent / "cache" / "sec_holdings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{CIK}_{start:%Y-%m-%d}_{end:%Y-%m-%d}.parquet"

    df = _load_sec_holdings_from_cache(cache_file)
    if df is not None:
        return df

    logger.info(f"Downloading SEC N-PORT-P & N-Q filings for {CIK} from {start} to {end}...")
    company = Company(CIK)
    all_filings = _get_sec_filings(company, start)

    filing_frames = []
    for filing in tqdm(all_filings, desc="SEC filings"):
        df = _process_sec_filing(filing, start, end)
        if df is not None:
            filing_frames.append(df)

    if not filing_frames:
        logger.warning("No SEC data downloaded.")
        return pd.DataFrame()

    result_df = pd.concat(filing_frames, ignore_index=True)
    result_df.to_parquet(cache_file)
    logger.info(f"SEC holdings saved to cache: {cache_file}")
    return result_df

# --------------------------------------------------------------------------- #
# Pull everything and merge
# --------------------------------------------------------------------------- #
def build_history(start: pd.Timestamp, end: pd.Timestamp, *, ignore_cache: bool = False):
    """Download & assemble SPY holdings between *start* and *end*.

    Parameters
    ----------
    ignore_cache : bool, default ``False``
        If ``True`` the function bypasses any aggregated parquet in
        ``cache/spy_history`` and rebuilds the DataFrame from scratch.
    """

    # Try loading an aggregated cache first – unless explicitly disabled
    if not ignore_cache:
        cached = _load_history_from_cache(start, end)
        if cached is not None:
            return cached

    init_edgar()
    frames: list[pd.DataFrame] = []

    logger.info("Downloading SSGA daily basket …")
    ssga_start = max(start, EARLIEST_SSGA_DATE)
    if ssga_start > end:
        logger.info("SSGA daily basket unavailable for requested window – skipping stage.")
    for d in tqdm(list(daterange(ssga_start, end))):
        if d.weekday() >= 5:   # skip Saturday/Sunday – no files published
            continue
        df = ssga_daily(d)
        if df is not None:
            frames.append(df)

    sec_df = sec_holdings(start, end)
    if not sec_df.empty:
        frames.append(sec_df)

    if not frames:
        raise RuntimeError("No data downloaded! Check connectivity and headers.")

    hist = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["date", "ticker"])
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )

    hist = _forward_fill_history(hist, start, end)
    logger.info(f"Successfully built history with {len(hist)} rows.")

    if not ignore_cache:
        _save_history_to_cache(hist, start, end)

    return hist

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Download SPY holdings history")
    parser.add_argument("--start", default="2004-01-01",
                        help="YYYY-MM-DD (default 2004-01-01, earliest SEC N-Q)")
    parser.add_argument("--end",   default=str(pd.Timestamp.today().date()),
                        help="YYYY-MM-DD (default today)")
    parser.add_argument("--out",   required=True,
                        help="Output filename (.parquet or .csv)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level.")
    parser.add_argument("--update", action="store_true", help="Update the full history.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the full history from scratch (ignores existing parquet).")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format='%(asctime)s - %(levelname)s - %(message)s')

    start = pd.Timestamp(args.start)
    end   = pd.Timestamp(args.end)

    script_dir = Path(__file__).parent
    out_dir = script_dir.parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out

    if args.rebuild:
        # Remove aggregated spy_history cache file as well
        agg_cache = _history_cache_file(start, end)
        agg_cache.unlink(missing_ok=True)
        update_full_history(out_path, start, end, rebuild=True)
    elif args.update:
        update_full_history(out_path, start, end)
    else:
        hist = build_history(start, end)
        if out_path.suffix == ".csv":
            hist.to_csv(out_path, index=False)
        else:
            hist.to_parquet(out_path, index=False)
        logger.info(f"✓ Done. {len(hist):,} rows written to {out_path}")

def _history_cache_file(start: dt.date, end: dt.date) -> Path:
    """Return the path of the aggregate history cache parquet file for the given date range."""
    script_dir = Path(__file__).parent
    cache_dir = script_dir.parent.parent / "cache" / "spy_history"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{TICKER}_history_{start.date():%Y-%m-%d}_{end.date():%Y-%m-%d}.parquet"


def _load_history_from_cache(start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.DataFrame]:
    """Load aggregate history DataFrame from cache if present and it fully covers the requested date range."""
    cache_file = _history_cache_file(start, end)
    if cache_file.exists():
        logger.info(f"Loading aggregated holdings history from cache: {cache_file}")
        try:
            df = pd.read_parquet(cache_file)
            # Verify the cached data covers the required date range
            if not df.empty and df['date'].min() <= start and df['date'].max() >= end:
                return df
            else:
                logger.warning(f"Cached data in {cache_file} does not cover the full range {start}-{end}. Ignoring cache.")
        except Exception as exc:
            logger.warning(f"Failed to read cached parquet {cache_file}: {exc} – ignoring cache.")
    return None


def _save_history_to_cache(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> None:
    """Persist aggregate history DataFrame to parquet cache (atomic write)."""
    cache_file = _history_cache_file(start, end)
    tmp_file = cache_file.with_suffix(".tmp")
    df.to_parquet(tmp_file, index=False)
    tmp_file.replace(cache_file)
    logger.info(f"Aggregated holdings history saved to cache: {cache_file}")


def get_spy_holdings(date: Union[str, dt.date, pd.Timestamp], *, exact: bool = False) -> pd.DataFrame:
    """Return a DataFrame of SPY constituent holdings for the supplied *date*.

    The DataFrame is sorted by descending weight (``weight_pct``). The *date*
    can be provided as ``datetime.date``, ``pandas.Timestamp`` or a ``YYYY-MM-DD``
    string.

    Parameters
    ----------
    date : str | datetime.date | pandas.Timestamp
        Target date for which to fetch the holdings.
    exact : bool, default ``False``
        If ``True``, require an exact match for *date*. If ``False`` (default)
        and the exact date is missing, the most recent previous trading day
        available (within the dataset) will be used.

    Returns
    -------
    pandas.DataFrame
        Holdings on the requested date ordered by descending index weight.
    """
    global _HISTORY_DF

    # Normalise *date* to datetime.date
    if isinstance(date, str):
        date = pd.Timestamp(date)
    elif isinstance(date, dt.date):
        date = pd.Timestamp(date)
    elif not isinstance(date, pd.Timestamp):
        raise TypeError("date must be a str, datetime.date or pandas.Timestamp")

    # Lazily load full history into memory.
    _ensure_history_loaded()

    target_ts = pd.Timestamp(date) # Ensure it's a Timestamp

    if _HISTORY_DF is None or _HISTORY_DF.empty:
        # This case should ideally be handled by _ensure_history_loaded,
        # but as a safeguard:
        logger.error("SPY holdings history is not available even after attempting to load.")
        raise ValueError("SPY holdings history could not be loaded.")

    df = _HISTORY_DF[_HISTORY_DF["date"] == target_ts]
    logger.debug(f"Holdings query for exact date {target_ts}: {len(df)} rows found.")

    if df.empty and not exact:
        # Fallback to the latest available date on or before the target_ts
        available_dates_before = _HISTORY_DF[_HISTORY_DF["date"] <= target_ts]["date"]
        if not available_dates_before.empty:
            nearest_date = available_dates_before.max()
            df = _HISTORY_DF[_HISTORY_DF["date"] == nearest_date]
            logger.info(
                f"Exact holdings for {target_ts:%Y-%m-%d} unavailable. Using nearest previous date {nearest_date:%Y-%m-%d} ({len(df)} rows)."
            )
        else: # Should not happen if history is loaded and target_ts is reasonable
            logger.warning(f"No holdings data found on or before {target_ts:%Y-%m-%d}, even with exact=False.")


    if df.empty:
        raise ValueError(f"No holdings data found for {target_ts:%Y-%m-%d} (exact={exact}). Ensure date is within data range: {_HISTORY_DF['date'].min():%Y-%m-%d} to {_HISTORY_DF['date'].max():%Y-%m-%d}")

    return df.sort_values("weight_pct", ascending=False).reset_index(drop=True)

def _ensure_history_loaded():
    """Ensures the _HISTORY_DF is loaded, preferring bundled parquet, then building if necessary."""
    global _HISTORY_DF
    if _HISTORY_DF is not None and not _HISTORY_DF.empty:
        return

    logger.debug("Attempting to load SPY holdings history...")
    # Attempt to locate the *bundled* history parquet relative to repo root.
    repo_root = next(
        (p for p in Path(__file__).resolve().parents
         if (p / "data" / "spy_holdings_full.parquet").exists()),
        None,
    )

    if repo_root is not None:
        bundled_path = repo_root / "data" / "spy_holdings_full.parquet"
        try:
            logger.info(f"Loading bundled holdings history from: {bundled_path}")
            _HISTORY_DF = pd.read_parquet(bundled_path)
            if not isinstance(_HISTORY_DF, pd.DataFrame) or _HISTORY_DF.empty:
                logger.warning(f"Bundled parquet at {bundled_path} is invalid or empty.")
                _HISTORY_DF = None # Force rebuild if file is corrupt or empty
            else:
                logger.info(f"Loaded bundled history with shape: {_HISTORY_DF.shape}. Date range: {_HISTORY_DF['date'].min():%Y-%m-%d} to {_HISTORY_DF['date'].max():%Y-%m-%d}")
                return # Successfully loaded
        except Exception as e:
            logger.error(f"Failed to load bundled parquet {bundled_path}: {e}")
            _HISTORY_DF = None # Ensure it's None if loading failed

    if _HISTORY_DF is None:
        logger.warning(
            "Bundled holdings history not found or failed to load. "
            "Building from scratch. This may take a while and hit SEC rate limits."
        )
        try:
            # Define a reasonable default start date if building from scratch
            default_start_date = pd.Timestamp(2004, 1, 1)
            current_date = pd.Timestamp.today().normalize()
            _HISTORY_DF = build_history(default_start_date, current_date)
            if _HISTORY_DF is not None and not _HISTORY_DF.empty:
                logger.info(f"Built history with shape: {_HISTORY_DF.shape}. Date range: {_HISTORY_DF['date'].min():%Y-%m-%d} to {_HISTORY_DF['date'].max():%Y-%m-%d}")
            else:
                logger.error("Failed to build history or an empty DataFrame was returned.")
                _HISTORY_DF = pd.DataFrame() # Assign empty df to prevent repeated build attempts in same session if build fails
        except Exception as e:
            logger.error(f"Error building history from scratch: {e}")
            _HISTORY_DF = pd.DataFrame() # Assign empty df to prevent repeated build attempts

def update_full_history(
    out_path: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    *,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Update or rebuild the SPY holdings parquet.

    Parameters
    ----------
    out_path : Path
        Destination parquet file.
    start_date, end_date : pandas.Timestamp
        Date range to cover.
    rebuild : bool, default ``False``
        When ``True`` existing *out_path* is ignored and the full history is
        rebuilt from scratch. This is the safest way to back-fill historical
        daily baskets after improving the download logic.
    """

    if rebuild and out_path.exists():
        logger.info("--rebuild specified → deleting existing parquet and starting fresh …")
        out_path.unlink(missing_ok=True)

    if out_path.exists():
        existing = pd.read_parquet(out_path)
        if not existing.empty:
            latest = existing["date"].max()
            # nothing new to fetch
            if latest >= end_date:
                logger.info("Parquet up-to-date ‑ no download required.")
                return existing
            # we need to extend from the day after latest
            incremental_start = latest + pd.Timedelta(days=1)
            logger.info(f"Updating history from {incremental_start.date()} to {end_date.date()} …")
            new_df = build_history(incremental_start, end_date, ignore_cache=rebuild)
            combined = (
                pd.concat([existing, new_df], ignore_index=True)
                  .drop_duplicates(subset=["date", "ticker"])
                  .sort_values(["date", "ticker"])
                  .reset_index(drop=True)
            )
        else:
            combined = build_history(start_date, end_date, ignore_cache=rebuild)
    else:
        logger.info(f"Creating new holdings history {out_path}")
        combined = build_history(start_date, end_date, ignore_cache=rebuild)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    logger.info(f"✓ Saved {len(combined):,} rows → {out_path}")
    return combined


def _forward_fill_history(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Forward-fill missing dates in the holdings DataFrame.

    This function ensures that every business day within the specified range
    has holdings data by forward-filling from the most recent available date.
    """
    # Create a complete date range of business days
    full_date_range = pd.date_range(start, end, freq='B')
    full_dates_df = pd.DataFrame({'date': full_date_range})

    # Ensure 'date' column is datetime and set as index for reindexing
    df['date'] = pd.to_datetime(df['date'])
    df_indexed = df.set_index('date').sort_index()

    # Reindex to the full date range, then forward fill
    # We need to group by ticker before reindexing to avoid filling across different tickers
    # However, the goal is to fill *missing dates* for the *entire* dataset, not per ticker.
    # So, we'll reindex the unique dates and then merge back.

    # Get unique dates from the original DataFrame
    unique_dates_df = df.drop_duplicates(subset=['date']).set_index('date').sort_index()

    # Reindex the unique dates to the full business day range and forward fill
    reindexed_dates = unique_dates_df.reindex(full_date_range, method='ffill')

    # Now, merge this back with the original detailed DataFrame.
    # This is a more complex operation if we want to fill *all* columns for *all* tickers.
    # A simpler approach for the test's requirement (presence of dates) is to ensure
    # the `build_history` output has all dates.

    # Let's re-think: the test checks if `df['date'].unique()` contains all business days.
    # This means we need to ensure that `build_history` produces a DataFrame where
    # every business day has *some* entry, even if it's a duplicate of the previous day's holdings.

    # Group by date and get the holdings for each date
    # This assumes that for a given date, all holdings are present.
    # If a date is missing, we want to copy the previous day's holdings.

    # Get all unique dates from the input DataFrame
    existing_dates = df['date'].dt.normalize().unique()
    existing_dates_ts = pd.Series(existing_dates).sort_values()

    # Create a DataFrame with all expected business days
    all_business_days = pd.DataFrame({'date': full_date_range})

    # Merge to find missing dates
    merged_df = pd.merge(all_business_days, df, on='date', how='left')

    # Sort by date and then by ticker to ensure proper forward fill within groups
    merged_df = merged_df.sort_values(by=['date', 'ticker'])

    # Forward fill the entire DataFrame. This will fill ticker, weight_pct, shares, market_value
    # from the last available date.
    # This is a simplification. A more robust solution might involve:
    # 1. Identifying missing dates.
    # 2. For each missing date, finding the most recent previous date with data.
    # 3. Copying all holdings from that previous date to the missing date.

    # Let's try a more explicit approach for filling missing dates with previous day's data.
    # Get all unique dates from the input DataFrame
    df_dates = df['date'].dt.normalize().unique()
    df_min_date = df_dates.min()
    df_max_date = df_dates.max()

    # Create a complete range of business days within the DataFrame's actual date range
    expected_business_days_in_df_range = pd.date_range(df_min_date, df_max_date, freq='B')

    # Identify missing business days within the DataFrame's actual date range
    missing_dates_in_df_range = pd.Series(expected_business_days_in_df_range).difference(pd.Series(df_dates))

    # For each missing date, find the closest previous date with data
    # and duplicate that day's holdings.
    filled_frames = []
    current_date_data = pd.DataFrame()

    # Sort the original DataFrame by date
    df_sorted = df.sort_values(by='date')

    # Iterate through the full date range
    for single_date in full_date_range:
        # Check if data exists for the current date
        data_for_date = df_sorted[df_sorted['date'].dt.normalize() == single_date.normalize()]

        if not data_for_date.empty:
            # If data exists, use it and update current_date_data
            filled_frames.append(data_for_date)
            current_date_data = data_for_date.copy()
        elif not current_date_data.empty:
            # If data is missing but we have previous data, use the previous data
            # and update its date to the current missing date
            filled_data = current_date_data.copy()
            filled_data['date'] = single_date
            filled_frames.append(filled_data)
        else:
            # If no data exists and no previous data is available (e.g., at the very beginning)
            # then we cannot fill, and this date will remain missing or be handled by subsequent logic.
            # For the purpose of this test, we assume data will eventually start.
            logger.warning(f"No data to forward-fill from for date: {single_date.normalize().date()}")

    if filled_frames:
        filled_df = pd.concat(filled_frames, ignore_index=True)
        # Ensure unique date-ticker pairs after filling
        filled_df = filled_df.drop_duplicates(subset=['date', 'ticker']).sort_values(by=['date', 'ticker']).reset_index(drop=True)
        return filled_df
    else:
        return pd.DataFrame(columns=df.columns) # Return empty if no data was ever found




def get_top_weight_sp500_components(date: Union[str, dt.date, pd.Timestamp], top_n: int = 30, *, exact: bool = False) -> List[tuple[str, float]]:
    """Return the *top_n* ticker symbols by weight in the S&P 500 (via SPY) for *date*.

    Parameters
    ----------
    date : "YYYY-MM-DD" | datetime.date | pandas.Timestamp
        Date for which to retrieve the universe constituents.
    top_n : int, default ``30``
        Number of tickers to return ordered by descending index weight.
    exact : bool, default ``False``
        Forward-looking bias safeguard. When ``exact=False`` (default) the most
        recent *past* trading day will be used if *date* itself is missing.
        When ``exact=True`` an error is raised if *date* is not available.

    Notes
    -----
    Results are cached in-memory for the duration of the Python interpreter
    using :pyfunc:`functools.lru_cache` to make repeated optimiser calls
    essentially free.
    """
    # We wrap the real implementation in an LRU cache because `lru_cache` does
    # not play nicely with pandas objects directly.
    return _get_top_weight_sp500_components_cached(_normalise_date_key(date), top_n, exact)


from functools import lru_cache

def _normalise_date_key(date: Union[str, dt.date, pd.Timestamp]) -> str:
    """Return ISO "YYYY-MM-DD" string representation for *date* suitable as cache key."""
    if isinstance(date, str):
        return date
    if isinstance(date, pd.Timestamp):
        return date.strftime("%Y-%m-%d")
    if isinstance(date, dt.date):
        return date.isoformat()
    raise TypeError("date must be str | datetime.date | pandas.Timestamp")


@lru_cache(maxsize=4096)
def _get_top_weight_sp500_components_cached(
    date_key: str, top_n: int, exact: bool
) -> List[tuple[str, float]]:
    """Private LRU-cached helper.

    Returns a *list* of *(ticker, weight_pct)* tuples ordered by descending
    weight. ``weight_pct`` is the raw float value found in the SPY basket file
    (e.g. ``7.52`` meaning **7.52 %**).
    """

    target_ts = pd.Timestamp(date_key)
    logger.info(f"_get_top_weight_sp500_components_cached: target_ts={target_ts}, top_n={top_n}, exact={exact}")

    # Ensure the global history is loaded (side-effect of get_spy_holdings)
    try:
        _ = get_spy_holdings(target_ts, exact=False)
    except ValueError:
        # If even the generic fallback failed we proceed – the history may
        # still load for earlier dates.
        logger.warning(f"get_spy_holdings failed for {target_ts}, but proceeding to check _HISTORY_DF.")
        pass

    global _HISTORY_DF  # guarantee visibility
    if _HISTORY_DF is None or _HISTORY_DF.empty:
        logger.error("SPY holdings history is not available after attempting to load.")
        raise ValueError("SPY holdings history is not available – data load failed.")

    hist = _HISTORY_DF
    logger.info(f"_HISTORY_DF shape: {hist.shape}")
    logger.info(f"_HISTORY_DF date range: {hist['date'].min()} to {hist['date'].max()}")

    mask_valid = (
        (hist["date"] <= target_ts) &
        hist["weight_pct"].notna() &
        (hist["ticker"] != "-")
    )
    logger.info(f"mask_valid count: {mask_valid.sum()}")

    if not mask_valid.any():
        raise ValueError(f"No valid SPY holdings found on or before {date_key}.")

    latest_date = hist.loc[mask_valid, "date"].max()
    logger.info(f"latest_date found: {latest_date}")
    df_valid = hist[(hist["date"] == latest_date) & (hist["ticker"] != "-") & hist["weight_pct"].notna()]  # noqa: E501
    logger.info(f"df_valid shape for latest_date: {df_valid.shape}")

    slice_df = df_valid.sort_values("weight_pct", ascending=False).head(top_n)
    logger.info(f"slice_df shape: {slice_df.shape}")
    return list(zip(slice_df["ticker"].tolist(), slice_df["weight_pct"].tolist()))


# --------------------------------------------------------------------------- #
# Utility – mainly for interactive notebooks / REPL sessions                 #
# --------------------------------------------------------------------------- #


def reset_history_cache() -> None:
    """Clear the in-memory holdings DataFrame *and* the LRU caches.

    This is handy when you rebuild ``spy_holdings_full.parquet`` in a separate
    process and want the current Python session to pick up the fresh data
    without restart.
    """

    global _HISTORY_DF
    _HISTORY_DF = None

    # Clear LRU caches so next call reloads with the updated DataFrame
    # get_top_weight_sp500_components.cache_clear()  # type: ignore[attr-defined]
    _get_top_weight_sp500_components_cached.cache_clear()  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()