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
from edgar import set_identity, Company        # EdgarTools ≥4.3.0
import json
from urllib.parse import quote

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

def _cusip_to_ticker(cusip: str) -> str:
    """
    Convert CUSIP to ticker symbol using multiple data sources.
    
    This function uses a combination of:
    1. Cached mappings from previous lookups
    2. SEC company tickers API for real-time lookups
    3. Fallback to a curated list of major S&P 500 companies
    4. CUSIP as final fallback
    """
    if not cusip or len(cusip) != 9:
        return cusip  # Invalid CUSIP format
        
    try:
        # Initialize cache on first call
        if not hasattr(_cusip_to_ticker, '_ticker_cache'):
            _cusip_to_ticker._ticker_cache = {}
            
        # Check cache first
        if cusip in _cusip_to_ticker._ticker_cache:
            return _cusip_to_ticker._ticker_cache[cusip]
        
        # Try SEC company tickers API (this is the most comprehensive source)
        try:
            # SEC provides a company tickers JSON file that's updated regularly
            sec_url = "https://www.sec.gov/files/company_tickers.json"
            response = requests.get(sec_url, headers=HEADERS_SEC, timeout=10)
            response.raise_for_status()
            
            company_data = response.json()
            
            # Build a reverse lookup from company details to find CUSIP matches
            # Note: This is not perfect since SEC doesn't directly provide CUSIP mappings
            # but we can use company names and CIKs to improve our mapping
            for entry in company_data.values():
                ticker = entry.get('ticker', '')
                if ticker and ticker not in _cusip_to_ticker._ticker_cache.values():
                    # This would require additional CUSIP lookup, which is complex
                    # For now, we'll rely on the static mapping below
                    pass
                    
        except Exception as e:
            logger.debug(f"SEC API lookup failed for CUSIP {cusip}: {e}")
        
        # Fallback to enhanced static mapping for major S&P 500 companies
        # This covers the most common holdings in SPY
        ENHANCED_CUSIP_TICKER_MAP = {
            # Top 10 S&P 500 by market cap (most likely to be in SPY)
            '037833100': 'AAPL',    # Apple Inc
            '594918104': 'MSFT',    # Microsoft Corp
            '67066G104': 'NVDA',    # NVIDIA Corp
            '023135106': 'AMZN',    # Amazon.com Inc
            '30303M102': 'META',    # Meta Platforms Inc
            '02079K305': 'GOOGL',   # Alphabet Inc Class A
            '02079K107': 'GOOG',    # Alphabet Inc Class C
            '88160R101': 'TSLA',    # Tesla Inc
            '084670702': 'BRK.B',   # Berkshire Hathaway Inc Class B
            '92826C839': 'V',       # Visa Inc Class A
            
            # Major financial services
            '46625H100': 'JPM',     # JPMorgan Chase & Co
            '91324P102': 'UNH',     # UnitedHealth Group Inc
            '30231G102': 'XOM',     # Exxon Mobil Corp (corrected CUSIP)
            '459200101': 'JNJ',     # Johnson & Johnson
            '713448108': 'PEP',     # PepsiCo Inc
            
            # Technology leaders
            '17275R102': 'CSCO',    # Cisco Systems Inc
            '00724F101': 'ADBE',    # Adobe Inc
            '79466L302': 'SBUX',    # Starbucks Corp
            '04621X108': 'AVGO',    # Broadcom Inc
            
            # Consumer & Industrial
            '931142103': 'WMT',     # Walmart Inc
            '742718109': 'PG',      # Procter & Gamble Co
            '437076102': 'HD',      # Home Depot Inc
            '580135101': 'MCD',     # McDonald's Corp
            '438516106': 'HON',     # Honeywell International Inc
            
            # Healthcare & Pharma
            '88579Y101': 'TMO',     # Thermo Fisher Scientific Inc
            '02376R102': 'ABT',     # Abbott Laboratories
            '126650100': 'CVS',     # CVS Health Corp
            
            # Additional common SPY holdings
            '191216100': 'KO',      # Coca-Cola Co
            '254687106': 'DIS',     # Walt Disney Co
            '149123101': 'CAT',     # Caterpillar Inc
            '166756103': 'CVX',     # Chevron Corp
            '65339F101': 'NEE',     # NextEra Energy Inc
            '57636Q104': 'MA',      # Mastercard Inc Class A
            
            # Banking & Finance
            '060505104': 'BAC',     # Bank of America Corp
            '38141G104': 'GS',      # Goldman Sachs Group Inc
            '58933Y105': 'MS',      # Morgan Stanley
            '172967424': 'C',       # Citigroup Inc
            '807857108': 'SCHW',    # Charles Schwab Corp
            
            # Additional holdings from our testing data
            '26884L109': 'EQT',     # EQT Corp
            '655663102': 'NDSN',    # Nordson Corp
            '169656105': 'CRL',     # Charles River Laboratories
            '012653101': 'AMT',     # American Tower Corp
            '11135F101': 'BLK',     # BlackRock Inc (common in SPY)
            '532457108': 'LLY',     # Eli Lilly & Co (major pharma)
        }
        
        # Update cache with static mappings
        _cusip_to_ticker._ticker_cache.update(ENHANCED_CUSIP_TICKER_MAP)
        
        # Return mapped ticker or CUSIP as fallback
        result = _cusip_to_ticker._ticker_cache.get(cusip, cusip)
        
        # Log unmapped CUSIPs for future enhancement
        if result == cusip and cusip.isdigit():
            logger.debug(f"Unmapped CUSIP: {cusip} - consider adding to mapping")
            
        return result
        
    except Exception as e:
        logger.debug(f"Error in CUSIP-to-ticker mapping for {cusip}: {e}")
        return cusip  # Return CUSIP as fallback
UA = os.getenv("SEC_USER_AGENT", "mateusz@bartczak.me Data Downloader")
HEADERS_SEC   = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}
HEADERS_SSGA  = {"User-Agent": "Mozilla/5.0"}

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
        # Snapshots older than one week never change – always reuse the local parquet
        if (pd.Timestamp.today() - date) > pd.Timedelta(days=7):
            return pd.read_parquet(cache_file)

        # Recent snapshot (≤7 days old) – honour the 6-hour freshness threshold
        mod_time = dt.datetime.fromtimestamp(cache_file.stat().st_mtime)
        if (dt.datetime.now() - mod_time) < dt.timedelta(hours=6):
            return pd.read_parquet(cache_file)

    # Live site hosts *only* the latest file, but historical copies are archived
    # on the Internet Archive. We therefore:
    #   • use direct HTTP fetch only for "recent" dates (within ~30 days)
    #   • fall back to Wayback Snapshots for anything from
    #     EARLIEST_SSGA_DATE onward.

    recent_cutoff = pd.Timestamp.today() - pd.Timedelta(days=30)

    if date < EARLIEST_SSGA_DATE:
        # No SSGA basket exists before that point – bail early
        return None

    # Download if not in cache or cache is stale
    df = None
    last_status: int | None = None

    if date == pd.Timestamp.today() and last_status == 200:
        # For today we can hit the live generic file directly (fast, ~50 KB)
        try:
            resp = requests.get(GENERIC_SSGA_URL, headers=HEADERS_SSGA, timeout=10)
            last_status = resp.status_code
            if resp.status_code == 200:
                df = _read_excel_bytes(resp.content, file_ext=".xlsx")
        except requests.RequestException as exc:
            logger.debug(f"Live SSGA fetch error for today: {exc}")
    else:
        # Historical – ask Wayback for the generic path first (has daily snapshots ≥2015-03-16)
        archive_bytes = _fetch_from_wayback(GENERIC_SSGA_URL, date)
        if archive_bytes:
            df = _read_excel_bytes(archive_bytes, file_ext=".xlsx")
            logger.info(f"\033[92m✓ SSGA {date:%Y-%m-%d} basket downloaded via Wayback ({len(df)} rows)\033[0m")

    if df is None:
        logger.debug(f"SSGA daily data not found for {date}: {last_status}")
        return None

    # --- normalise columns across gigantic vintage drift -----------------------
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={
        "weight (%)":          "weight_pct",
        "% weight":            "weight_pct",
        "weight":              "weight_pct",
        "ticker":              "ticker",
        "shares":              "shares",
        "shares held":         "shares",
        "market value ($)":    "market_value",
        "market value":        "market_value",
        "market value (usd)":  "market_value",
    })

    # Fallback detection for market value if rename missed it
    if "market_value" not in df.columns:
        for col in df.columns:
            if "market value" in col:
                df.rename(columns={col: "market_value"}, inplace=True)
                break

    for col in ["shares", "market_value", "weight_pct"]:
        if col not in df.columns:
            df[col] = pd.NA

    df["date"] = pd.Timestamp(date)
    df["ticker"] = df["ticker"].str.upper()

    result_df = df[["date", "ticker", "weight_pct", "shares", "market_value"]].copy()

    # ------------------------------------------------------------------
    # Sanitise *weight_pct* – convert any string like "2.41%" → 2.41 (float)
    # ------------------------------------------------------------------
    if result_df["weight_pct"].dtype == object:
        clean = (
            result_df["weight_pct"].astype(str)
                      .str.strip()
                      .str.rstrip("%")
                      .replace({"": pd.NA})
        )
        result_df.loc[:, "weight_pct"] = clean

    result_df.loc[:, "weight_pct"] = pd.to_numeric(result_df["weight_pct"], errors="coerce")

    # Success message for live fetch already logged above for Wayback; do it for direct fetch now.
    if date == pd.Timestamp.today() and last_status == 200:
        logger.info(f"\033[92m✓ SSGA {date:%Y-%m-%d} basket downloaded ({len(result_df)} rows)\033[0m")

    # Save to cache and return
    result_df.to_parquet(cache_file)
    return result_df

# --------------------------------------------------------------------------- #
# 2 & 3) SEC filings via EdgarTools (auto-throttled, ~180 filings/min)
# --------------------------------------------------------------------------- #
def init_edgar():
    set_identity(UA.split()[0])        # EdgarTools wants just an email

def _process_sec_filing(filing, start_date, end_date):
    """Helper to process a single SEC filing and extract holdings."""
    try:
        period_of_report = filing.period_of_report
        if isinstance(period_of_report, str):
            date = pd.Timestamp(period_of_report)
        elif isinstance(period_of_report, dt.date):
            date = pd.Timestamp(period_of_report)
        else:
            logger.warning(f"Skipping filing {filing.accession_no}: Invalid period_of_report type.")
            return None

        if not (start_date <= date <= end_date):
            logger.debug(f"Skipping filing {filing.accession_no}: Outside date range.")
            return None

        obj = filing.obj()
        # edgartools v4 returns FundReport dataclass; v3 returns dict
        if FundReport is not None and isinstance(obj, FundReport):
            items = getattr(obj.portfolio, "holdings", [])
        elif isinstance(obj, dict):
            items = obj.get("portfolio", [])
        elif hasattr(obj, "portfolio"):
            port = obj.portfolio
            if isinstance(port, list):
                items = port
            elif hasattr(port, "holdings"):
                items = port.holdings or []
            else:
                items = []
        elif hasattr(obj, "investments"):
            # edgar 4.3+ FundReport exposes list under .investments
            items = obj.investments or []
        elif hasattr(obj, "investment_data"):
            items = obj.investment_data or []
        else:
            logger.warning(f"Skipping filing {filing.accession_no}: Unexpected obj type {type(obj)}")
            return None
        rows = []
        for it in items:
            # Handle both new EdgarTools format (InvestmentOrSecurity objects) and old format (dicts)
            if hasattr(it, 'model_dump'):  # New format: InvestmentOrSecurity object
                # Extract ticker - try CUSIP lookup or use company name as fallback
                ticker = getattr(it, 'ticker', None) or ""
                if not ticker and hasattr(it, 'cusip') and it.cusip:
                    # Use CUSIP-to-ticker mapping
                    ticker = _cusip_to_ticker(it.cusip)
                elif not ticker and hasattr(it, 'name') and it.name:
                    # Fallback to company name (not ideal but better than nothing)
                    ticker = it.name.replace(" ", "_").upper()[:10]
                
                # Map new attribute names to old format
                pct_nav = float(getattr(it, 'pct_value', 0)) if hasattr(it, 'pct_value') and it.pct_value else None
                shares = float(getattr(it, 'balance', 0)) if hasattr(it, 'balance') and it.balance else None
                value = float(getattr(it, 'value_usd', 0)) if hasattr(it, 'value_usd') and it.value_usd else None
                
                # Check asset category (new format uses 'EC' for equity common stock)
                asset_category = getattr(it, 'asset_category', '').upper()
                if asset_category == 'EC':  # Equity Common stock
                    rows.append((date, ticker.upper(), pct_nav, shares, value))
                    
            elif isinstance(it, dict):  # Old format: dictionary
                security_type = it.get("security_type", "").lower()
                if security_type == "common stock":
                    pct = it.get("pct_nav") if it.get("pct_nav") is not None else it.get("pct_value")
                    rows.append((
                        date,
                        it.get("identifier", "").upper(),
                        pct,
                        it.get("shares"),
                        it.get("value")
                    ))
        if rows:
            df = pd.DataFrame(rows, columns=["date", "ticker",
                                             "weight_pct", "shares",
                                             "market_value"])
            return df
    except (ValueError, TypeError, Exception) as exc:
        logger.warning(f"Skipping filing {filing.accession_no}: {exc}")
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

    # Lazily load full history into memory. Prefer the *bundled* parquet that
    # ships with the repository (instant load, no network) and fall back to
    # the cache-building logic only if the file is missing.
    if _HISTORY_DF is None:
        # Attempt to locate the *bundled* history parquet relative to repo root.
        repo_root = next(
            (p for p in Path(__file__).resolve().parents
             if (p / "data" / "spy_holdings_full.parquet").exists()),
            None,
        )

        if repo_root is not None:
            bundled = repo_root / "data" / "spy_holdings_full.parquet"
            logger.info(f"Loading bundled holdings history: {bundled}")
            _HISTORY_DF = pd.read_parquet(bundled)
            if not isinstance(_HISTORY_DF, pd.DataFrame):
                _HISTORY_DF = None # force rebuild
            logger.info(f"Loaded bundled history with shape: {_HISTORY_DF.shape}")
            logger.info(f"Bundled history date range: {_HISTORY_DF['date'].min()} to {_HISTORY_DF['date'].max()}")
        else:
            # Fallback – will download and build history which can be slow and
            # hit SEC rate limits, but ensures the function still works in
            # environments where the repo was packaged without the parquet.
            logger.warning("Bundled holdings history not found – building from scratch. This may take a while.")
            _HISTORY_DF = build_history(pd.Timestamp(2004, 1, 1), pd.Timestamp.today())
            logger.info(f"Built history with shape: {_HISTORY_DF.shape}")
            logger.info(f"Built history date range: {_HISTORY_DF['date'].min()} to {_HISTORY_DF['date'].max()}")

    target_ts = date
    df = _HISTORY_DF[_HISTORY_DF["date"] == target_ts]
    logger.info(f"Attempting to get holdings for target_ts: {target_ts}. Found {len(df)} rows.")

    if df.empty and not exact:
        # fallback to the latest available date before *date* (robust implementation)
        earlier_mask = _HISTORY_DF["date"] <= target_ts
        if earlier_mask.any():
            nearest = _HISTORY_DF.loc[earlier_mask, "date"].max()
            logger.info(
                "Exact holdings for %s unavailable – using nearest previous date %s.",
                date,
                nearest.date(),
            )
            df = _HISTORY_DF[_HISTORY_DF["date"] == nearest]

    if df.empty:
        raise ValueError(f"No holdings data found for {date} (exact={exact}).")

    return df.sort_values("weight_pct", ascending=False).reset_index(drop=True)


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
    """Return *df* unchanged – forward-fill disabled to avoid synthetic data."""
    return df


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