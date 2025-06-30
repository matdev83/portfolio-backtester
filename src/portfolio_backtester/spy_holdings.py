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
EARLIEST_SSGA_DATE = dt.date(2011, 1, 3)
# URL patterns for SSGA daily basket across historical site changes
URL_PATTERNS = [
    # current pattern 2017-08 → present  (.xlsx, optional "-YYYYMMDD" suffix)
    "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy{suffix}.xlsx",
    # 2015-03 → 2017-07 (.xlsx – extra "us" in path)
    "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-us-spy{suffix}.xlsx",
    # 2011-03 → 2015-02 (.xls, different host & hard-coded dash before date)
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
UA = os.getenv("SEC_USER_AGENT", "mateusz@bartczak.me Data Downloader")
HEADERS_SEC   = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}
HEADERS_SSGA  = {"User-Agent": "Mozilla/5.0"}

def daterange(start, end):
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)

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
_CACHE_EXPIRY_HOURS = 6

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
    return pd.read_excel(io.BytesIO(buf), engine="openpyxl")

# --------------------------------------------------------------------------- #
# 1) SSGA daily XLSX
# --------------------------------------------------------------------------- #
def _fetch_from_wayback(orig_url: str, date: dt.date) -> bytes | None:
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

def ssga_daily(date):
    """
    Return DataFrame for one date or None if 404.
    Uses a 6-hour cache in ../cache/ssga_daily/
    """
    script_dir = Path(__file__).parent
    cache_dir = script_dir.parent.parent / "cache" / "ssga_daily"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{date:%Y-%m-%d}.parquet"

    # Check cache
    if cache_file.exists():
        mod_time = dt.datetime.fromtimestamp(cache_file.stat().st_mtime)
        if (dt.datetime.now() - mod_time) < dt.timedelta(hours=6):
            return pd.read_parquet(cache_file)

    # Download if not in cache or cache is stale
    suffix = "" if date == dt.date.today() else f"-{date:%Y%m%d}"
    tried_urls: list[str] = []
    last_status: int | None = None

    # 1) Direct download attempts ------------------------------------------------
    for pattern in URL_PATTERNS:
        url_try = pattern.format(suffix=suffix, ymd=date.strftime("%Y%m%d"))
        tried_urls.append(url_try)
        resp = requests.get(url_try, headers=HEADERS_SSGA, timeout=30)
        last_status = resp.status_code
        if resp.status_code == 200:
            ext = Path(url_try).suffix.lower()
            df = _read_excel_bytes(resp.content, file_ext=ext)
            break
    else:
        # 2) Wayback fallback ----------------------------------------------------
        df = None
        for orig_url in tried_urls:
            archive_bytes = _fetch_from_wayback(orig_url, date)
            if archive_bytes:
                ext = Path(orig_url).suffix.lower()
                try:
                    df = _read_excel_bytes(archive_bytes, file_ext=ext)
                    logger.info(
                        f"\033[92m✓ SSGA {date:%Y-%m-%d} basket downloaded via Wayback ({len(df)} rows)\033[0m"
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    logger.debug(f"Wayback parse failed for {orig_url}: {exc}")

    if df is None:
        logger.warning(f"SSGA daily data not found for {date}: {last_status}")
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
        "market value (usd)":  "market_value",
    })

    df["date"] = pd.to_datetime(date)
    df["ticker"] = df["ticker"].str.upper()

    result_df = df[["date", "ticker", "weight_pct", "shares", "market_value"]]

    # We only log success for direct downloads earlier; skip repeating.
    if last_status == 200:
        logger.info(
            f"\033[92m✓ SSGA {date:%Y-%m-%d} basket downloaded ({len(result_df)} rows)\033[0m"
        )

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
            date = dt.datetime.strptime(period_of_report, "%Y-%m-%d").date()
        elif isinstance(period_of_report, dt.date):
            date = period_of_report
        else:
            logger.warning(f"Skipping filing {filing.accession_no}: Invalid period_of_report type.")
            return None

        if not (start_date <= date <= end_date):
            logger.debug(f"Skipping filing {filing.accession_no}: Outside date range.")
            return None

        obj = filing.obj()
        # edgartools v4 returns FundReport dataclass; v3 returns dict
        if FundReport is not None and isinstance(obj, FundReport):
            items = obj.portfolio or []
        elif isinstance(obj, dict):
            items = obj.get("portfolio", [])
        else:
            logger.warning(f"Skipping filing {filing.accession_no}: Unexpected obj type {type(obj)}")
            return None
        rows = [
            (
                date,
                it.get("identifier", "").upper(),
                it.get("pct_nav"),
                it.get("shares"),
                it.get("value")
            )
            for it in items
            if it.get("security_type", "").lower() == "common stock"
        ]
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

def _get_sec_filings(company: Company, start_date: dt.date) -> list:
    """Fetches and filters SEC filings for NPORT-P and N-Q forms."""
    forms = company.get_filings().filter(form=["NPORT-P", "N-Q"])
    nport = [f for f in forms if f.form == "NPORT-P" and start_date <= f.filing_date]
    nq    = [f for f in forms if f.form == "N-Q"      and start_date <= f.filing_date]
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
def build_history(start, end):
    # Try loading an aggregated cache first
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

    _save_history_to_cache(hist, start, end)

    return hist

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Download SPY holdings history")
    parser.add_argument("--start", default="2004-01-01",
                        help="YYYY-MM-DD (default 2004-01-01, earliest SEC N-Q)")
    parser.add_argument("--end",   default=str(dt.date.today()),
                        help="YYYY-MM-DD (default today)")
    parser.add_argument("--out",   required=True,
                        help="Output filename (.parquet or .csv)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level.")
    parser.add_argument("--update", action="store_true", help="Update the full history.")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format='%(asctime)s - %(levelname)s - %(message)s')

    start = dt.datetime.strptime(args.start, "%Y-%m-%d").date()
    end   = dt.datetime.strptime(args.end,   "%Y-%m-%d").date()

    script_dir = Path(__file__).parent
    out_dir = script_dir.parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out

    if args.update:
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
    return cache_dir / f"{TICKER}_history_{start:%Y-%m-%d}_{end:%Y-%m-%d}.parquet"


def _load_history_from_cache(start: dt.date, end: dt.date) -> Optional[pd.DataFrame]:
    """Load aggregate history DataFrame from cache or return None if not present / stale."""
    cache_file = _history_cache_file(start, end)
    if cache_file.exists():
        mod_time = dt.datetime.fromtimestamp(cache_file.stat().st_mtime)
        if (dt.datetime.now() - mod_time) < dt.timedelta(hours=_CACHE_EXPIRY_HOURS):
            logger.info(f"Loading aggregated holdings history from cache: {cache_file}")
            return pd.read_parquet(cache_file)
    return None


def _save_history_to_cache(df: pd.DataFrame, start: dt.date, end: dt.date) -> None:
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
        date = dt.datetime.strptime(date, "%Y-%m-%d").date()
    elif isinstance(date, pd.Timestamp):
        date = date.date()
    elif not isinstance(date, dt.date):
        raise TypeError("date must be a str, datetime.date or pandas.Timestamp")

    # Lazily load full history into memory (using cached parquet if available)
    if _HISTORY_DF is None:
        _HISTORY_DF = build_history(dt.date(2004, 1, 1), dt.date.today())

    target_ts = pd.Timestamp(date)
    df = _HISTORY_DF[_HISTORY_DF["date"] == target_ts]

    if df.empty and not exact:
        # fallback to the latest available date before *date*
        avail_dates = _HISTORY_DF["date"].unique()
        earlier = avail_dates[avail_dates <= target_ts]
        if len(earlier):
            nearest = max(earlier)
            logger.info(f"Exact holdings for {date} unavailable – using nearest previous date {nearest.date()}.")
            df = _HISTORY_DF[_HISTORY_DF["date"] == nearest]

    if df.empty:
        raise ValueError(f"No holdings data found for {date} (exact={exact}).")

    return df.sort_values("weight_pct", ascending=False).reset_index(drop=True)


def update_full_history(out_path: Path, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """Incrementally update *out_path* parquet file between *start_date* and *end_date*.

    If the file exists it is loaded and only the missing date range after the
    latest stored date is downloaded. If no update is necessary the existing
    DataFrame is returned unchanged.
    """
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        if not existing.empty:
            latest = existing["date"].max().date()
            # nothing new to fetch
            if latest >= end_date:
                logger.info("Parquet up-to-date ‑ no download required.")
                return existing
            # we need to extend from the day after latest
            incremental_start = latest + dt.timedelta(days=1)
            logger.info(f"Updating history from {incremental_start} to {end_date} …")
            new_df = build_history(incremental_start, end_date)
            combined = pd.concat([existing, new_df], ignore_index=True).drop_duplicates(subset=["date", "ticker"]).sort_values(["date", "ticker"]).reset_index(drop=True)
            combined = _forward_fill_history(combined, start_date, end_date)
        else:
            combined = build_history(start_date, end_date)
    else:
        logger.info(f"Creating new holdings history {out_path}")
        combined = build_history(start_date, end_date)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    logger.info(f"✓ Saved {len(combined):,} rows → {out_path}")
    return combined


def _forward_fill_history(df: pd.DataFrame, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Return DataFrame with missing business days forward-filled for each ticker."""
    if df.empty:
        return df

    full_dates = pd.date_range(start, end, freq="B")  # NYSE business days approximation

    def _ffill(group: pd.DataFrame) -> pd.DataFrame:
        g = (
            group.set_index("date")
                 .reindex(full_dates)
                 .sort_index()
                 .ffill()
        )
        g["ticker"] = group["ticker"].iloc[0]
        return g

    filled = (
        df.sort_values(["ticker", "date"])
          .groupby("ticker", group_keys=False)
          .apply(_ffill)
          .reset_index(names="date")
    )
    # keep only requested columns order
    return filled[["date", "ticker", "weight_pct", "shares", "market_value"]]


if __name__ == "__main__":
    main()