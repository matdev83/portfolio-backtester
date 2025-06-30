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

import pandas as pd
import requests
from lxml import etree
from tqdm import tqdm
from edgar import set_identity, Company        # EdgarTools ≥4.3.0

logger = logging.getLogger(__name__)

##############################################################################
# Config – change these four lines if you’d rather pull IVV or VOO instead
##############################################################################
TICKER   = "SPY"
CIK      = "0000884394"        # SPDR® S&P 500 ETF Trust
ISSUER_DAILY_URL = (
    "https://www.ssga.com/library-content/products/fund-data/etfs/us/"
    "holdings-daily-us-en-spy{suffix}.xlsx"
)
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
# 1) SSGA daily XLSX
# --------------------------------------------------------------------------- #
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
    url    = ISSUER_DAILY_URL.format(suffix=suffix, ymd=date.strftime("%Y-%m-%d"))

    resp = requests.get(url, headers=HEADERS_SSGA, timeout=30)
    if resp.status_code != 200:
        logger.warning(f"SSGA daily data not found for {date}: {resp.status_code}")
        return None

    df = pd.read_excel(io.BytesIO(resp.content), engine="openpyxl")
    df.columns = [c.strip().lower() for c in df.columns]
    # Normalise column names across vintages
    df = df.rename(columns={
        "weight (%)": "weight_pct",
        "% weight":   "weight_pct",
        "ticker":     "ticker",
        "shares":     "shares",
        "shares held": "shares",
        "market value ($)": "market_value",
        "market value (usd)": "market_value"
    })
    df["date"] = pd.to_datetime(date)
    df["ticker"] = df["ticker"].str.upper()
    
    result_df = df[["date", "ticker", "weight_pct", "shares", "market_value"]]
    
    # Save to cache
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
        if not isinstance(obj, dict):
            logger.warning(f"Skipping filing {filing.accession_no}: Object is not a dictionary.")
            return None
        items = obj.get("portfolio", [])
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
    init_edgar()
    frames = []

    logger.info("Downloading SSGA daily basket …")
    for d in tqdm(list(daterange(start, end))):
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
    logger.info(f"Successfully built history with {len(hist)} rows.")
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
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format='%(asctime)s - %(levelname)s - %(message)s')

    start = dt.datetime.strptime(args.start, "%Y-%m-%d").date()
    end   = dt.datetime.strptime(args.end,   "%Y-%m-%d").date()

    hist = build_history(start, end)

    # Save to ../data/ relative to the script's location
    script_dir = Path(__file__).parent
    out_dir = script_dir.parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out

    if out_path.suffix == ".csv":
        hist.to_csv(out_path, index=False)
    else:
        hist.to_parquet(out_path, index=False)

    logger.info(f"✓ Done. {len(hist):,} rows written to {out_path}")

if __name__ == "__main__":
    main()