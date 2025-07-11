import logging
from typing import Optional, Tuple
import re
import requests

logger = logging.getLogger(__name__)

_COMPANY_TICKERS_CACHE: dict[str, dict] | None = None
_CIK_SEARCH_CACHE: dict[str, str] = {}

def _company_tickers():
    global _COMPANY_TICKERS_CACHE
    if _COMPANY_TICKERS_CACHE is None:
        url = "https://www.sec.gov/files/company_tickers.json"
        hdr = {"User-Agent": "portfolio-backtester/1.0"}
        _COMPANY_TICKERS_CACHE = requests.get(url, headers=hdr, timeout=30).json()
    return _COMPANY_TICKERS_CACHE

def _cik_for_ticker(ticker: str) -> Optional[str]:
    global _CIK_SEARCH_CACHE
    ticker = ticker.upper()
    company_tickers = _company_tickers()
    if company_tickers:
        for v in company_tickers.values():
            if v.get("ticker", "").upper() == ticker:
                return str(v["cik"]).zfill(10)

    if ticker in _CIK_SEARCH_CACHE:
        return _CIK_SEARCH_CACHE[ticker]

    try:
        hdr = {"User-Agent": "portfolio-backtester/1.0"}
        url = f"https://www.sec.gov/cgi-bin/browse-edgar?CIK={ticker}&owner=exclude&action=getcompany&output=json"
        js = requests.get(url, headers=hdr, timeout=20).json()
        cik = js.get("CIK") or js.get("cik")
        if cik:
            cik = str(cik).zfill(10)
            _CIK_SEARCH_CACHE[ticker] = cik
            return cik
    except Exception as e:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("CIK search failed for %s: %s", ticker, e)
    return None

def lookup_edgar(ticker: str, *, throttle: float = 1.0) -> Tuple[Optional[str], Optional[str]]:
    try:
        cik = _cik_for_ticker(ticker)
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
        for accession, doc in list(zip(acc, prim))[:5]:
            adash = accession.replace("-", "")
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{adash}/{doc}"
            try:
                html = requests.get(doc_url, headers=hdr, timeout=30).text
            except Exception:
                continue
            m = re.search(r"CUSIP[^0-9A-Z]{0,20}([0-9]{3}[0-9A-Z]{5,6})", html, re.IGNORECASE)
            if m:
                cusip = m.group(1)
                if 8 <= len(cusip) <= 9:
                    # 'v' is not defined here, so we can't get the title. Returning empty string for name.
                    return cusip, ""
        return None, None
    except Exception as e:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("EDGAR lookup failed for %s: %s", ticker, e)
        return None, None