import logging
from typing import Optional, Tuple
import re
import requests
import time

logger = logging.getLogger(__name__)

_CUSIP_RE = re.compile(r"\b(?=[0-9A-Z]*[A-Z])(?=[0-9A-Z]*[0-9])[0-9A-Z]{8,9}\b")


def lookup_duckduckgo(ticker: str, *, throttle: float = 1.0) -> Tuple[Optional[str], Optional[str]]:
    """Scrape first search-result page for a 8-9-char alnum CUSIP."""
    try:
        time.sleep(throttle)
        url = f"https://duckduckgo.com/html/?q={ticker}+cusip"
        headers = {"User-Agent": "portfolio-backtester/1.0 (+https://github.com/)"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        html = resp.text
        hits = set()
        for m in _CUSIP_RE.finditer(html):
            hits.add(m.group(0))
        hits = {h for h in hits if len(h) in (8, 9)}
        if len(hits) == 1:
            cusip = hits.pop()
            return cusip, ""
    except Exception as e:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("DuckDuckGo lookup failed for %s: %s", ticker, e)
    return None, None
