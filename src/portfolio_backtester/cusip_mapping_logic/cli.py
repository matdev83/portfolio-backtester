import argparse
import sys
import pandas as pd
import logging

logger = logging.getLogger(__name__)
from .. import cusip_mapping

def main():
    parser = argparse.ArgumentParser(description="Update local CUSIP DB for all S&P 500 constituents")
    parser.add_argument("--update-missing", action="store_true", help="Fetch any still-unmapped tickers via OpenFIGI")
    parser.add_argument("--throttle", type=float, default=1.0, help="Seconds to wait between API calls (default 1.0)")
    args = parser.parse_args()

    if args.update_missing:
        if logger.isEnabledFor(logging.INFO):
            logger.info("Downloading S&P 500 constituent list from Wikipedia...")
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        symbols = set(tables[0].iloc[:, 0].tolist())
        symbols.update(tables[1].iloc[:, 1].dropna().tolist())
        symbols = {s.upper() for s in symbols if isinstance(s, str) and s.isalpha() and len(s) <= 5}

        db = cusip_mapping.CusipMappingDB()
        unresolved = [s for s in sorted(symbols) if s not in db._cache]
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Total unresolved tickers: {len(unresolved)}")
        for i, sym in enumerate(unresolved, 1):
            try:
                cusip, _ = db.resolve(sym, throttle=args.throttle)
                if logger.isEnabledFor(logging.INFO):
                    logger.info(f"[{i}/{len(unresolved)}] {sym} => {cusip}")
            except Exception as e:
                if logger.isEnabledFor(logging.ERROR):
                    logger.error(f"[{i}/{len(unresolved)}] {sym} FAILED: {e}")