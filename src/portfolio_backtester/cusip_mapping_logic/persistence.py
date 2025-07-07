import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def append_to_db(live_db_file: Path, ticker: str, cusip: str, name: str, *, source: str) -> None:
    try:
        with live_db_file.open("a", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([ticker, cusip, name, source])
    except Exception as e:
        logger.warning("Could not write to DB: %s", e)

def load_seeds(seed_files: list[Path], cache: dict) -> None:
    for path in seed_files:
        if not path.exists():
            continue
        try:
            with path.open() as fh:
                reader = csv.reader(fh)
                for row in reader:
                    if len(row) < 2:
                        continue
                    cusip = row[0]
                    ticker = row[1]
                    name = row[2] if len(row) >= 3 else ""
                    ticker = ticker.strip().upper()
                    cusip = cusip.strip()
                    name = name.strip()
                    if ticker and cusip and 8 <= len(cusip) <= 9:
                        cache.setdefault(ticker, (cusip, name))
        except Exception as e:
            logger.debug("Could not read seed %s: %s", path, e)

def load_live_db(live_db_file: Path, cache: dict) -> None:
    try:
        with live_db_file.open() as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                ticker = row["ticker"].upper()
                cusip = row["cusip"]
                name = row.get("name", "")
                if ticker and cusip:
                    cache[ticker] = (cusip, name)
    except Exception as e:
        logger.debug("Could not read live DB: %s", e)