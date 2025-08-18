import os
import time
import logging
from pathlib import Path
from typing import List

import pandas as pd
from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .base_data_source import BaseDataSource

logger = logging.getLogger(__name__)


class StooqDataSource(BaseDataSource):
    """Data source for fetching daily OHLCV data from Stooq using pandas-datareader.

    The returned DataFrame emulates the layout produced by YFinanceDataSource –
    index is a DateTimeIndex of trading days and each column contains the Close
    price for a single ticker.  All downstream code should therefore continue to
    work unchanged.
    """

    def __init__(self, cache_expiry_hours: int = 24) -> None:
        self.cache_expiry_hours = cache_expiry_hours
        try:
            SCRIPT_DIR = Path(__file__).parent.resolve()
        except NameError:
            SCRIPT_DIR = Path.cwd()
        # Keep using the same cache folder as the YFinance backend for full
        # backward-compatibility.
        self.data_dir = SCRIPT_DIR.parent.parent.parent / "data"
        logger.debug(f"StooqDataSource initialised. Data directory: {self.data_dir}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_from_cache(self, file_path: Path, ticker: str) -> pd.DataFrame | None:
        """Return cached DataFrame if it is fresh enough, else None."""
        if (
            file_path.exists()
            and (time.time() - os.path.getmtime(file_path)) / 3600 < self.cache_expiry_hours
        ):
            try:
                df: pd.DataFrame = pd.read_csv(file_path, index_col=0)
                df.index = pd.to_datetime(df.index, format="%Y-%m-%d", errors="coerce")
                # Remove rows with null indices
                valid_indices = df.index.dropna()
                df = df.loc[valid_indices]
                if df.empty or not isinstance(df.index, pd.DatetimeIndex):
                    logger.debug(
                        f"Cached file for {ticker} is empty or has invalid dates. Forcing re-download."
                    )
                    return None

                # Convert numeric columns to proper types
                numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                logger.debug(f"Loaded {ticker} from cache.")
                return df
            except (
                Exception
            ) as exc:  # pragma: no cover – any parsing error => fallback to live download
                logger.debug(
                    f"Could not load {ticker} from cache ({exc}). Cache might be corrupted."
                )
        return None

    def _download_data(
        self, ticker: str, start_date: str, end_date: str, file_path: Path
    ) -> pd.DataFrame | None:
        """Download data from Stooq and write it to *file_path* as CSV."""
        try:
            # Import pandas_datareader lazily to avoid triggering deprecated warnings
            # at import/collection time in test environments that do not use
            # this data source. Importing here prevents collection-time side
            # effects from third-party libraries.
            try:
                import pandas_datareader.data as web  # local import
            except Exception as imp_exc:
                logger.warning(f"pandas_datareader unavailable: {imp_exc}")
                return None

            downloaded_df: pd.DataFrame = web.DataReader(
                ticker, "stooq", start=start_date, end=end_date
            )
            if downloaded_df is not None and not downloaded_df.empty:
                # Ensure chronological order (oldest first) because some Stooq
                # feeds are returned newest-first.
                downloaded_df.sort_index(inplace=True)
                downloaded_df.to_csv(file_path, index=True)
                logger.debug(f"Downloaded {ticker} successfully from Stooq.")
                return downloaded_df
            else:
                logger.warning(f"No data returned for {ticker} from Stooq.")
        except Exception as exc:
            logger.error(f"Failed to download {ticker} from Stooq: {exc}")
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch daily OHLCV data for *tickers* between *start_date* and *end_date*.

        Returns a DataFrame with MultiIndex columns: (Ticker, Field) where Field
        includes 'Open', 'High', 'Low', 'Close', 'Volume'.
        """
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            if logger.isEnabledFor(logging.INFO):
                logger.info(f"Created data directory at: {self.data_dir}")

        all_ticker_data = []
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                f"Fetching data from Stooq for {len(tickers)} tickers from {start_date} to {end_date}."
            )

        # Map tickers that differ between Yahoo and Stooq so the rest of the
        # codebase (which expects Yahoo symbols like "^GSPC") keeps working.
        SYMBOL_MAP = {"^GSPC": "^SPX"}

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=Console(),
        ) as progress:
            dl_task = progress.add_task("[green]Downloading data...", total=len(tickers))

            for orig_ticker in tickers:
                fetch_ticker = SYMBOL_MAP.get(orig_ticker, orig_ticker)

                file_path = self.data_dir / f"{orig_ticker}.csv"

                df = self._load_from_cache(file_path, orig_ticker)
                if df is None:
                    progress.update(dl_task, description=f"[yellow]Downloading {orig_ticker}…")
                    df = self._download_data(fetch_ticker, start_date, end_date, file_path)

                if df is not None and not df.empty:
                    df.index.name = "Date"

                    # Ensure we have the expected OHLCV columns
                    expected_columns = ["Open", "High", "Low", "Close", "Volume"]
                    available_columns = [col for col in expected_columns if col in df.columns]

                    if not available_columns:
                        logger.error(
                            f"No OHLCV columns found for {orig_ticker}. Available columns: {df.columns.tolist()}"
                        )
                        progress.advance(dl_task)
                        continue

                    # Create MultiIndex DataFrame for this ticker
                    ticker_df = df[available_columns].copy()

                    # Create MultiIndex columns: (Ticker, Field)
                    ticker_df.columns = pd.MultiIndex.from_product(
                        [[orig_ticker], ticker_df.columns], names=["Ticker", "Field"]
                    )

                    all_ticker_data.append(ticker_df)

                progress.advance(dl_task)

        if not all_ticker_data:
            logger.warning("No data fetched for any ticker from Stooq.")
            return pd.DataFrame()

        # Concatenate all ticker data
        result_df = pd.concat(all_ticker_data, axis=1)
        if logger.isEnabledFor(logging.INFO):
            logger.info("Data fetching from Stooq complete.")
        return result_df
