
import yfinance as yf
import pandas as pd
import os
import time
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console
from pathlib import Path
import logging

from .base_data_source import BaseDataSource

logger = logging.getLogger(__name__)

class YFinanceDataSource(BaseDataSource):
    """Data source for fetching data from Yahoo Finance."""

    def __init__(self, cache_expiry_hours=24):
        self.cache_expiry_hours = cache_expiry_hours
        try:
            SCRIPT_DIR = Path(__file__).parent.resolve()
        except NameError:
            SCRIPT_DIR = Path.cwd()
        self.data_dir = SCRIPT_DIR.parent.parent.parent / 'data'
        if logger.isEnabledFor(logging.DEBUG):

            logger.debug(f"YFinanceDataSource initialized. Data directory: {self.data_dir}")

    def _load_from_cache(self, file_path: Path, ticker: str) -> pd.DataFrame | None:
        """Attempts to load data from cache. Returns DataFrame if successful, None otherwise."""
        if file_path.exists() and (time.time() - os.path.getmtime(file_path)) / 3600 < self.cache_expiry_hours:
            try:
                df = pd.read_csv(file_path, index_col=0)
                df.index = pd.to_datetime(df.index, format='%Y-%m-%d', errors='coerce')
                df = df[~df.index.isnull()]
                if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
                    if logger.isEnabledFor(logging.DEBUG):

                        logger.debug(f"Cached file for {ticker} is empty or dates are invalid. Forcing re-download.")
                    return None
                if df.empty:
                    if logger.isEnabledFor(logging.DEBUG):

                        logger.debug(f"Cached file for {ticker} is empty. Forcing re-download.")
                    return None
                df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
                if logger.isEnabledFor(logging.DEBUG):

                    logger.debug(f"Loaded {ticker} from cache.")
                return df
            except Exception as e:
                if logger.isEnabledFor(logging.DEBUG):

                    logger.debug(f"Could not load {ticker} from cache ({e}). Cache might be corrupted.")
        return None

    def _download_data(self, ticker: str, start_date: str, end_date: str, file_path: Path) -> pd.DataFrame | None:
        """Downloads data from yfinance and saves to cache. Returns DataFrame if successful, None otherwise."""
        try:
            # Use auto_adjust=True to get adjusted prices
            downloaded_df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
            if downloaded_df is not None and not downloaded_df.empty:
                # When auto_adjust=True, yfinance typically returns OHLC adjusted,
                # and 'Close' is the adjusted close.
                # We save the full OHLCV for potential future use, even if only 'Close' is used now.
                downloaded_df.to_csv(file_path, index=True)
                if logger.isEnabledFor(logging.DEBUG):

                    logger.debug(f"Downloaded {ticker} successfully.")
                return downloaded_df
            else:
                if logger.isEnabledFor(logging.WARNING):

                    logger.warning(f"Could not download data for {ticker}. DataFrame is empty.")
        except Exception as e:
            logger.error(f"Failed to download {ticker}: {e}")
        return None

    def get_data(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Downloads or reads from cache the price data for a given list of tickers."""
        all_closes = []
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}.")

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            if logger.isEnabledFor(logging.INFO):
                logger.info(f"Created data directory at: {self.data_dir}")

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=Console()
        ) as progress:
            download_task = progress.add_task("[green]Downloading data...", total=len(tickers))

            for ticker in tickers:
                file_path = self.data_dir / f'{ticker}.csv'
                df = self._load_from_cache(file_path, ticker)

                if df is None:
                    progress.update(download_task, description=f"[yellow]Downloading {ticker}...")
                    df = self._download_data(ticker, start_date, end_date, file_path)

                if df is not None and not df.empty:
                    df.index.name = 'Date'
                    # With auto_adjust=True, 'Close' should be the adjusted close.
                    # If 'Close' is not present, it's an issue.
                    if 'Close' in df.columns:
                        close_series = df['Close']
                    # Fallback for unusual cases or if data format from cache/download changes unexpectedly
                    elif 'Adj Close' in df.columns:
                        if logger.isEnabledFor(logging.WARNING):

                            logger.warning(f"'Close' column not found for {ticker} despite auto_adjust=True. Using 'Adj Close'.")
                        close_series = df['Adj Close']
                    elif ticker in df.columns: # Should not happen with yf.download structure
                        if logger.isEnabledFor(logging.WARNING):

                            logger.warning(f"'Close' not found, using column named '{ticker}' for {ticker}.")
                        close_series = df[ticker]
                    else:
                        logger.error(f"Could not find 'Close' or 'Adj Close' column in downloaded/cached data for {ticker}")
                        # Skip this ticker if no valid close price column is found
                        progress.advance(download_task)
                        continue

                    close_series.name = ticker
                    all_closes.append(close_series)
                progress.advance(download_task)

        if not all_closes:
            logger.warning("No data fetched for any ticker.")
            return pd.DataFrame()

        result_df = pd.concat(all_closes, axis=1)
        if logger.isEnabledFor(logging.INFO):
            logger.info("Data fetching complete.")
        return result_df
