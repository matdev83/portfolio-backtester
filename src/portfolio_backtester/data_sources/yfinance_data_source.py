
import yfinance as yf
import pandas as pd
import os
import time
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console
from pathlib import Path
import logging

from .base_data_source import BaseDataSource

logger = logging.getLogger(__name__)

class YFinanceDataSource(BaseDataSource):
    """Data source for fetching data from Yahoo Finance."""

    def __init__(self, cache_expiry_hours=6):
        self.cache_expiry_hours = cache_expiry_hours
        try:
            SCRIPT_DIR = Path(__file__).parent.resolve()
        except NameError:
            SCRIPT_DIR = Path.cwd()
        self.data_dir = SCRIPT_DIR.parent.parent / 'data'
        logger.debug(f"YFinanceDataSource initialized. Data directory: {self.data_dir}")

    def _load_from_cache(self, file_path: Path, ticker: str) -> pd.DataFrame | None:
        """Attempts to load data from cache. Returns DataFrame if successful, None otherwise."""
        if file_path.exists() and (time.time() - os.path.getmtime(file_path)) / 3600 < self.cache_expiry_hours:
            try:
                df = pd.read_csv(file_path, index_col=0)
                df.index = pd.to_datetime(df.index, format='%Y-%m-%d', errors='coerce')
                df = df[~df.index.isnull()]
                if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
                    raise ValueError("Failed to parse dates from cached file.")
                df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
                logger.debug(f"Loaded {ticker} from cache.")
                return df
            except Exception as e:
                logger.debug(f"Could not load {ticker} from cache ({e}). Cache might be corrupted.")
        return None

    def _download_data(self, ticker: str, start_date: str, end_date: str, file_path: Path) -> pd.DataFrame | None:
        """Downloads data from yfinance and saves to cache. Returns DataFrame if successful, None otherwise."""
        try:
            downloaded_df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
            if downloaded_df is not None and not downloaded_df.empty:
                downloaded_df.to_csv(file_path, index=True)
                logger.debug(f"Downloaded {ticker} successfully.")
                return downloaded_df
            else:
                logger.warning(f"Could not download data for {ticker}. DataFrame is empty.")
        except Exception as e:
            logger.error(f"Failed to download {ticker}: {e}")
        return None

    def get_data(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Downloads or reads from cache the price data for a given list of tickers."""
        all_closes = []
        logger.info(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}.")

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"Created data directory at: {self.data_dir}")

        with Progress(
            SpinnerColumn(),
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
                    if 'Close' in df.columns:
                        close_series = df['Close']
                    elif 'Adj Close' in df.columns:
                        close_series = df['Adj Close']
                    elif ticker in df.columns:
                        close_series = df[ticker]
                    else:
                        logger.error(f"Could not find 'Close', 'Adj Close' or '{ticker}' column in downloaded data for {ticker}")
                        raise ValueError(f"Could not find 'Close', 'Adj Close' or '{ticker}' column in downloaded data for {ticker}")
                    close_series.name = ticker
                    all_closes.append(close_series)
                progress.advance(download_task)

        if not all_closes:
            logger.warning("No data fetched for any ticker.")
            return pd.DataFrame()

        result_df = pd.concat(all_closes, axis=1)
        logger.info("Data fetching complete.")
        return result_df
