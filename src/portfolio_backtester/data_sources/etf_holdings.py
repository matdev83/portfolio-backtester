import pandas as pd
import requests
import logging
from pathlib import Path
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ETFHoldingsDataSource:
    """
    Data source for fetching and caching ETF holdings data from ETF.com.
    Currently focused on IWC (iShares Micro-Cap ETF) for micro-cap exclusion.
    """
    def __init__(self, cache_dir: Path | str = None, cache_expiry_days: int = 7):
        if cache_dir is None:
            # Default to a cache directory within the project's data folder
            self.cache_dir = Path(__file__).parent.parent.parent / "cache" / "etf_holdings"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_expiry_days = cache_expiry_days
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.debug(f"ETFHoldingsDataSource initialized. Cache directory: {self.cache_dir}")

    def _get_cache_filepath(self, etf_ticker: str) -> Path:
        return self.cache_dir / f"{etf_ticker}_holdings.csv"

    def _load_from_cache(self, etf_ticker: str) -> pd.DataFrame | None:
        filepath = self._get_cache_filepath(etf_ticker)
        if filepath.exists():
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            if datetime.now() - file_mod_time < timedelta(days=self.cache_expiry_days):
                try:
                    df = pd.read_csv(filepath)
                    logger.debug(f"Loaded {etf_ticker} holdings from cache.")
                    return df
                except Exception as e:
                    logger.warning(f"Could not load {etf_ticker} holdings from cache: {e}")
        return None

    def _download_holdings(self, etf_ticker: str) -> pd.DataFrame | None:
        url = f"https://www.etf.com/etf-holdings-data-download?type=holdings&ticker={etf_ticker}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        try:
            logger.info(f"Downloading {etf_ticker} holdings from ETF.com...")
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
            
            # ETF.com sometimes returns HTML even for data download links if there's an issue
            if "html" in response.headers.get("Content-Type", ""):
                logger.error(f"Received HTML instead of CSV for {etf_ticker} holdings. Check URL or website.")
                return None

            # The content is typically CSV
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            # Basic validation: check for expected columns
            if 'Ticker' not in df.columns or 'Weight' not in df.columns:
                logger.error(f"Downloaded data for {etf_ticker} missing expected columns (Ticker, Weight). Available columns: {df.columns.tolist()}")
                return None

            # Clean up ticker symbols (e.g., remove leading/trailing spaces, convert to uppercase)
            df['Ticker'] = df['Ticker'].astype(str).str.strip().str.upper()
            
            # Save to cache
            filepath = self._get_cache_filepath(etf_ticker)
            df.to_csv(filepath, index=False)
            logger.info(f"Successfully downloaded and cached {etf_ticker} holdings.")
            return df
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {etf_ticker} holdings: {e}")
        except pd.errors.EmptyDataError:
            logger.warning(f"Downloaded file for {etf_ticker} is empty.")
        except Exception as e:
            logger.error(f"An unexpected error occurred while downloading {etf_ticker} holdings: {e}")
        return None

    def get_holdings(self, etf_ticker: str) -> pd.DataFrame | None:
        """
        Retrieves ETF holdings data, either from cache or by downloading.
        """
        df = self._load_from_cache(etf_ticker)
        if df is None:
            df = self._download_holdings(etf_ticker)
        return df

    def get_micro_cap_tickers(self, etf_ticker: str = "IWC", weight_threshold: float = 0.0001) -> List[str]:
        """
        Retrieves a list of micro-cap tickers from the specified ETF holdings.
        Filters out tickers with very low weight (e.g., less than 0.01%).
        """
        holdings_df = self.get_holdings(etf_ticker)
        if holdings_df is None or holdings_df.empty:
            logger.warning(f"Could not retrieve holdings for {etf_ticker}. Returning empty micro-cap list.")
            return []
        
        # Filter by weight and return unique tickers
        micro_cap_tickers = holdings_df[holdings_df['Weight'] > weight_threshold]['Ticker'].unique().tolist()
        logger.info(f"Identified {len(micro_cap_tickers)} micro-cap tickers from {etf_ticker} holdings.")
        return micro_cap_tickers
