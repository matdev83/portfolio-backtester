import setuptools  # Ensure setuptools is imported before pandas_datareader
import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
import pandas as pd
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

from .base_data_source import BaseDataSource
from .stooq_data_source import StooqDataSource
from .yfinance_data_source import YFinanceDataSource

logger = logging.getLogger(__name__)

class HybridDataSource(BaseDataSource):
    """
    Hybrid data source that uses Stooq as primary source and yfinance as fallback.
    
    This data source implements a fail-tolerance workflow:
    1. Attempts to fetch data from Stooq (default)
    2. For any failed tickers, falls back to yfinance
    3. Handles format differences between the two sources
    4. Provides comprehensive error handling and reporting
    """

    def __init__(self, cache_expiry_hours: int = 24, prefer_stooq: bool = True) -> None:
        """
        Initialize the hybrid data source.
        
        Args:
            cache_expiry_hours: Hours before cached data expires
            prefer_stooq: Whether to prefer Stooq over yfinance (default: True)
        """
        self.cache_expiry_hours = cache_expiry_hours
        self.prefer_stooq = prefer_stooq
        
        # Initialize both data sources
        self.stooq_source = StooqDataSource(cache_expiry_hours=cache_expiry_hours)
        self.yfinance_source = YFinanceDataSource(cache_expiry_hours=cache_expiry_hours)
        
        # Use the same data directory as the individual sources
        try:
            SCRIPT_DIR = Path(__file__).parent.resolve()
        except NameError:
            SCRIPT_DIR = Path.cwd()
        self.data_dir = SCRIPT_DIR.parent.parent.parent / "data"
        
        # Track which tickers failed from which sources
        self.failed_tickers: Dict[str, Set[str]] = {
            'stooq': set(),
            'yfinance': set()
        }
        
        logger.debug(f"HybridDataSource initialized. Data directory: {self.data_dir}, prefer_stooq: {prefer_stooq}")

    def _is_data_valid(self, df: pd.DataFrame, ticker: str, min_rows: int = 3) -> bool:
        """
        Check if the downloaded data is valid and usable.
        
        Args:
            df: DataFrame to validate
            ticker: Ticker symbol for logging
            min_rows: Minimum number of rows required
            
        Returns:
            True if data is valid, False otherwise
        """
        if df is None or df.empty:
            logger.debug(f"Data validation failed for {ticker}: DataFrame is None or empty")
            return False
        
        # Check if we have enough data points
        if len(df) < min_rows:
            logger.debug(f"Data validation failed for {ticker}: Only {len(df)} rows (minimum: {min_rows})")
            return False
        
        # Check for valid date index
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.debug(f"Data validation failed for {ticker}: Index is not DatetimeIndex")
            return False
        
        # Check for required columns (at least Close should be present)
        if isinstance(df.columns, pd.MultiIndex):
            # For MultiIndex, check if Close exists in the Field level
            if 'Close' not in df.columns.get_level_values('Field'):
                logger.debug(f"Data validation failed for {ticker}: No Close column in MultiIndex")
                return False
            
            # Check if the specific ticker exists in the Ticker level
            if ticker not in df.columns.get_level_values('Ticker'):
                logger.debug(f"Data validation failed for {ticker}: Ticker not found in MultiIndex")
                return False
        else:
            # For flat columns, check if ticker column exists
            if ticker not in df.columns:
                logger.debug(f"Data validation failed for {ticker}: Ticker column not found. Available: {df.columns.tolist()}")
                return False
        
        # Extract close prices for validation
        try:
            if isinstance(df.columns, pd.MultiIndex):
                close_series = df[(ticker, 'Close')]
            else:
                close_series = df[ticker]
        except KeyError:
            logger.debug(f"Data validation failed for {ticker}: Cannot extract close price data")
            return False
        
        # Check for excessive NaN values (more than 50% is suspicious)
        nan_ratio = close_series.isna().sum() / len(close_series)
        if nan_ratio > 0.5:
            logger.debug(f"Data validation failed for {ticker}: Too many NaN values ({nan_ratio:.1%})")
            return False
        
        # Check for reasonable price values (positive and not too extreme)
        valid_prices = close_series.dropna()
        if len(valid_prices) == 0:
            logger.debug(f"Data validation failed for {ticker}: No valid price data")
            return False
        
        if (valid_prices <= 0).any():
            logger.debug(f"Data validation failed for {ticker}: Contains non-positive prices")
            return False
        
        # Check for extreme values (prices > $100,000 or < $0.01 are suspicious)
        if (valid_prices > 100000).any() or (valid_prices < 0.01).any():
            logger.debug(f"Data validation failed for {ticker}: Contains extreme price values")
            return False
        
        logger.debug(f"Data validation passed for {ticker}: {len(df)} rows, {nan_ratio:.1%} NaN ratio")
        return True

    def _normalize_data_format(self, df: pd.DataFrame, source: str, tickers: List[str]) -> pd.DataFrame:
        """
        Normalize data format from different sources to a consistent MultiIndex format.
        
        Args:
            df: DataFrame from the source
            source: Source name ('stooq' or 'yfinance')
            tickers: List of tickers that were requested
            
        Returns:
            Normalized DataFrame with MultiIndex columns (Ticker, Field)
        """
        if df is None or df.empty:
            return pd.DataFrame()
        
        logger.debug(f"Normalizing data format from {source} for {len(tickers)} tickers")
        
        # Stooq already returns MultiIndex format, yfinance returns flat format
        if source == 'stooq':
            # Stooq data should already be in MultiIndex format
            if isinstance(df.columns, pd.MultiIndex):
                return df
            else:
                # Fallback: convert to MultiIndex if somehow it's not
                logger.warning(f"Stooq data not in expected MultiIndex format, converting...")
                return self._convert_to_multiindex(df, tickers)
        
        elif source == 'yfinance':
            # yfinance returns flat format with ticker names as columns
            return self._convert_to_multiindex(df, tickers)
        
        else:
            raise ValueError(f"Unknown source: {source}")

    def _convert_to_multiindex(self, df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
        """
        Convert flat DataFrame to MultiIndex format (Ticker, Field).
        
        Args:
            df: Flat DataFrame with tickers as columns
            tickers: List of ticker symbols
            
        Returns:
            DataFrame with MultiIndex columns
        """
        if df.empty:
            return df
        
        # Create MultiIndex structure
        multiindex_data = []
        
        for ticker in tickers:
            if ticker in df.columns:
                # For yfinance, we typically only have Close prices
                ticker_data = pd.DataFrame({
                    'Close': df[ticker]
                })
                # Add other OHLC columns if available (though yfinance usually just gives Close)
                for col in ['Open', 'High', 'Low', 'Volume']:
                    if col in df.columns:
                        ticker_data[col] = df[col]
                    else:
                        # For missing OHLC data, use Close price as approximation
                        if col in ['Open', 'High', 'Low']:
                            ticker_data[col] = df[ticker]
                        elif col == 'Volume':
                            ticker_data[col] = np.nan
                
                # Create MultiIndex columns for this ticker
                ticker_data.columns = pd.MultiIndex.from_product(
                    [[ticker], ticker_data.columns], 
                    names=['Ticker', 'Field']
                )
                multiindex_data.append(ticker_data)
        
        if multiindex_data:
            result = pd.concat(multiindex_data, axis=1)
            logger.debug(f"Converted to MultiIndex format: {result.shape}")
            return result
        else:
            logger.warning("No valid ticker data found for MultiIndex conversion")
            return pd.DataFrame()

    def _fetch_from_source(self, source: str, tickers: List[str], start_date: str, end_date: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Fetch data from a specific source and return successful/failed tickers.
        
        Args:
            source: Source name ('stooq' or 'yfinance')
            tickers: List of tickers to fetch
            start_date: Start date string
            end_date: End date string
            
        Returns:
            Tuple of (data_df, successful_tickers, failed_tickers)
        """
        if not tickers:
            return pd.DataFrame(), [], []
        
        logger.info(f"Fetching data from {source} for {len(tickers)} tickers")
        
        try:
            if source == 'stooq':
                data_df = self.stooq_source.get_data(tickers, start_date, end_date)
            elif source == 'yfinance':
                data_df = self.yfinance_source.get_data(tickers, start_date, end_date)
            else:
                raise ValueError(f"Unknown source: {source}")
            
            if data_df is None or data_df.empty:
                logger.warning(f"No data returned from {source}")
                return pd.DataFrame(), [], tickers
            
            # Normalize the data format
            normalized_df = self._normalize_data_format(data_df, source, tickers)
            
            # Check which tickers were successfully fetched
            successful_tickers = []
            failed_tickers = []
            
            for ticker in tickers:
                # Check if ticker exists in the normalized data
                if isinstance(normalized_df.columns, pd.MultiIndex):
                    ticker_cols = [col for col in normalized_df.columns if col[0] == ticker]
                    if ticker_cols:
                        # Extract ticker data for validation
                        ticker_data = normalized_df[ticker_cols]
                        if self._is_data_valid(ticker_data, ticker):
                            successful_tickers.append(ticker)
                        else:
                            failed_tickers.append(ticker)
                    else:
                        failed_tickers.append(ticker)
                else:
                    if ticker in normalized_df.columns:
                        ticker_data = normalized_df[[ticker]]
                        if self._is_data_valid(ticker_data, ticker):
                            successful_tickers.append(ticker)
                        else:
                            failed_tickers.append(ticker)
                    else:
                        failed_tickers.append(ticker)
            
            logger.info(f"{source} results: {len(successful_tickers)} successful, {len(failed_tickers)} failed")
            if failed_tickers:
                logger.debug(f"{source} failed tickers: {failed_tickers}")
            
            return normalized_df, successful_tickers, failed_tickers
            
        except Exception as e:
            logger.error(f"Error fetching data from {source}: {e}")
            return pd.DataFrame(), [], tickers

    def get_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch data using fail-tolerance workflow: Stooq first, then yfinance fallback.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            
        Returns:
            DataFrame with MultiIndex columns (Ticker, Field)
        """
        if not tickers:
            logger.warning("No tickers provided")
            return pd.DataFrame()
        
        logger.info(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        # Reset failure tracking
        self.failed_tickers = {'stooq': set(), 'yfinance': set()}
        
        # Determine primary and fallback sources
        if self.prefer_stooq:
            primary_source = 'stooq'
            fallback_source = 'yfinance'
        else:
            primary_source = 'yfinance'
            fallback_source = 'stooq'
        
        all_data_frames = []
        remaining_tickers = tickers.copy()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=Console(),
        ) as progress:
            
            # Step 1: Try primary source
            primary_task = progress.add_task(f"[green]Fetching from {primary_source}...", total=len(tickers))
            
            primary_data, primary_successful, primary_failed = self._fetch_from_source(
                primary_source, remaining_tickers, start_date, end_date
            )
            
            if not primary_data.empty:
                all_data_frames.append(primary_data)
            
            self.failed_tickers[primary_source].update(primary_failed)
            progress.update(primary_task, completed=len(primary_successful))
            
            # Step 2: Try fallback source for failed tickers
            if primary_failed:
                fallback_task = progress.add_task(f"[yellow]Fallback to {fallback_source}...", total=len(primary_failed))
                
                fallback_data, fallback_successful, fallback_failed = self._fetch_from_source(
                    fallback_source, primary_failed, start_date, end_date
                )
                
                if not fallback_data.empty:
                    all_data_frames.append(fallback_data)
                
                self.failed_tickers[fallback_source].update(fallback_failed)
                progress.update(fallback_task, completed=len(fallback_successful))
                
                # Update primary task to show total progress
                progress.update(primary_task, completed=len(primary_successful) + len(fallback_successful))
        
        # Combine all data
        if all_data_frames:
            result_df = pd.concat(all_data_frames, axis=1)
            # Remove duplicate columns if any
            if isinstance(result_df.columns, pd.MultiIndex):
                result_df = result_df.loc[:, ~result_df.columns.duplicated()]
        else:
            result_df = pd.DataFrame()
        
        # Report final results
        total_failed = len(self.failed_tickers['stooq'] & self.failed_tickers['yfinance'])
        total_successful = len(tickers) - total_failed
        
        logger.info(f"Data fetching complete: {total_successful}/{len(tickers)} tickers successful")
        
        if total_failed > 0:
            completely_failed = self.failed_tickers['stooq'] & self.failed_tickers['yfinance']
            logger.warning(f"Failed to fetch data for {total_failed} tickers: {list(completely_failed)}")
        
        # Log source usage statistics
        stooq_only = len(self.failed_tickers['yfinance'] - self.failed_tickers['stooq'])
        yfinance_only = len(self.failed_tickers['stooq'] - self.failed_tickers['yfinance'])
        logger.info(f"Source usage: {stooq_only} from Stooq only, {yfinance_only} from yfinance fallback")
        
        return result_df

    def get_failure_report(self) -> Dict[str, any]:
        """
        Get a detailed report of data fetching failures.
        
        Returns:
            Dictionary containing failure statistics and details
        """
        return {
            'stooq_failures': list(self.failed_tickers['stooq']),
            'yfinance_failures': list(self.failed_tickers['yfinance']),
            'total_failures': list(self.failed_tickers['stooq'] & self.failed_tickers['yfinance']),
            'stooq_only_count': len(self.failed_tickers['yfinance'] - self.failed_tickers['stooq']),
            'yfinance_only_count': len(self.failed_tickers['stooq'] - self.failed_tickers['yfinance']),
            'complete_failure_count': len(self.failed_tickers['stooq'] & self.failed_tickers['yfinance'])
        } 