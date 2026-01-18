"""Market Data Multi-Provider data source adapter.

This module provides a data source implementation that wraps the
market-data-multi-provider package, enabling transparent integration
with portfolio-backtester's existing data source architecture.

Features:
- Compatible with BaseDataSource interface
- Automatic symbol mapping (local ↔ canonical format)
- Multi-provider fallback (stooq → yfinance → tradingview)
- Centralized caching via MDMP canonical store
- Coverage validation with trading calendars
"""

import logging
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .base_data_source import BaseDataSource
from .symbol_mapper import to_canonical_id, from_canonical_id

logger = logging.getLogger(__name__)


class MarketDataMultiProviderDataSource(BaseDataSource):
    """Data source using market-data-multi-provider package.

    This adapter wraps MarketDataClient to provide compatibility with
    portfolio-backtester's BaseDataSource interface. It handles:

    1. Symbol format conversion (SPY → AMEX:SPY)
    2. Data fetching via MDMP's multi-provider system
    3. Output normalization to MultiIndex DataFrame format
    4. Error handling and logging

    Attributes:
        client: MarketDataClient instance
        data_dir: Path to data storage directory
    """

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        cache_expiry_hours: int = 24,
        min_coverage_ratio: Optional[float] = None,
    ) -> None:
        """Initialize the data source.

        Args:
            data_dir: Directory for data storage (uses MDMP default if None)
            cache_expiry_hours: Ignored - MDMP handles caching internally.
                Kept for interface compatibility.
            min_coverage_ratio: Minimum coverage ratio for NYSE calendar validation.
                Set to None to disable coverage checks (default).
        """
        try:
            from market_data_multi_provider import MarketDataClient  # type: ignore[attr-defined]
        except ImportError as e:
            raise ImportError(
                "market-data-multi-provider is not installed. "
                "Install with: pip install -e ../market-data-multi-provider"
            ) from e

        self.client = MarketDataClient(data_dir=data_dir)
        self.data_dir = Path(data_dir) if data_dir else None
        self._cache_expiry_hours = cache_expiry_hours  # Stored for interface compat
        self._min_coverage_ratio = min_coverage_ratio

        logger.debug(
            f"MarketDataMultiProviderDataSource initialized. "
            f"Data dir: {self.data_dir or 'MDMP default'}"
        )

    def get_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for tickers.

        This method:
        1. Converts local tickers to canonical IDs
        2. Fetches data using MDMP client
        3. Normalizes output to expected MultiIndex format

        Args:
            tickers: List of ticker symbols (e.g., ["SPY", "QQQ", "AAPL"])
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with MultiIndex columns (Ticker, Field) where Field
            includes 'Open', 'High', 'Low', 'Close', 'Volume'.
            Returns empty DataFrame if no data available.

        Raises:
            ImportError: If market-data-multi-provider is not installed
        """
        if not tickers:
            logger.warning("No tickers provided")
            return pd.DataFrame()

        logger.info(f"Fetching {len(tickers)} tickers from {start_date} to {end_date} via MDMP")

        # Parse dates
        try:
            start = self._parse_date(start_date)
            end = self._parse_date(end_date)
        except ValueError as e:
            logger.error(f"Invalid date format: {e}")
            return pd.DataFrame()

        # Build symbol mapping: canonical_id -> original_ticker
        symbol_map: Dict[str, str] = {}
        canonical_ids: List[str] = []

        for ticker in tickers:
            canonical = to_canonical_id(ticker)
            symbol_map[canonical] = ticker
            canonical_ids.append(canonical)
            logger.debug(f"Mapped {ticker} → {canonical}")

        # Fetch data using MDMP
        try:
            results = self.client.fetch_many(
                canonical_ids,
                start=start,
                end=end,
                use_cache=True,
                use_canonical=True,
                min_coverage_ratio=self._min_coverage_ratio,
            )
        except Exception as e:
            logger.error(f"MDMP fetch_many failed: {e}")
            return pd.DataFrame()

        # Process results into MultiIndex format
        # results is a dict[str, pd.DataFrame] mapping canonical_id -> DataFrame
        all_ticker_data: List[pd.DataFrame] = []
        successful = 0
        failed = 0
        failed_tickers: List[str] = []

        for canonical_id, df in results.items():
            original_ticker = symbol_map.get(canonical_id, from_canonical_id(canonical_id))

            if df is None or df.empty:
                logger.warning(f"No data for {original_ticker} ({canonical_id})")
                failed += 1
                failed_tickers.append(original_ticker)
                continue

            # Validate and extract OHLCV columns
            ticker_df = self._normalize_ohlcv(df, original_ticker)

            if ticker_df is not None and not ticker_df.empty:
                all_ticker_data.append(ticker_df)
                successful += 1
                logger.debug(f"Fetched {original_ticker}: {len(ticker_df)} rows")
            else:
                failed += 1
                failed_tickers.append(original_ticker)

        # Log summary
        logger.info(
            f"MDMP fetch complete: {successful}/{len(tickers)} successful, " f"{failed} failed"
        )
        if failed_tickers:
            logger.warning(f"Failed tickers: {failed_tickers[:10]}")

        if not all_ticker_data:
            logger.warning("No data fetched for any ticker")
            return pd.DataFrame()

        # Combine all ticker data
        result = pd.concat(all_ticker_data, axis=1)

        # Ensure DatetimeIndex
        if not isinstance(result.index, pd.DatetimeIndex):
            result.index = pd.to_datetime(result.index)

        # Sort by date
        result.sort_index(inplace=True)

        return result

    def _parse_date(self, date_str: str) -> date:
        """Parse date string to date object.

        Args:
            date_str: Date in YYYY-MM-DD format

        Returns:
            date object

        Raises:
            ValueError: If date format is invalid
        """
        return datetime.strptime(date_str, "%Y-%m-%d").date()

    def _normalize_ohlcv(self, df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
        """Normalize DataFrame to expected OHLCV MultiIndex format.

        Args:
            df: Raw DataFrame from MDMP
            ticker: Original ticker symbol for column naming

        Returns:
            DataFrame with MultiIndex columns (Ticker, Field) or None on error
        """
        if df.empty:
            return None

        # Expected OHLCV columns
        expected_cols = ["Open", "High", "Low", "Close", "Volume"]

        # Find available columns (case-insensitive matching)
        col_map: Dict[str, str] = {}
        df_cols_lower = {c.lower(): c for c in df.columns}

        for expected in expected_cols:
            if expected in df.columns:
                col_map[expected] = expected
            elif expected.lower() in df_cols_lower:
                col_map[expected] = df_cols_lower[expected.lower()]

        if "Close" not in col_map:
            logger.warning(f"No Close column found for {ticker}")
            return None

        available_cols = [col_map[c] for c in expected_cols if c in col_map]

        if not available_cols:
            logger.warning(f"No OHLCV columns found for {ticker}")
            return None

        # Extract data and rename columns
        ticker_df = df[available_cols].copy()
        ticker_df.columns = [c for c in expected_cols if c in col_map]

        # Create MultiIndex columns: (Ticker, Field)
        ticker_df.columns = pd.MultiIndex.from_product(
            [[ticker], ticker_df.columns], names=["Ticker", "Field"]
        )

        return ticker_df

    def get_failure_report(self) -> Dict[str, Any]:
        """Get a report of any fetch failures.

        Returns:
            Dictionary with failure information
        """
        # MDMP doesn't track failures the same way HybridDataSource does,
        # but we can provide a placeholder for interface compatibility
        return {
            "total_failures": 0,
            "provider_failures": {},
            "message": "Failure tracking not available for MDMP source",
        }


__all__ = ["MarketDataMultiProviderDataSource"]
