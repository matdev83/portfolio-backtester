"""Market Data Multi-Provider data source adapter.

This module provides a data source implementation that wraps the
market-data-multi-provider package, enabling transparent integration
with portfolio-backtester's existing data source architecture.

Features:
- Compatible with BaseDataSource interface
- Automatic symbol mapping (local ↔ canonical format)
- Multi-provider fallback (stooq → yfinance → tradingview)
- Reads OHLCV via MDMP fetch/cache APIs only (no PB-side canonical parquet I/O)
- Coverage validation with trading calendars
"""

import logging
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import pandas as pd

from .base_data_source import BaseDataSource
from .mdmp_facade import MarketDataClient
from .symbol_mapper import from_canonical_id, to_canonical_id

logger = logging.getLogger(__name__)


def bare_ticker_from_canonical_id(canonical_id: str) -> str:
    """Return the bare symbol (e.g. ``EWI``) from ``EXCHANGE:SYMBOL``."""
    s = canonical_id.strip()
    return s.split(":", 1)[1].upper() if ":" in s else s.upper()


def align_mdmp_results_to_requested(
    canonical_ids: List[str],
    results: Dict[str, pd.DataFrame],
) -> Dict[str, Optional[pd.DataFrame]]:
    """Map MDMP result frames onto each *requested* canonical id.

    MDMP may return dict keys that match on-disk layout (e.g. ``NYSE:EWI``) while
    portfolio-backtester requested another alias (e.g. ``AMEX:EWI`` from
    :func:`to_canonical_id`). Keys are matched by bare ticker when the exact
    requested key is absent, so downstream ``symbol_map`` lookups stay correct.

    Args:
        canonical_ids: Canonical ids passed to ``fetch_many`` / ``fetch_many_cached_only``.
        results: Raw mapping returned by MDMP.

    Returns:
        For each *first* occurrence of a canonical id in ``canonical_ids``, the
        best-matching DataFrame or ``None`` if no row exists for that request token.
    """
    pool: Dict[str, pd.DataFrame] = dict(results)
    out: Dict[str, Optional[pd.DataFrame]] = {}

    for cid in canonical_ids:
        if cid in out:
            continue
        if cid in pool:
            out[cid] = pool.pop(cid)
            continue
        bare = bare_ticker_from_canonical_id(cid)
        match_key: Optional[str] = None
        for k in list(pool.keys()):
            if bare_ticker_from_canonical_id(k) == bare:
                match_key = k
                break
        if match_key is not None:
            df = pool.pop(match_key)
            out[cid] = df
            if match_key != cid and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "MDMP aligned result key %r onto requested canonical %r (same bare ticker)",
                    match_key,
                    cid,
                )
        else:
            out[cid] = None

    return out


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
        preferred_provider: Optional[str] = None,
        allow_fallbacks: bool = True,
        max_workers: Optional[int] = None,
        cache_only: bool = False,
        cache_max_age_seconds: Optional[int] = 14400,
    ) -> None:
        """Initialize the data source.

        Args:
            data_dir: Directory for data storage (uses MDMP default if None)
            cache_expiry_hours: Ignored - MDMP handles caching internally.
                Kept for interface compatibility.
            min_coverage_ratio: Minimum coverage ratio for NYSE calendar validation.
                Set to None to disable coverage checks (default).
            preferred_provider: Preferred MDMP provider ID (e.g., "stooq").
            allow_fallbacks: Whether MDMP may fall back to other providers.
            max_workers: Max worker threads used by MDMP fetch_many (None = MDMP default).
            cache_only: If true, only use cached MDMP data and skip downloads.
            cache_max_age_seconds: Max cache age in seconds for MDMP cache reads.
                Use ``None`` to defer to MDMP defaults (e.g. no parquet age filter).
        """
        if MarketDataClient is None:
            raise ImportError(
                "market-data-multi-provider is not installed. "
                "Install with: pip install -e ../market-data-multi-provider"
            )

        self.client = MarketDataClient(data_dir=data_dir)
        self.data_dir = Path(data_dir) if data_dir else None
        self._cache_expiry_hours = cache_expiry_hours  # Stored for interface compat
        self._min_coverage_ratio = min_coverage_ratio
        self._preferred_provider = preferred_provider
        self._allow_fallbacks = allow_fallbacks
        self._max_workers = max_workers
        self._cache_only = cache_only
        self._cache_max_age_seconds = cache_max_age_seconds

        logger.info(
            "MarketDataMultiProviderDataSource initialized (reproducibility): "
            "preferred_provider=%r, allow_fallbacks=%s, cache_only=%s, "
            "cache_max_age_seconds=%s, data_dir=%s",
            self._preferred_provider,
            bool(self._allow_fallbacks),
            bool(self._cache_only),
            self._cache_max_age_seconds,
            str(self.data_dir) if self.data_dir is not None else "MDMP default",
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
        # Filter out invalid empty tickers
        tickers = [t for t in tickers if t and t.strip()]

        # One row per canonical id (e.g. global benchmark ``AMEX:SPY`` vs scenario ``SPY``).
        # Otherwise fetch_many returns one frame while len(tickers) counts duplicates and
        # logs misleading "N-1/N successful".
        seen_canonical: set[str] = set()
        deduped: List[str] = []
        for raw in tickers:
            cid = to_canonical_id(raw)
            if cid in seen_canonical:
                continue
            seen_canonical.add(cid)
            deduped.append(raw)
        tickers = deduped

        if not tickers:
            logger.warning("No valid tickers provided after filtering empty strings")
            return pd.DataFrame()

        logger.info(f"Fetching {len(tickers)} tickers from {start_date} to {end_date} via MDMP")
        logger.info(
            "MDMP fetch boundary (reproducibility): preferred_provider=%r, "
            "allow_fallbacks=%s, cache_only=%s, cache_max_age_seconds=%s",
            self._preferred_provider,
            bool(self._allow_fallbacks),
            bool(self._cache_only),
            self._cache_max_age_seconds,
        )

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
            if self._cache_only:
                logger.info("MDMP cache-only enabled: skipping provider downloads.")
                results = self._fetch_cached_many(canonical_ids, start, end)
            else:
                fetch_kw: Dict[str, Any] = dict(
                    start=start,
                    end=end,
                    preferred_provider=cast(Any, self._preferred_provider),
                    allow_fallbacks=bool(self._allow_fallbacks),
                    use_cache=True,
                    max_workers=self._max_workers,
                    use_canonical=True,
                    min_coverage_ratio=self._min_coverage_ratio,
                )
                if self._cache_max_age_seconds is not None:
                    fetch_kw["cache_max_age_seconds"] = int(self._cache_max_age_seconds)
                results = self.client.fetch_many(canonical_ids, **fetch_kw)
        except Exception as e:
            logger.error(f"MDMP fetch_many failed: {e}")
            return pd.DataFrame()

        aligned = align_mdmp_results_to_requested(canonical_ids, results)

        # Process results into MultiIndex format
        all_ticker_data: List[pd.DataFrame] = []
        successful = 0
        failed = 0
        failed_tickers: List[str] = []
        processed_canonical: set[str] = set()

        for canonical_id in canonical_ids:
            if canonical_id in processed_canonical:
                continue
            processed_canonical.add(canonical_id)

            df = aligned.get(canonical_id)
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

        unique_cids = list(dict.fromkeys(canonical_ids))
        raw_missing_keys = [cid for cid in unique_cids if cid not in results]
        recovered = len(
            [
                cid
                for cid in raw_missing_keys
                for _df in [aligned.get(cid)]
                if _df is not None and not _df.empty
            ]
        )
        if recovered > 0:
            logger.info(
                "MDMP bare-ticker alignment recovered %s series whose dict keys "
                "differed from the requested canonical id (e.g. disk NYSE:* vs request AMEX:*).",
                recovered,
            )
        missing_after_align = [
            cid for cid in unique_cids for _a in [aligned.get(cid)] if _a is None or _a.empty
        ]
        if missing_after_align:
            missing_labels = [
                symbol_map.get(cid, from_canonical_id(cid)) for cid in missing_after_align
            ]
            preview = ", ".join(missing_labels[:20])
            suffix = " ..." if len(missing_labels) > 20 else ""
            logger.warning(
                "MDMP omitted %s/%s requested symbols after alignment (no usable frame; "
                "typical for cache_only when those series are not on disk): [%s%s]",
                len(missing_after_align),
                len(unique_cids),
                preview,
                suffix,
            )

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

    def _fetch_cached_many(
        self, canonical_ids: List[str], start: date, end: date
    ) -> Dict[str, pd.DataFrame]:
        """Load cached canonical data from MDMP without downloading."""
        try:
            fetch_kw: Dict[str, Any] = dict(
                start=start,
                end=end,
                cache_max_age_seconds=self._cache_max_age_seconds,
                max_workers=self._max_workers,
            )
            results = self.client.fetch_many_cached_only(canonical_ids, **fetch_kw)
        except Exception as e:
            logger.error("MDMP fetch_many_cached_only failed: %s", e)
            return {}

        return dict(results)

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


__all__ = [
    "MarketDataMultiProviderDataSource",
    "align_mdmp_results_to_requested",
    "bare_ticker_from_canonical_id",
]
