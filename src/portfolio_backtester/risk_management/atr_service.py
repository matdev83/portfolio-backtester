"""
Optimized ATR calculation service for risk management.

This service provides fast, cached ATR calculations shared between
stop loss and take profit handlers, eliminating code duplication
and leveraging the existing Numba optimizations.
"""

import pandas as pd
import numpy as np
import hashlib
from typing import Dict
from dataclasses import dataclass
import threading

from ..numba_optimized import atr_fast_fixed


@dataclass(frozen=True)
class ATRCacheKey:
    """Cache key for ATR calculations."""

    symbols: tuple
    period: str  # date as string
    lookback: int  # atr_length
    data_timestamp: str  # latest data timestamp for cache invalidation
    data_hash: str  # hash of actual data values for proper cache invalidation


class OptimizedATRService:
    """
    Fast, shared ATR calculation service using Numba optimization.

    Eliminates the 80+ lines of duplicated ATR code between stop loss
    and take profit handlers while providing significant performance improvements.
    """

    def __init__(self, cache_size: int = 1000):
        self._cache: Dict[ATRCacheKey, pd.Series] = {}
        self._cache_size = cache_size
        self._cache_lock = threading.RLock()

    def calculate_atr(
        self,
        asset_ohlc_history: pd.DataFrame,
        current_date: pd.Timestamp,
        atr_length: int = 14,
    ) -> pd.Series:
        """
        Calculate ATR for all assets using optimized Numba implementation.

        Args:
            asset_ohlc_history: OHLC data with MultiIndex columns (Ticker, Field)
            current_date: Current evaluation date
            atr_length: ATR window length

        Returns:
            Series with ATR values indexed by ticker
        """
        if asset_ohlc_history is None or asset_ohlc_history.empty:
            return pd.Series(dtype=float)

        # Extract symbols for cache key
        symbols = tuple(sorted(self._extract_tickers(asset_ohlc_history)))

        # Filter data up to current date for data timestamp check
        ohlc_data = asset_ohlc_history[asset_ohlc_history.index <= current_date]
        data_timestamp = str(ohlc_data.index.max()) if not ohlc_data.empty else "empty"

        # Create hash of the relevant data for proper cache invalidation
        data_hash = self._calculate_data_hash(ohlc_data, symbols, atr_length)

        # Create cache key based on (symbol, period, lookback) tuple
        cache_key = ATRCacheKey(
            symbols=symbols,
            period=str(current_date),
            lookback=atr_length,
            data_timestamp=data_timestamp,
            data_hash=data_hash,
        )

        # Check cache (thread-safe)
        with self._cache_lock:
            if cache_key in self._cache:
                return self._cache[cache_key].copy()

        # Check minimum data requirements
        if len(ohlc_data) < atr_length:
            tickers = self._extract_tickers(asset_ohlc_history)
            result = pd.Series(np.nan, index=tickers, dtype=float)
            self._cache_result(cache_key, result)
            return result

        # Calculate ATR using optimized method
        if isinstance(ohlc_data.columns, pd.MultiIndex) and "Field" in ohlc_data.columns.names:
            result = self._calculate_atr_multiindex(ohlc_data, atr_length, current_date)
        else:
            result = self._calculate_atr_simple(ohlc_data, atr_length, current_date)

        # Cache and return
        self._cache_result(cache_key, result)
        return result

    def _calculate_atr_multiindex(
        self, ohlc_data: pd.DataFrame, atr_length: int, current_date: pd.Timestamp
    ) -> pd.Series:
        """Calculate ATR for MultiIndex OHLC data using Numba optimization."""
        tickers = list(ohlc_data.columns.get_level_values("Ticker").unique())
        atr_results = {}

        for ticker in tickers:
            try:
                # Extract OHLC data for this ticker
                high = ohlc_data[(ticker, "High")]
                low = ohlc_data[(ticker, "Low")]
                close = ohlc_data[(ticker, "Close")]

                # Check data validity
                if (
                    len(high) < atr_length
                    or high.isna().all()
                    or low.isna().all()
                    or close.isna().all()
                ):
                    atr_results[ticker] = np.nan
                    continue

                # Use optimized Numba calculation
                atr_values = atr_fast_fixed(
                    high.values.astype(np.float64),
                    low.values.astype(np.float64),
                    close.values.astype(np.float64),
                    atr_length,
                )

                # Get ATR value for current date
                if current_date in high.index:
                    date_idx = high.index.get_loc(current_date)
                    atr_results[ticker] = atr_values[date_idx]
                else:
                    atr_results[ticker] = np.nan

            except (KeyError, IndexError, ValueError):
                atr_results[ticker] = np.nan

        return pd.Series(atr_results, dtype=float)

    def _calculate_atr_simple(
        self, ohlc_data: pd.DataFrame, atr_length: int, current_date: pd.Timestamp
    ) -> pd.Series:
        """Fallback ATR calculation for simple price data (not OHLC)."""
        # For non-OHLC data, use volatility-based approximation
        if current_date not in ohlc_data.index:
            return pd.Series(np.nan, index=ohlc_data.columns, dtype=float)

        returns = ohlc_data.pct_change(fill_method=None)
        rolling_std = returns.rolling(window=atr_length, min_periods=atr_length).std()

        if current_date in rolling_std.index:
            current_prices = ohlc_data.loc[current_date]
            std_today = rolling_std.loc[current_date]
            atr_values = current_prices * std_today
            return pd.Series(atr_values, dtype=float)
        else:
            return pd.Series(np.nan, index=ohlc_data.columns, dtype=float)

    def _extract_tickers(self, data: pd.DataFrame) -> list:
        """Extract ticker list from DataFrame columns."""
        if hasattr(data, "columns"):
            if isinstance(data.columns, pd.MultiIndex) and "Ticker" in data.columns.names:
                return list(data.columns.get_level_values("Ticker").unique())
            else:
                return list(data.columns)
        return []

    def _calculate_data_hash(self, ohlc_data: pd.DataFrame, symbols: tuple, atr_length: int) -> str:
        """Calculate a hash of the data relevant for ATR calculation.

        This ensures that cache hits only occur when the actual data content is identical,
        not just when the timestamps are the same.
        """
        if ohlc_data.empty:
            return "empty_data"

        try:
            # For MultiIndex OHLC data, hash the relevant OHLC values needed for ATR
            if isinstance(ohlc_data.columns, pd.MultiIndex) and "Field" in ohlc_data.columns.names:
                hash_data = []
                for symbol in symbols:
                    if len(ohlc_data) >= atr_length:
                        # Only hash the last `atr_length` rows that are needed for ATR calculation
                        relevant_data = ohlc_data.tail(atr_length)
                        try:
                            high_values = relevant_data[(symbol, "High")].values
                            low_values = relevant_data[(symbol, "Low")].values
                            close_values = relevant_data[(symbol, "Close")].values

                            # Convert to bytes for hashing, handling NaN values
                            high_bytes = np.nan_to_num(high_values, nan=-999.0).tobytes()
                            low_bytes = np.nan_to_num(low_values, nan=-999.0).tobytes()
                            close_bytes = np.nan_to_num(close_values, nan=-999.0).tobytes()

                            hash_data.extend([high_bytes, low_bytes, close_bytes])
                        except (KeyError, IndexError):
                            # If data is missing for this symbol, include a placeholder
                            hash_data.append(f"missing_{symbol}".encode())
            else:
                # For simple data, hash all values
                if len(ohlc_data) >= atr_length:
                    relevant_data = ohlc_data.tail(atr_length)
                    hash_data = [np.nan_to_num(relevant_data.values, nan=-999.0).tobytes()]
                else:
                    hash_data = ["insufficient_data".encode()]

            # Create combined hash
            hasher = hashlib.md5(usedforsecurity=False)
            for data_chunk in hash_data:
                hasher.update(data_chunk)

            return hasher.hexdigest()

        except Exception:
            # Fallback to a simple representation if hashing fails
            return f"fallback_{len(ohlc_data)}_{ohlc_data.shape}"

    def _cache_result(self, cache_key: ATRCacheKey, result: pd.Series):
        """Cache the ATR calculation result with size management."""
        with self._cache_lock:
            # Simple cache size management
            if len(self._cache) >= self._cache_size:
                # Remove oldest entries (simple FIFO)
                keys_to_remove = list(self._cache.keys())[: len(self._cache) // 2]
                for key in keys_to_remove:
                    del self._cache[key]

            self._cache[cache_key] = result.copy()

    def clear_cache(self):
        """Clear the ATR cache."""
        with self._cache_lock:
            self._cache.clear()

    def cache_info(self) -> Dict:
        """Get cache statistics."""
        with self._cache_lock:
            return {"size": len(self._cache), "max_size": self._cache_size}


# Global shared ATR service instance
_atr_service = OptimizedATRService(cache_size=1000)


def get_atr_service() -> OptimizedATRService:
    """Get the global ATR service instance."""
    return _atr_service


def calculate_atr_fast(
    asset_ohlc_history: pd.DataFrame, current_date: pd.Timestamp, atr_length: int = 14
) -> pd.Series:
    """
    Fast ATR calculation using shared service.

    This is the primary interface for ATR calculations throughout
    the risk management system.
    """
    return _atr_service.calculate_atr(asset_ohlc_history, current_date, atr_length)
