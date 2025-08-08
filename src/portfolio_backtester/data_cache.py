"""
Data preprocessing cache for performance optimization.

This module provides caching mechanisms for expensive data operations that are
repeatedly performed during optimization, such as:
- Return calculations (pct_change)
- Data slicing by date
- Rolling statistics preprocessing
- Index mapping for fast lookups

The cache significantly reduces redundant calculations during WFO optimization.
"""

import pandas as pd
from typing import Dict, Optional, Any
import logging

from .cache.data_hasher import DataHasher
from .cache.cache_manager import CacheManager
from .cache.cache_statistics import CacheStatistics
from .cache.window_processor import WindowProcessor

logger = logging.getLogger(__name__)


class DataPreprocessingCache:
    """
    Cache for expensive data preprocessing operations like return calculations.

    This cache helps avoid redundant calculations during optimization by storing
    computed results and reusing them when the same data is requested again.

    This class now acts as a facade over specialized cache components, maintaining
    backward compatibility while improving internal structure.
    """

    def __init__(self, max_cache_size: int = 100):
        """
        Initialize the data cache.

        Args:
            max_cache_size: Maximum number of cached items before cleanup
        """
        self.max_cache_size = max_cache_size

        # Initialize component classes
        self._hasher = DataHasher()
        self._cache_manager = CacheManager(max_cache_size)
        self._statistics = CacheStatistics()
        self._window_processor = WindowProcessor()

        # Maintain backward compatibility properties
        self.cached_returns = self._cache_manager.cached_returns
        self.cached_window_returns = self._cache_manager.cached_window_returns
        self.data_hashes = self._cache_manager.data_hashes

    def _get_data_hash(self, data: pd.DataFrame, identifier: str) -> str:
        """
        Generate a hash for DataFrame to use as cache key.

        Args:
            data: DataFrame to hash
            identifier: Additional identifier for uniqueness

        Returns:
            Hash string for the data
        """
        return self._hasher.get_data_hash(data, identifier)

    def _get_window_hash(
        self, data: pd.DataFrame, window_start: pd.Timestamp, window_end: pd.Timestamp
    ) -> str:
        """
        Generate a hash for window-specific data.

        Args:
            data: DataFrame to hash
            window_start: Start of the window
            window_end: End of the window

        Returns:
            Hash string for the windowed data
        """
        return self._hasher.get_window_hash(data, window_start, window_end)

    def _cleanup_cache_if_needed(self):
        """Remove oldest entries if cache exceeds max size."""
        self._cache_manager.cleanup_cache_if_needed()

    def get_cached_returns(self, data: pd.DataFrame, identifier: str = "default") -> pd.DataFrame:
        """
        Get cached return calculations or compute and cache them.

        Args:
            data: Price data DataFrame
            identifier: Unique identifier for this dataset

        Returns:
            DataFrame of returns (pct_change)
        """
        data_hash = self._get_data_hash(data, f"returns_{identifier}")

        # Check cache
        cached_result = self._cache_manager.get_cached_item(data_hash, is_window=False)
        if cached_result is not None:
            self._statistics.log_cache_operation(
                f"returns: {identifier}", data_hash, hit=True, is_window=False
            )
            return cached_result

        # Cache miss - compute returns
        self._statistics.log_cache_operation(
            f"returns: {identifier}", data_hash, hit=False, is_window=False
        )

        returns = self._window_processor.compute_returns(data)

        # Store in cache
        self._cache_manager.store_cached_item(
            data_hash, returns, is_window=False, identifier=f"returns_{identifier}"
        )

        return returns

    def get_cached_window_returns(
        self, data: pd.DataFrame, window_start: pd.Timestamp, window_end: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Get cached return calculations for a specific window or compute and cache them.

        Args:
            data: Price data DataFrame for the window
            window_start: Start of the window
            window_end: End of the window

        Returns:
            DataFrame of returns (pct_change) for the window
        """
        window_hash = self._get_window_hash(data, window_start, window_end)

        # Check cache
        cached_result = self._cache_manager.get_cached_item(window_hash, is_window=True)
        if cached_result is not None:
            self._statistics.log_cache_operation(
                f"window returns: {window_start} to {window_end}",
                window_hash,
                hit=True,
                is_window=True,
            )
            return cached_result

        # Cache miss - compute returns
        self._statistics.log_cache_operation(
            f"window returns: {window_start} to {window_end}",
            window_hash,
            hit=False,
            is_window=True,
        )

        returns = self._window_processor.compute_returns(data)

        # Store in cache
        self._cache_manager.store_cached_item(window_hash, returns, is_window=True)

        return returns

    def precompute_window_returns(
        self, daily_data: pd.DataFrame, windows: list
    ) -> Dict[str, pd.DataFrame]:
        """
        Pre-compute returns for all windows and cache them.

        Args:
            daily_data: Full daily price data
            windows: List of (tr_start, tr_end, te_start, te_end) tuples

        Returns:
            Dictionary mapping window hashes to return DataFrames
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Pre-computing returns for {len(windows)} windows")

        window_returns = {}

        for window_idx, (tr_start, tr_end, te_start, te_end) in enumerate(windows):
            # Get the full window data (training + test)
            window_data = daily_data.loc[tr_start:te_end]

            if window_data.empty:
                continue

            # Extract close prices using the window processor
            close_prices_df = self._window_processor.extract_close_prices(window_data)
            if close_prices_df is None:
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(f"Could not extract Close prices for window {window_idx}")
                continue

            # Cache the returns for this window
            window_returns_df = self.get_cached_window_returns(close_prices_df, tr_start, te_end)
            window_hash = self._get_window_hash(close_prices_df, tr_start, te_end)
            window_returns[window_hash] = window_returns_df

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Pre-computed returns for {len(window_returns)} windows")

        return window_returns

    def get_window_returns_by_dates(
        self, daily_data: pd.DataFrame, window_start: pd.Timestamp, window_end: pd.Timestamp
    ) -> Optional[pd.DataFrame]:
        """
        Get cached window returns by date range.

        Args:
            daily_data: Full daily price data
            window_start: Start of the window
            window_end: End of the window

        Returns:
            DataFrame of returns for the window, or None if not cached
        """
        # Use window processor to get window data and extract close prices
        close_prices_df = self._window_processor.get_window_data_by_dates(
            daily_data, window_start, window_end
        )
        if close_prices_df is None:
            return None

        window_hash = self._get_window_hash(close_prices_df, window_start, window_end)
        return self._cache_manager.get_cached_item(window_hash, is_window=True)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        cache_size_info = self._cache_manager.get_cache_size_info()
        return self._statistics.get_statistics(cache_size_info)

    def clear_cache(self):
        """Clear all cached data."""
        self._cache_manager.clear_cache()
        self._statistics.reset_statistics()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Data cache cleared")


# Global cache instance
_global_cache: Optional[DataPreprocessingCache] = None


def get_global_cache() -> DataPreprocessingCache:
    """Get the global data preprocessing cache instance.

    Returns:
        DataPreprocessingCache: The singleton cache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = DataPreprocessingCache()
    return _global_cache


def clear_global_cache():
    """Clear the global data preprocessing cache."""
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear_cache()


def get_cache_stats() -> Dict[str, Any]:
    """Get statistics from the global cache."""
    # Use DIP interface instead of legacy get_global_cache
    from .interfaces import create_cache_manager

    cache_manager = create_cache_manager()
    return cache_manager.get_cache_stats()
