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
import hashlib
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class DataPreprocessingCache:
    """
    Cache for expensive data preprocessing operations like return calculations.
    
    This cache helps avoid redundant calculations during optimization by storing
    computed results and reusing them when the same data is requested again.
    """
    
    def __init__(self, max_cache_size: int = 100):
        """
        Initialize the data cache.
        
        Args:
            max_cache_size: Maximum number of cached items before cleanup
        """
        self.max_cache_size = max_cache_size
        self.cached_returns: Dict[str, pd.DataFrame] = {}
        self.cached_window_returns: Dict[str, pd.DataFrame] = {}  # New: window-specific returns
        self.data_hashes: Dict[str, str] = {}
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'window_hits': 0,
            'window_misses': 0
        }
    
    def _get_data_hash(self, data: pd.DataFrame, identifier: str) -> str:
        """
        Generate a hash for DataFrame to use as cache key.
        
        Args:
            data: DataFrame to hash
            identifier: Additional identifier for uniqueness
            
        Returns:
            Hash string for the data
        """
        # Create a hash based on data shape, index, columns, and sample values
        hash_input = f"{identifier}_{data.shape}_{data.index.min()}_{data.index.max()}"
        hash_input += f"_{list(data.columns)}_{data.iloc[0, 0] if not data.empty else 'empty'}"
        
        if len(data) > 1:
            hash_input += f"_{data.iloc[-1, 0]}"
        
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _get_window_hash(self, data: pd.DataFrame, window_start: pd.Timestamp, window_end: pd.Timestamp) -> str:
        """
        Generate a hash for window-specific data.
        
        Args:
            data: DataFrame to hash
            window_start: Start of the window
            window_end: End of the window
            
        Returns:
            Hash string for the windowed data
        """
        hash_input = f"window_{window_start}_{window_end}_{data.shape}"
        hash_input += f"_{list(data.columns)}_{data.iloc[0, 0] if not data.empty else 'empty'}"
        
        if len(data) > 1:
            hash_input += f"_{data.iloc[-1, 0]}"
        
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _cleanup_cache_if_needed(self):
        """Remove oldest entries if cache exceeds max size."""
        total_items = len(self.cached_returns) + len(self.cached_window_returns)
        if total_items > self.max_cache_size:
            # Remove oldest entries (simplified - remove half)
            items_to_remove = total_items - self.max_cache_size // 2
            
            # Remove from regular cache first
            keys_to_remove = list(self.cached_returns.keys())[:items_to_remove // 2]
            for key in keys_to_remove:
                del self.cached_returns[key]
            
            # Remove from window cache
            remaining_to_remove = items_to_remove - len(keys_to_remove)
            if remaining_to_remove > 0:
                window_keys_to_remove = list(self.cached_window_returns.keys())[:remaining_to_remove]
                for key in window_keys_to_remove:
                    del self.cached_window_returns[key]
    
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
        
        if data_hash in self.cached_returns:
            self._cache_stats['hits'] += 1
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Cache HIT for returns: {identifier}")
            return self.cached_returns[data_hash]
        
        # Cache miss - compute returns
        self._cache_stats['misses'] += 1
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Cache MISS for returns: {identifier} - computing...")
        
        returns = data.pct_change(fill_method=None).fillna(0)
        
        # Store in cache
        self.cached_returns[data_hash] = returns
        self.data_hashes[f"returns_{identifier}"] = data_hash
        
        self._cleanup_cache_if_needed()
        return returns
    
    def get_cached_window_returns(
        self, 
        data: pd.DataFrame, 
        window_start: pd.Timestamp, 
        window_end: pd.Timestamp
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
        
        if window_hash in self.cached_window_returns:
            self._cache_stats['window_hits'] += 1
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Cache HIT for window returns: {window_start} to {window_end}")
            return self.cached_window_returns[window_hash]
        
        # Cache miss - compute returns
        self._cache_stats['window_misses'] += 1
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Cache MISS for window returns: {window_start} to {window_end} - computing...")
        
        returns = data.pct_change(fill_method=None).fillna(0)
        
        # Store in cache
        self.cached_window_returns[window_hash] = returns
        
        self._cleanup_cache_if_needed()
        return returns
    
    def precompute_window_returns(
        self, 
        daily_data: pd.DataFrame, 
        windows: list
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
            
            # Extract close prices for return calculation
            if isinstance(window_data.columns, pd.MultiIndex) and 'Close' in window_data.columns.get_level_values(1):
                close_prices = window_data.xs('Close', level='Field', axis=1)
            elif not isinstance(window_data.columns, pd.MultiIndex):
                close_prices = window_data
            else:
                # Try to find Close prices in less structured MultiIndex
                try:
                    if 'Close' in window_data.columns.get_level_values(-1):
                        close_prices = window_data.xs('Close', level=-1, axis=1)
                    else:
                        if logger.isEnabledFor(logging.WARNING):
                            logger.warning(f"Could not extract Close prices for window {window_idx}")
                        continue
                except Exception as e:
                    if logger.isEnabledFor(logging.WARNING):
                        logger.warning(f"Error extracting Close prices for window {window_idx}: {e}")
                    continue
            
            # Cache the returns for this window
            window_returns_df = self.get_cached_window_returns(close_prices, tr_start, te_end)
            window_hash = self._get_window_hash(close_prices, tr_start, te_end)
            window_returns[window_hash] = window_returns_df
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Pre-computed returns for {len(window_returns)} windows")
        
        return window_returns
    
    def get_window_returns_by_dates(
        self, 
        daily_data: pd.DataFrame,
        window_start: pd.Timestamp, 
        window_end: pd.Timestamp
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
        window_data = daily_data.loc[window_start:window_end]
        
        if window_data.empty:
            return None
        
        # Extract close prices
        if isinstance(window_data.columns, pd.MultiIndex) and 'Close' in window_data.columns.get_level_values(1):
            close_prices = window_data.xs('Close', level='Field', axis=1)
        elif not isinstance(window_data.columns, pd.MultiIndex):
            close_prices = window_data
        else:
            try:
                if 'Close' in window_data.columns.get_level_values(-1):
                    close_prices = window_data.xs('Close', level=-1, axis=1)
                else:
                    return None
            except Exception:
                return None
        
        window_hash = self._get_window_hash(close_prices, window_start, window_end)
        return self.cached_window_returns.get(window_hash)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self._cache_stats['hits'] + self._cache_stats['misses']
        window_requests = self._cache_stats['window_hits'] + self._cache_stats['window_misses']
        
        return {
            'total_cached_items': len(self.cached_returns) + len(self.cached_window_returns),
            'regular_cache_items': len(self.cached_returns),
            'window_cache_items': len(self.cached_window_returns),
            'regular_hit_rate': self._cache_stats['hits'] / max(1, total_requests),
            'window_hit_rate': self._cache_stats['window_hits'] / max(1, window_requests),
            'total_requests': total_requests,
            'window_requests': window_requests,
            **self._cache_stats
        }
    
    def clear_cache(self):
        """Clear all cached data."""
        self.cached_returns.clear()
        self.cached_window_returns.clear()
        self.data_hashes.clear()
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'window_hits': 0,
            'window_misses': 0
        }
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Data cache cleared")


# Global cache instance
_global_cache: Optional[DataPreprocessingCache] = None


def get_global_cache() -> DataPreprocessingCache:
    """Get or create the global data preprocessing cache."""
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
    return get_global_cache().get_cache_stats()