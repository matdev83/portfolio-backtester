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
import numpy as np
import hashlib
from typing import Dict, Optional, Tuple, Any
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class DataPreprocessingCache:
    """
    Cache for expensive data preprocessing operations.
    
    This cache stores computed results for:
    - Return calculations
    - Date-based data slicing
    - Index mappings
    - Rolling statistics
    """
    
    def __init__(self, max_cache_size_mb: int = 500):
        """
        Initialize the data preprocessing cache.
        
        Args:
            max_cache_size_mb: Maximum cache size in megabytes
        """
        self.max_cache_size_mb = max_cache_size_mb
        self.cached_returns: Dict[str, pd.DataFrame] = {}
        self.cached_slices: Dict[str, pd.DataFrame] = {}
        self.date_position_maps: Dict[str, Dict[pd.Timestamp, int]] = {}
        self.data_hashes: Dict[str, str] = {}
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'size_mb': 0.0
        }
        
        if logger.isEnabledFor(logging.DEBUG):

        
            logger.debug(f"Initialized DataPreprocessingCache with max size: {max_cache_size_mb}MB")
    
    def _get_data_hash(self, data: pd.DataFrame, identifier: str = "") -> str:
        """Generate a hash for DataFrame to use as cache key."""
        # Handle empty DataFrame
        if len(data) == 0:
            hash_input = f"{identifier}_empty_{data.shape}"
        else:
            # Use shape, index range, and sample values for hash
            hash_input = f"{identifier}_{data.shape}_{data.index[0]}_{data.index[-1]}"
            # Sample a few values for hash uniqueness
            sample_values = data.iloc[::max(1, len(data)//10)].values.flatten()[:10]
            hash_input += f"_{hash(tuple(sample_values))}"
        
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def _estimate_size_mb(self, obj: Any) -> float:
        """Estimate memory size of object in MB."""
        if isinstance(obj, pd.DataFrame):
            return obj.memory_usage(deep=True).sum() / (1024 * 1024)
        elif isinstance(obj, pd.Series):
            return obj.memory_usage(deep=True) / (1024 * 1024)
        elif isinstance(obj, dict):
            return sum(self._estimate_size_mb(v) for v in obj.values())
        else:
            return 0.001  # Minimal size for other objects
    
    def _cleanup_cache_if_needed(self):
        """Remove oldest entries if cache size exceeds limit."""
        current_size = sum(
            self._estimate_size_mb(cache) 
            for cache in [self.cached_returns, self.cached_slices, self.date_position_maps]
        )
        
        if current_size > self.max_cache_size_mb:
            if logger.isEnabledFor(logging.WARNING):

                logger.warning(f"Cache size ({current_size:.1f}MB) exceeds limit ({self.max_cache_size_mb}MB). Clearing cache.")
            self.clear_cache()
    
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
    
    def get_date_position_map(self, data: pd.DataFrame, identifier: str = "default") -> Dict[pd.Timestamp, int]:
        """
        Get cached date-to-position mapping for fast data slicing.
        
        Args:
            data: DataFrame with DatetimeIndex
            identifier: Unique identifier for this dataset
            
        Returns:
            Dictionary mapping dates to integer positions
        """
        data_hash = self._get_data_hash(data, f"positions_{identifier}")
        
        if data_hash in self.date_position_maps:
            self._cache_stats['hits'] += 1
            if logger.isEnabledFor(logging.DEBUG):

                logger.debug(f"Cache HIT for date positions: {identifier}")
            return self.date_position_maps[data_hash]
        
        # Cache miss - compute position mapping
        self._cache_stats['misses'] += 1
        if logger.isEnabledFor(logging.DEBUG):

            logger.debug(f"Cache MISS for date positions: {identifier} - computing...")
        
        position_map = {date: idx for idx, date in enumerate(data.index)}
        
        # Store in cache
        self.date_position_maps[data_hash] = position_map
        self.data_hashes[f"positions_{identifier}"] = data_hash
        
        return position_map
    
    def get_data_slice_fast(self, data: pd.DataFrame, end_date: pd.Timestamp, 
                           position_map: Optional[Dict[pd.Timestamp, int]] = None,
                           identifier: str = "default") -> pd.DataFrame:
        """
        Get data slice up to end_date using fast integer indexing.
        
        Args:
            data: Source DataFrame
            end_date: End date for slice
            position_map: Pre-computed position mapping (optional)
            identifier: Unique identifier for caching
            
        Returns:
            Sliced DataFrame
        """
        # Create cache key
        slice_key = f"{identifier}_{end_date}_{data.shape[0]}"
        slice_hash = hashlib.md5(slice_key.encode()).hexdigest()[:16]
        
        if slice_hash in self.cached_slices:
            self._cache_stats['hits'] += 1
            if logger.isEnabledFor(logging.DEBUG):

                logger.debug(f"Cache HIT for data slice: {slice_key}")
            return self.cached_slices[slice_hash]
        
        # Cache miss - compute slice
        self._cache_stats['misses'] += 1
        
        if position_map is None:
            position_map = self.get_date_position_map(data, identifier)
        
        if end_date in position_map:
            end_idx = position_map[end_date]
            sliced_data = data.iloc[:end_idx + 1]
        else:
            # Fallback to boolean indexing if date not found
            sliced_data = data[data.index <= end_date]
        
        # Store in cache (only if reasonable size)
        if self._estimate_size_mb(sliced_data) < 50:  # Don't cache very large slices
            self.cached_slices[slice_hash] = sliced_data
            self._cleanup_cache_if_needed()
        
        return sliced_data
    
    @lru_cache(maxsize=128)
    def get_rolling_window_indices(self, data_length: int, window_size: int) -> np.ndarray:
        """
        Get pre-computed indices for rolling window operations.
        
        Args:
            data_length: Length of the data series
            window_size: Size of rolling window
            
        Returns:
            Array of start indices for each rolling window
        """
        return np.arange(max(0, data_length - window_size + 1))
    
    def clear_cache(self):
        """Clear all cached data."""
        self.cached_returns.clear()
        self.cached_slices.clear()
        self.date_position_maps.clear()
        self.data_hashes.clear()
        
        # Reset LRU cache
        self.get_rolling_window_indices.cache_clear()
        
        self._cache_stats = {'hits': 0, 'misses': 0, 'size_mb': 0.0}
        logger.debug("Data preprocessing cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self._cache_stats['hits'] + self._cache_stats['misses']
        hit_rate = self._cache_stats['hits'] / total_requests if total_requests > 0 else 0
        
        current_size = sum(
            self._estimate_size_mb(cache) 
            for cache in [self.cached_returns, self.cached_slices, self.date_position_maps]
        )
        
        return {
            'hits': self._cache_stats['hits'],
            'misses': self._cache_stats['misses'],
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'current_size_mb': current_size,
            'max_size_mb': self.max_cache_size_mb,
            'cached_returns_count': len(self.cached_returns),
            'cached_slices_count': len(self.cached_slices),
            'cached_position_maps_count': len(self.date_position_maps)
        }
    
    def log_cache_stats(self):
        """Log current cache statistics."""
        stats = self.get_cache_stats()
        logger.info(
            f"Cache Stats - Hit Rate: {stats['hit_rate']:.2%} "
            f"({stats['hits']}/{stats['total_requests']}) | "
            f"Size: {stats['current_size_mb']:.1f}MB/{stats['max_size_mb']}MB | "
            f"Entries: Returns={stats['cached_returns_count']}, "
            f"Slices={stats['cached_slices_count']}, "
            f"Maps={stats['cached_position_maps_count']}"
        )


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