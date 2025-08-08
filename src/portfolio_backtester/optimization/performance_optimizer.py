"""
Performance optimization utilities for the optimization system.

This module provides performance enhancements for the refactored architecture,
including memory optimization, parallel processing improvements, and caching.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from functools import lru_cache
import gc
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Optional dependency for memory monitoring
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """Optimizes memory usage during optimization runs."""

    def __init__(self, memory_threshold_gb: float = 8.0):
        """
        Initialize memory optimizer.

        Args:
            memory_threshold_gb: Memory threshold in GB to trigger cleanup
        """
        self.memory_threshold_gb = memory_threshold_gb
        self.memory_threshold_bytes = memory_threshold_gb * 1024**3

    def check_memory_usage(self) -> Dict[str, float]:
        """
        Check current memory usage.

        Returns:
            Dictionary with memory statistics
        """
        if not PSUTIL_AVAILABLE:
            return {"rss_gb": 0.0, "vms_gb": 0.0, "percent": 0.0, "available_gb": 0.0}

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        return {
            "rss_gb": memory_info.rss / 1024**3,
            "vms_gb": memory_info.vms / 1024**3,
            "percent": process.memory_percent(),
            "available_gb": psutil.virtual_memory().available / 1024**3,
        }

    def should_cleanup(self) -> bool:
        """
        Check if memory cleanup is needed.

        Returns:
            True if cleanup should be performed
        """
        memory_stats = self.check_memory_usage()
        return memory_stats["rss_gb"] > self.memory_threshold_gb

    def cleanup_memory(self) -> Dict[str, float]:
        """
        Perform memory cleanup.

        Returns:
            Memory statistics before and after cleanup
        """
        before = self.check_memory_usage()

        # Force garbage collection
        gc.collect()

        # Clear any cached data
        self.clear_caches()

        after = self.check_memory_usage()

        logger.info(
            f"Memory cleanup: {before['rss_gb']:.2f}GB -> {after['rss_gb']:.2f}GB "
            f"(freed {before['rss_gb'] - after['rss_gb']:.2f}GB)"
        )

        # Return memory usage in GB as floats (before/after snapshot)
        return {"before": float(before["rss_gb"]), "after": float(after["rss_gb"])}

    def clear_caches(self):
        """Clear LRU caches and other cached data."""
        # Clear function caches
        for obj in gc.get_objects():
            if hasattr(obj, "cache_clear") and callable(obj.cache_clear):
                try:
                    obj.cache_clear()
                except Exception:
                    pass


class DataFrameOptimizer:
    """Optimizes DataFrame memory usage and operations."""

    @staticmethod
    def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame dtypes to reduce memory usage.

        Args:
            df: DataFrame to optimize

        Returns:
            Optimized DataFrame
        """
        df_optimized = df.copy()

        for col in df_optimized.columns:
            col_type = df_optimized[col].dtype

            if col_type == "object":
                # Try to convert to category if low cardinality
                if df_optimized[col].nunique() / len(df_optimized) < 0.5:
                    df_optimized[col] = df_optimized[col].astype("category")

            elif col_type == "float64":
                # Downcast to float32 if possible
                col_min = df_optimized[col].min()
                col_max = df_optimized[col].max()

                if col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:
                    df_optimized[col] = df_optimized[col].astype(np.float32)

            elif col_type == "int64":
                # Downcast to smaller int types if possible
                col_min = df_optimized[col].min()
                col_max = df_optimized[col].max()

                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                    df_optimized[col] = df_optimized[col].astype(np.int8)
                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                    df_optimized[col] = df_optimized[col].astype(np.int16)
                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                    df_optimized[col] = df_optimized[col].astype(np.int32)

        return df_optimized

    @staticmethod
    def get_memory_usage(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get detailed memory usage of DataFrame.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with memory usage statistics
        """
        memory_usage = df.memory_usage(deep=True)

        return {
            "total_mb": memory_usage.sum() / 1024**2,
            "index_mb": memory_usage.iloc[0] / 1024**2,
            "columns_mb": memory_usage.iloc[1:].sum() / 1024**2,
            "per_column_mb": {  # detail only; not part of primary float metrics dict
                col: memory_usage[col] / 1024**2 for col in df.columns
            },
        }


class ParallelOptimizer:
    """Optimizes parallel processing for evaluation tasks."""

    def __init__(self, n_jobs: int = -1, chunk_size: Optional[int] = None):
        """
        Initialize parallel optimizer.

        Args:
            n_jobs: Number of parallel jobs (-1 for all cores)
            chunk_size: Size of chunks for parallel processing
        """
        cpu_count = os.cpu_count()
        if cpu_count is None:
            cpu_count = 1

        if n_jobs == -1:
            n_jobs = cpu_count

        self.n_jobs = min(n_jobs, cpu_count)
        if chunk_size is None:
            self.chunk_size: int = max(1, self.n_jobs * 2)
        else:
            self.chunk_size = chunk_size

        # Persistent ThreadPoolExecutor reused across calls
        self._executor = ThreadPoolExecutor(max_workers=self.n_jobs)
        import atexit

        atexit.register(self._executor.shutdown, wait=True)
        # Thread-local storage for worker state
        self._local = threading.local()

    def get_optimal_chunk_size(self, total_items: int) -> int:
        """
        Calculate optimal chunk size for parallel processing.

        Args:
            total_items: Total number of items to process

        Returns:
            Optimal chunk size
        """
        if int(total_items) <= int(self.n_jobs or 1):
            return 1

        # Aim for 2-4 chunks per worker
        target_chunks_per_worker = 3
        n_jobs_val = int(self.n_jobs or 1)
        optimal_size = max(1, int(total_items) // max(1, n_jobs_val * target_chunks_per_worker))

        return min(optimal_size, self.chunk_size)

    def parallel_map(self, func, items: List[Any], **kwargs) -> List[Any]:
        """
        Execute function in parallel with optimized chunking.

        Args:
            func: Function to execute
            items: Items to process
            **kwargs: Additional arguments for the function

        Returns:
            List of results
        """
        if len(items) <= 1 or self.n_jobs == 1:
            return [func(item, **kwargs) for item in items]

        chunk_size = self.get_optimal_chunk_size(len(items))
        chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

        results: List[Any] = []

        # Submit all chunks once; reuse persistent executor
        future_to_chunk = {
            self._executor.submit(self._process_chunk, func, chunk, **kwargs): chunk
            for chunk in chunks
        }
        for future in as_completed(future_to_chunk):
            try:
                chunk_results = future.result()
                results.extend(chunk_results)
            except Exception as e:
                logger.error(f"Chunk processing failed: {e}")
                chunk = future_to_chunk[future]
                results.extend([None] * len(chunk))
        return results

    def _process_chunk(self, func, chunk: List[Any], **kwargs) -> List[Any]:
        """
        Process a chunk of items.

        Args:
            func: Function to execute
            chunk: Chunk of items to process
            **kwargs: Additional arguments

        Returns:
            List of results for the chunk
        """
        return [func(item, **kwargs) for item in chunk]


class CacheOptimizer:
    """Optimizes caching for frequently accessed data."""

    def __init__(self, max_cache_size: int = 1000):
        """
        Initialize cache optimizer.

        Args:
            max_cache_size: Maximum number of items to cache
        """
        self.max_cache_size = max_cache_size
        self._caches: Dict[str, Any] = {}

    def get_cached_function(self, func, cache_key: str):
        """
        Get a cached version of a function.

        Args:
            func: Function to cache
            cache_key: Unique key for this cache

        Returns:
            Cached function
        """
        if cache_key not in self._caches:
            self._caches[cache_key] = lru_cache(maxsize=self.max_cache_size)(func)

        return self._caches[cache_key]

    def clear_cache(self, cache_key: Optional[str] = None):
        """
        Clear cache(s).

        Args:
            cache_key: Specific cache to clear, or None for all caches
        """
        if cache_key is None:
            for cache in self._caches.values():
                cache.cache_clear()
        elif cache_key in self._caches:
            self._caches[cache_key].cache_clear()

    def get_cache_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics for each cached function
        """
        stats = {}
        for key, cache in self._caches.items():
            if hasattr(cache, "cache_info"):
                info = cache.cache_info()
                stats[key] = {
                    "hits": info.hits,
                    "misses": info.misses,
                    "maxsize": info.maxsize,
                    "currsize": info.currsize,
                    "hit_rate": (
                        info.hits / (info.hits + info.misses)
                        if (info.hits + info.misses) > 0
                        else 0
                    ),
                }
        return stats


class PerformanceMonitor:
    """Monitors and reports performance metrics during optimization."""

    def __init__(self):
        """Initialize performance monitor."""
        # Store lists of measurements; summary is computed on demand
        self.metrics: Dict[str, List[float]] = {
            "evaluation_times": [],
            "memory_usage": [],
            "cache_hit_rates": [],
            "parallel_efficiency": [],
        }
        self.memory_optimizer = MemoryOptimizer()

    def record_evaluation_time(self, time_seconds: float):
        """Record evaluation time."""
        self.metrics["evaluation_times"].append(time_seconds)

    def record_memory_usage(self):
        """Record current memory usage."""
        memory_stats = self.memory_optimizer.check_memory_usage()
        self.metrics["memory_usage"].append(memory_stats["rss_gb"])

    def record_cache_performance(self, cache_stats: Dict[str, Dict[str, int]]):
        """Record cache performance."""
        avg_hit_rate = np.mean(
            [stats["hit_rate"] for stats in cache_stats.values() if stats["hit_rate"] is not None]
        )
        self.metrics["cache_hit_rates"].append(float(avg_hit_rate))

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary.

        Returns:
            Dictionary with performance statistics
        """
        summary = {}

        if self.metrics["evaluation_times"]:
            eval_times = np.array(self.metrics["evaluation_times"])
            summary["evaluation_times"] = {
                "mean": float(np.mean(eval_times)),
                "std": float(np.std(eval_times)),
                "min": float(np.min(eval_times)),
                "max": float(np.max(eval_times)),
                "total": float(np.sum(eval_times)),
            }

        if self.metrics["memory_usage"]:
            memory_usage = np.array(self.metrics["memory_usage"])
            summary["memory_usage"] = {
                "mean_gb": float(np.mean(memory_usage)),
                "max_gb": float(np.max(memory_usage)),
                "min_gb": float(np.min(memory_usage)),
            }

        if self.metrics["cache_hit_rates"]:
            hit_rates = np.array(self.metrics["cache_hit_rates"])
            summary["cache_performance"] = {
                "mean_hit_rate": float(np.mean(hit_rates)),
                "min_hit_rate": float(np.min(hit_rates)),
            }

        return summary


# Global performance optimizers
_memory_optimizer = MemoryOptimizer()
_parallel_optimizer = ParallelOptimizer()
_cache_optimizer = CacheOptimizer()
_performance_monitor = PerformanceMonitor()


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage.

    Args:
        df: DataFrame to optimize

    Returns:
        Memory-optimized DataFrame
    """
    return DataFrameOptimizer.optimize_dtypes(df)


def get_memory_stats() -> Dict[str, float]:
    """Get current memory statistics."""
    return _memory_optimizer.check_memory_usage()


def cleanup_memory_if_needed() -> Optional[Dict[str, float]]:
    """Clean up memory if threshold is exceeded."""
    if _memory_optimizer.should_cleanup():
        return _memory_optimizer.cleanup_memory()
    return None


def get_performance_summary() -> Dict[str, Any]:
    """Get overall performance summary."""
    return _performance_monitor.get_performance_summary()


def record_evaluation_performance(time_seconds: float):
    """Record evaluation performance metrics."""
    _performance_monitor.record_evaluation_time(time_seconds)
    _performance_monitor.record_memory_usage()
