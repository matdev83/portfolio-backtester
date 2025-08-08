"""
Configuration validation cache manager.

This module provides caching functionality for YAML configuration validation
to avoid redundant analysis when files haven't changed between runs.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ConfigValidationCache:
    """
    Manages caching for configuration validation to avoid redundant analysis.

    Uses a two-stage cache invalidation approach:
    1. File count check - quick validation that number of files hasn't changed
    2. Timestamp check - validates that no individual files have been modified
    """

    def __init__(self, cache_file_path: Optional[Path] = None):
        """
        Initialize the cache manager.

        Args:
            cache_file_path: Optional custom path for cache file. If None, uses data/config_validation_cache.json
        """
        project_root = Path(__file__).parent.parent.parent
        if cache_file_path is None:
            data_dir = project_root / "data"
            data_dir.mkdir(exist_ok=True)
            cache_file_path = data_dir / "config_validation_cache.json"

        self.cache_file = cache_file_path
        self.cache_data = self._load_cache()

        # Define monitored directories
        self.monitored_dirs = {
            "scenarios": project_root / "config" / "scenarios",
            "strategies": project_root / "src" / "portfolio_backtester" / "strategies",
            "parameters": project_root / "config" / "parameters.yaml",
        }

    def _load_cache(self) -> Dict[str, Any]:
        """Load cache data from file."""
        if not self.cache_file.exists():
            logger.debug(f"Cache file does not exist: {self.cache_file}")
            return {}

        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                cache_data: Dict[str, Any] = json.load(f)
            logger.debug(f"Loaded cache from {self.cache_file}")
            return cache_data
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load cache file {self.cache_file}: {e}")
            return {}

    def _save_cache(self):
        """Save cache data to file."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache_data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved cache to {self.cache_file}")
        except IOError as e:
            logger.warning(f"Failed to save cache file {self.cache_file}: {e}")

    def _count_files_recursive(self, directory: Path, pattern: str = "*.yaml") -> int:
        """Count files recursively in directory matching pattern."""
        if not directory.exists():
            return 0

        count = 0
        for file_path in directory.rglob(pattern):
            if file_path.is_file():
                count += 1
        return count

    def _get_file_info_recursive(self, directory: Path, pattern: str = "*.yaml") -> List[Dict]:
        """Get file information (path and mtime) recursively."""
        files_info: List[Dict[str, Any]] = []
        if not directory.exists():
            return files_info

        for file_path in directory.rglob(pattern):
            if file_path.is_file():
                try:
                    stat_info = file_path.stat()
                    files_info.append(
                        {
                            "path": str(file_path.resolve()),
                            "mtime": stat_info.st_mtime,
                            "size": stat_info.st_size,
                        }
                    )
                except OSError as e:
                    logger.warning(f"Failed to get stats for {file_path}: {e}")

        return files_info

    def _get_current_state(self) -> Dict[str, Any]:
        """Get current state of all monitored directories and files."""
        state: Dict[str, Any] = {"timestamp": time.time(), "file_counts": {}, "files": {}}

        # Count files in each directory
        state["file_counts"]["scenarios"] = self._count_files_recursive(
            self.monitored_dirs["scenarios"], "*.yaml"
        )
        state["file_counts"]["strategies"] = self._count_files_recursive(
            self.monitored_dirs["strategies"], "*.py"
        )

        # Handle parameters file separately (single file)
        params_file = self.monitored_dirs["parameters"]
        if params_file.exists():
            stat_info = params_file.stat()
            state["file_counts"]["parameters"] = 1
            state["files"]["parameters"] = [
                {
                    "path": str(params_file.resolve()),
                    "mtime": stat_info.st_mtime,
                    "size": stat_info.st_size,
                }
            ]
        else:
            state["file_counts"]["parameters"] = 0
            state["files"]["parameters"] = []

        # Get detailed file information
        state["files"]["scenarios"] = self._get_file_info_recursive(
            self.monitored_dirs["scenarios"], "*.yaml"
        )
        state["files"]["strategies"] = self._get_file_info_recursive(
            self.monitored_dirs["strategies"], "*.py"
        )

        return state

    def _compare_file_counts(
        self, cached_state: Dict, current_state: Dict
    ) -> Tuple[bool, List[str]]:
        """Compare file counts between cached and current state."""
        reasons = []

        for location in ["scenarios", "strategies", "parameters"]:
            cached_count = cached_state.get("file_counts", {}).get(location, 0)
            current_count = current_state["file_counts"][location]

            if cached_count != current_count:
                reasons.append(f"{location}: {cached_count} -> {current_count} files")

        return len(reasons) == 0, reasons

    def _compare_file_timestamps(
        self, cached_state: Dict, current_state: Dict
    ) -> Tuple[bool, List[str]]:
        """Compare file timestamps between cached and current state."""
        reasons = []

        for location in ["scenarios", "strategies", "parameters"]:
            cached_files = {f["path"]: f for f in cached_state.get("files", {}).get(location, [])}
            current_files = {f["path"]: f for f in current_state["files"][location]}

            # Check for new files
            new_files = set(current_files.keys()) - set(cached_files.keys())
            if new_files:
                reasons.append(f"{location}: {len(new_files)} new files")

            # Check for deleted files
            deleted_files = set(cached_files.keys()) - set(current_files.keys())
            if deleted_files:
                reasons.append(f"{location}: {len(deleted_files)} deleted files")

            # Check for modified files
            modified_count = 0
            for file_path in set(cached_files.keys()) & set(current_files.keys()):
                cached_file = cached_files[file_path]
                current_file = current_files[file_path]

                if (
                    cached_file["mtime"] != current_file["mtime"]
                    or cached_file["size"] != current_file["size"]
                ):
                    modified_count += 1

            if modified_count > 0:
                reasons.append(f"{location}: {modified_count} modified files")

        return len(reasons) == 0, reasons

    def is_cache_valid(self, cache_key: str = "validation") -> Tuple[bool, Optional[str]]:
        """
        Check if cache is valid for the given key.

        Args:
            cache_key: The cache key to check (e.g., "validation", "cross_check")

        Returns:
            Tuple of (is_valid, reason_for_invalidation)
        """
        if cache_key not in self.cache_data:
            return False, "No cached data found"

        cached_entry = self.cache_data[cache_key]
        current_state = self._get_current_state()

        # Stage 1: Quick file count check
        counts_match, count_reasons = self._compare_file_counts(
            cached_entry.get("state", {}), current_state
        )

        if not counts_match:
            reason = f"File count changes detected: {'; '.join(count_reasons)}"
            logger.info(f"Cache invalidated: {reason}")
            return False, reason

        # Stage 2: Detailed timestamp check
        timestamps_match, timestamp_reasons = self._compare_file_timestamps(
            cached_entry.get("state", {}), current_state
        )

        if not timestamps_match:
            reason = f"File modifications detected: {'; '.join(timestamp_reasons)}"
            logger.info(f"Cache invalidated: {reason}")
            return False, reason

        # Cache is valid
        cache_age = time.time() - cached_entry.get("state", {}).get("timestamp", 0)
        logger.info(f"Using cached validation results (age: {cache_age:.1f}s)")
        return True, None

    def get_cached_result(self, cache_key: str = "validation") -> Optional[Dict]:
        """
        Get cached validation result.

        Args:
            cache_key: The cache key to retrieve

        Returns:
            Cached result or None if not found/invalid
        """
        is_valid, _ = self.is_cache_valid(cache_key)
        if not is_valid:
            return None

        result: Optional[Dict] = self.cache_data[cache_key].get("result")
        return result

    def store_result(self, result: Dict, cache_key: str = "validation"):
        """
        Store validation result in cache.

        Args:
            result: The validation result to cache
            cache_key: The cache key to use
        """
        current_state = self._get_current_state()

        self.cache_data[cache_key] = {"result": result, "state": current_state}

        self._save_cache()

        total_files = sum(current_state["file_counts"].values())
        logger.info(f"Cached validation results for {total_files} files")

    def clear_cache(self, cache_key: Optional[str] = None):
        """
        Clear cache data.

        Args:
            cache_key: Specific key to clear, or None to clear all
        """
        if cache_key is None:
            self.cache_data.clear()
            logger.info("Cleared entire validation cache")
        elif cache_key in self.cache_data:
            del self.cache_data[cache_key]
            logger.info(f"Cleared cache for key: {cache_key}")

        self._save_cache()

    def get_cache_info(self) -> Dict:
        """Get information about current cache state."""
        info = {
            "cache_file": str(self.cache_file),
            "cache_exists": self.cache_file.exists(),
            "cached_keys": list(self.cache_data.keys()),
            "monitored_directories": {k: str(v) for k, v in self.monitored_dirs.items()},
        }

        if self.cache_data:
            for key, entry in self.cache_data.items():
                state = entry.get("state", {})
                info[f"{key}_timestamp"] = state.get("timestamp")
                info[f"{key}_file_counts"] = state.get("file_counts", {})

        return info


# Global cache instance
_cache_instance = None


def get_cache() -> ConfigValidationCache:
    """Get the global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = ConfigValidationCache()
    return _cache_instance
