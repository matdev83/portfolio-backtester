"""
Utilities for report directory management.

This module provides functions to generate content-based hashes for
report directories and create organized report structures.
"""

import hashlib
import inspect
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..strategies._core.base.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


def get_strategy_source_path(strategy_class: type[BaseStrategy]) -> Optional[Path]:
    """
    Get the absolute path to the source file of a strategy class.

    Args:
        strategy_class: The strategy class.

    Returns:
        The absolute path to the source file, or None if it cannot be determined.
    """
    try:
        source_file = inspect.getsourcefile(strategy_class)
        if source_file:
            return Path(source_file).resolve()
    except (TypeError, OSError):
        logger.warning(f"Could not determine source file for strategy {strategy_class.__name__}")
    return None


def generate_content_hash(
    strategy_file_path: Optional[Path] = None,
    strategy_class: Optional[type[BaseStrategy]] = None,
    config_file_path: Optional[Path] = None,
    config_contents: Optional[str] = None,
) -> str:
    """
    Generate MD5 hash from strategy file contents and config file contents.

    Args:
        strategy_file_path: Path to strategy source file (mutually exclusive with strategy_class).
        strategy_class: Strategy class to derive file path from (mutually exclusive with strategy_file_path).
        config_file_path: Path to configuration file (mutually exclusive with config_contents).
        config_contents: String contents of configuration file (mutually exclusive with config_file_path).

    Returns:
        MD5 hash string (32 hex characters), or empty string if hashing fails.

    Raises:
        ValueError: If both strategy sources are missing and config is missing,
                    or if neither config_file_path nor config_contents is provided.
    """
    hasher = hashlib.md5()

    # Add strategy file contents to hash (if available)
    strategy_path = strategy_file_path
    if strategy_path is None and strategy_class is not None:
        strategy_path = get_strategy_source_path(strategy_class)
        if strategy_path is None:
            logger.warning(f"Could not get strategy file path from class {strategy_class.__name__}")
            return ""

    if strategy_path is not None:
        try:
            with open(strategy_path, "rb") as f:
                hasher.update(f.read())
        except (IOError, FileNotFoundError) as e:
            logger.warning(f"Could not read strategy source file {strategy_path} for hashing: {e}")
            return ""

    # Add config file contents to hash
    if config_file_path is None:
        if config_contents is None:
            if strategy_path is None:
                raise ValueError("Either strategy_file_path, strategy_class, config_file_path, or config_contents must be provided")
            # Only strategy hash, no config - return early
            return hasher.hexdigest()
        hasher.update(config_contents.encode("utf-8"))
    else:
        try:
            with open(config_file_path, "rb") as f:
                hasher.update(f.read())
        except (IOError, FileNotFoundError) as e:
            logger.warning(f"Could not read config file {config_file_path} for hashing: {e}")
            return ""

    return hasher.hexdigest()


def create_report_directory(
    base_reports_dir: Path,
    name: str,
    content_hash: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> Path:
    """
    Create a report directory with hash-based structure.

    Directory structure: <base_reports_dir>/<name>_<hash>/<YYYYMMDD_HHmmSS>

    Args:
        base_reports_dir: Base directory for reports (e.g., Path("data/reports")).
        name: Strategy/script name for the directory.
        content_hash: Optional pre-computed hash. If None, creates simple directory.
        timestamp: Optional timestamp string. If None, uses current time.

    Returns:
        Path to the created report directory.
    """
    base_reports_dir = Path(base_reports_dir)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if content_hash:
        # Create hash-based directory structure
        hash_dir_name = f"{name}_{content_hash}"
        report_dir = base_reports_dir / hash_dir_name / timestamp
    else:
        # Fallback to simple directory structure
        report_dir = base_reports_dir / f"{name}_{timestamp}"

    report_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for different types of outputs
    (report_dir / "plots").mkdir(exist_ok=True)
    (report_dir / "data").mkdir(exist_ok=True)

    return report_dir
