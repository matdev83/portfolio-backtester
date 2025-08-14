"""Universe loader module for managing named universes.

This module provides functionality to load and manage named universes from text files.
Universe files are stored in config/universes/ directory with .txt extension.

File format:
- One ticker per line
- Comments supported with # (entire line or end of line)
- Empty lines are ignored
- Whitespace is stripped

Example universe file (config/universes/sp500_top10.txt):
```
# Top 10 S&P 500 companies by market cap
AAPL
MSFT
GOOGL  # Alphabet Class A
AMZN
NVDA
TSLA
META
BRK.B  # Berkshire Hathaway Class B
UNH
JNJ
```
"""

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import List
from .api_stability import api_stable

logger = logging.getLogger(__name__)

# Path to universes directory
UNIVERSES_DIR = Path(__file__).parent.parent.parent / "config" / "universes"

# Regex pattern for valid ticker symbols
# Allows: A-Z, 0-9, dots, hyphens (common in ticker symbols)
TICKER_PATTERN = re.compile(r"^[A-Z0-9.-]+$")


class UniverseLoaderError(Exception):
    """Custom exception for universe loading errors."""

    pass


def _validate_ticker(ticker: str) -> bool:
    """
    Validate ticker symbol format.

    Args:
        ticker: Ticker symbol to validate

    Returns:
        True if ticker is valid, False otherwise
    """
    if not ticker:
        return False

    # Basic length check (1-10 characters is reasonable for most tickers)
    if len(ticker) < 1 or len(ticker) > 10:
        return False

    # Pattern check
    return bool(TICKER_PATTERN.match(ticker))


def _parse_universe_file(file_path: Path) -> List[str]:
    """
    Parse a universe file and extract ticker symbols.

    Args:
        file_path: Path to the universe file

    Returns:
        List of ticker symbols

    Raises:
        UniverseLoaderError: If file cannot be read or contains invalid tickers
    """
    if not file_path.exists():
        raise UniverseLoaderError(f"Universe file not found: {file_path}")

    if not file_path.is_file():
        raise UniverseLoaderError(f"Universe path is not a file: {file_path}")

    tickers = []
    invalid_tickers = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                # Remove comments (everything after #)
                if "#" in line:
                    line = line[: line.index("#")]

                # Strip whitespace
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Validate ticker
                ticker = line.upper()  # Normalize to uppercase
                if _validate_ticker(ticker):
                    tickers.append(ticker)
                else:
                    invalid_tickers.append((line_num, line))

    except Exception as e:
        raise UniverseLoaderError(f"Error reading universe file {file_path}: {e}")

    # Report invalid tickers
    if invalid_tickers:
        error_details = []
        for line_num, invalid_ticker in invalid_tickers:
            error_details.append(f"Line {line_num}: '{invalid_ticker}'")

        error_msg = f"Invalid tickers found in {file_path}:\n" + "\n".join(error_details)
        raise UniverseLoaderError(error_msg)

    # Remove duplicates while preserving order
    seen = set()
    unique_tickers = []
    for ticker in tickers:
        if ticker not in seen:
            seen.add(ticker)
            unique_tickers.append(ticker)

    if len(unique_tickers) != len(tickers):
        duplicates_count = len(tickers) - len(unique_tickers)
        logger.warning(f"Removed {duplicates_count} duplicate tickers from {file_path}")

    return unique_tickers


@lru_cache(maxsize=128)
@api_stable(version="1.0", strict_params=True, strict_return=False)
def load_named_universe(universe_name: str) -> List[str]:
    """
    Load a named universe from a text file.

    Args:
        universe_name: Name of the universe (without .txt extension)

    Returns:
        List of ticker symbols

    Raises:
        UniverseLoaderError: If universe cannot be loaded
    """
    if not universe_name:
        raise UniverseLoaderError("Universe name cannot be empty")

    # Sanitize universe name (remove any path components for security)
    clean_name = Path(universe_name).name
    if clean_name != universe_name:
        logger.warning(f"Universe name sanitized from '{universe_name}' to '{clean_name}'")

    # Construct file path
    file_path = UNIVERSES_DIR / f"{clean_name}.txt"

    logger.debug(f"Loading universe '{universe_name}' from {file_path}")

    try:
        tickers = _parse_universe_file(file_path)
        logger.info(f"Loaded universe '{universe_name}' with {len(tickers)} tickers")
        return tickers

    except UniverseLoaderError:
        # Re-raise universe loader errors as-is
        raise
    except Exception as e:
        # Wrap other exceptions
        raise UniverseLoaderError(f"Unexpected error loading universe '{universe_name}': {e}")


def load_multiple_named_universes(universe_names: List[str]) -> List[str]:
    """
    Load multiple named universes and return their union.

    Args:
        universe_names: List of universe names to load

    Returns:
        List of unique ticker symbols from all universes

    Raises:
        UniverseLoaderError: If any universe cannot be loaded
    """
    if not universe_names:
        return []

    all_tickers = []
    for universe_name in universe_names:
        tickers = load_named_universe(universe_name)
        all_tickers.extend(tickers)

    # Remove duplicates while preserving order
    seen = set()
    unique_tickers = []
    for ticker in all_tickers:
        if ticker not in seen:
            seen.add(ticker)
            unique_tickers.append(ticker)

    logger.info(f"Loaded {len(universe_names)} universes with {len(unique_tickers)} unique tickers")
    return unique_tickers


def list_available_universes() -> List[str]:
    """
    List all available named universes.

    Returns:
        List of universe names (without .txt extension)
    """
    if not UNIVERSES_DIR.exists():
        logger.warning(f"Universes directory does not exist: {UNIVERSES_DIR}")
        return []

    universe_files = list(UNIVERSES_DIR.glob("*.txt"))
    universe_names = [f.stem for f in universe_files]
    universe_names.sort()

    logger.debug(f"Found {len(universe_names)} available universes")
    return universe_names


def validate_universe_exists(universe_name: str) -> bool:
    """
    Check if a named universe exists.

    Args:
        universe_name: Name of the universe to check

    Returns:
        True if universe exists, False otherwise
    """
    if not universe_name:
        return False

    clean_name = Path(universe_name).name
    file_path = UNIVERSES_DIR / f"{clean_name}.txt"
    return file_path.exists() and file_path.is_file()


def clear_universe_cache() -> None:
    """Clear the universe loading cache."""
    load_named_universe.cache_clear()
    logger.debug("Universe cache cleared")


def get_universe_info(universe_name: str) -> dict[str, object]:
    """
    Get information about a named universe.

    Args:
        universe_name: Name of the universe

    Returns:
        Dictionary with universe information

    Raises:
        UniverseLoaderError: If universe cannot be loaded
    """
    tickers = load_named_universe(universe_name)

    clean_name = Path(universe_name).name
    file_path = UNIVERSES_DIR / f"{clean_name}.txt"

    return {
        "name": universe_name,
        "file_path": str(file_path),
        "ticker_count": len(tickers),
        "tickers": tickers,
        "file_exists": file_path.exists(),
        "file_size_bytes": file_path.stat().st_size if file_path.exists() else 0,
    }


# Export public API
__all__ = [
    "UniverseLoaderError",
    "load_named_universe",
    "load_multiple_named_universes",
    "list_available_universes",
    "validate_universe_exists",
    "clear_universe_cache",
    "get_universe_info",
]
