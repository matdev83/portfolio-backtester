"""
Context-aware study management for Optuna.

This module provides utilities to automatically manage Optuna study databases
by comparing the hash of the current configuration (scenario + strategy source)
with a hash stored in the study's user attributes.
"""

import hashlib
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type

from ..strategies._core.base.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


def get_strategy_source_path(strategy_class: Type[BaseStrategy]) -> Optional[Path]:
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


def generate_context_hash(scenario_config: Dict[str, Any], strategy_path: Path) -> str:
    """
    Generate a SHA256 hash from the scenario config and strategy source code.

    Args:
        scenario_config: The scenario configuration dictionary.
        strategy_path: The path to the strategy's source code file.

    Returns:
        A SHA256 hash string.
    """
    hasher = hashlib.sha256()

    # 1. Add scenario config to hash
    # Convert dict to a canonical string representation (sorted keys)
    scenario_str = str(sorted(scenario_config.items()))
    hasher.update(scenario_str.encode("utf-8"))

    # 2. Add strategy source code to hash
    try:
        with open(strategy_path, "rb") as f:
            hasher.update(f.read())
    except (IOError, FileNotFoundError) as e:
        logger.warning(f"Could not read strategy source file {strategy_path} for hashing: {e}")

    return hasher.hexdigest()
