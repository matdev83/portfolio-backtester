"""Core registry re-exports for compatibility with new layout.

Expose get_strategy_registry and related functions from the new internal path.
"""

from .registry.strategy_registry import (
    get_strategy_registry,
    clear_strategy_registry,
)

__all__ = [
    "get_strategy_registry",
    "clear_strategy_registry",
]
