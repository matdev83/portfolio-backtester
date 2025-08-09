"""Core location for StrategyFactory after strategy layout refactor.

This module re-exports the concrete `StrategyFactory` API from the new
implementation module in `_core`.
"""

from .strategy_factory_impl import StrategyFactory  # noqa: F401

# Back-compat aliases for tests that patch portfolio_backtester.strategies.strategy_factory
import sys as _sys

_sys.modules.setdefault(
    "portfolio_backtester.strategies.strategy_factory",
    _sys.modules[__name__],
)

__all__ = ["StrategyFactory"]
