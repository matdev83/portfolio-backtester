"""Compatibility wrapper for BaseStrategy in new core path.

This file re-exports the original BaseStrategy to the `_core` namespace so
that code can import `portfolio_backtester.strategies._core.base.base_strategy`
while we migrate files.
"""

from .base.base_strategy import BaseStrategy  # noqa: F401

__all__ = ["BaseStrategy"]
