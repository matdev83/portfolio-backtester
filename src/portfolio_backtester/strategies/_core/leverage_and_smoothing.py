"""Core location for leverage and smoothing utilities after strategy layout refactor.

This module re-exports the stable public API from `portfolio_backtester.utils` so
callers can import from `portfolio_backtester.strategies._core.leverage_and_smoothing`.
"""

from portfolio_backtester.utils.portfolio_utils import (  # noqa: F401
    apply_leverage_and_smoothing,
)

__all__ = ["apply_leverage_and_smoothing"]
