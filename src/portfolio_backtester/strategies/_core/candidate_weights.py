"""Core location for candidate weights utilities after strategy layout refactor.

This module re-exports the stable public API from `portfolio_backtester.utils` so
callers can import from `portfolio_backtester.strategies._core.candidate_weights`.
"""

from portfolio_backtester.utils.portfolio_utils import (  # noqa: F401
    default_candidate_weights,
)

__all__ = ["default_candidate_weights"]
