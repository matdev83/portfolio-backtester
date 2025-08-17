"""
Common test strategies for Hypothesis-based property testing.

This package provides reusable strategies for property-based testing with Hypothesis.
These strategies generate test data with appropriate constraints for testing
different components of the portfolio backtester.
"""

from .common_strategies import (
    price_dataframes,
    return_series,
    return_matrices,
    price_series,
    timestamps,
    frequencies,
    weights_and_leverage,
)

__all__ = [
    "price_dataframes",
    "return_series",
    "return_matrices",
    "price_series",
    "timestamps",
    "frequencies",
    "weights_and_leverage",
]