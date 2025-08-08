"""
DEPRECATED: This module has been moved to src/portfolio_backtester/testing/strategies/dummy_signal_strategy.py

This file provides backward compatibility imports with deprecation warnings.
Please update your imports to use the new location.
"""

import warnings
from ...testing.strategies.dummy_signal_strategy import DummySignalStrategy

warnings.warn(
    "Importing from 'portfolio_backtester.strategies.signal.dummy_signal_strategy' is deprecated. "
    "Please use 'portfolio_backtester.testing.strategies.dummy_signal_strategy' instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ["DummySignalStrategy"]