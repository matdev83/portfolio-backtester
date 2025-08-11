"""
DEPRECATED: This module has been moved to src/portfolio_backtester/testing/strategies/stop_loss_tester_strategy.py

This file provides backward compatibility imports with deprecation warnings.
Please update your imports to use the new location.
"""

import warnings
from ...testing.strategies.stop_loss_tester_strategy import StopLossTesterStrategy

warnings.warn(
    "Importing from 'portfolio_backtester.strategies.diagnostic.stop_loss_tester_strategy' is deprecated. "
    "Please use 'portfolio_backtester.testing.strategies.stop_loss_tester_strategy' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["StopLossTesterStrategy"]
