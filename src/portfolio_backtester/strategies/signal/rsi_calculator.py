"""
DEPRECATED: This module has been moved to src/portfolio_backtester/utils/signal_processing/rsi_calculator.py

This file provides backward compatibility imports with deprecation warnings.
Please update your imports to use the new location.
"""

import warnings
from ...utils.signal_processing.rsi_calculator import RSICalculator

warnings.warn(
    "Importing from 'portfolio_backtester.strategies.signal.rsi_calculator' is deprecated. "
    "Please use 'portfolio_backtester.utils.signal_processing.rsi_calculator' instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ["RSICalculator"]