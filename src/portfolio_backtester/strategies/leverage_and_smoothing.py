"""
DEPRECATED: This module has been moved to src/portfolio_backtester/utils/portfolio_utils.py

This file provides backward compatibility imports with deprecation warnings.
Please update your imports to use the new location.
"""

import warnings
from ..utils.portfolio_utils import apply_leverage_and_smoothing

warnings.warn(
    "Importing from 'portfolio_backtester.strategies.leverage_and_smoothing' is deprecated. "
    "Please use 'portfolio_backtester.utils.portfolio_utils.apply_leverage_and_smoothing' instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ["apply_leverage_and_smoothing"]