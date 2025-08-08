"""
DEPRECATED: This module has been moved to src/portfolio_backtester/utils/signal_processing/signal_generator.py

This file provides backward compatibility imports with deprecation warnings.
Please update your imports to use the new location.
"""

import warnings
from ...utils.signal_processing.signal_generator import UvxySignalGenerator

warnings.warn(
    "Importing from 'portfolio_backtester.strategies.signal.signal_generator' is deprecated. "
    "Please use 'portfolio_backtester.utils.signal_processing.signal_generator' instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ["UvxySignalGenerator"]