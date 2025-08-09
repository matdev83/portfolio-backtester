"""
DEPRECATED: This module has been moved to src/portfolio_backtester/utils/signal_processing/price_data_processor.py

This file provides backward compatibility imports with deprecation warnings.
Please update your imports to use the new location.
"""

import warnings
from ...utils.signal_processing.price_data_processor import PriceDataProcessor

warnings.warn(
    "Importing from 'portfolio_backtester.strategies.signal.price_data_processor' is deprecated. "
    "Please use 'portfolio_backtester.utils.signal_processing.price_data_processor' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["PriceDataProcessor"]
