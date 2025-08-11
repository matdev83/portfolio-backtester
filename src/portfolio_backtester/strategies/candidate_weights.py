"""
DEPRECATED: This module has been moved to src/portfolio_backtester/utils/portfolio_utils.py

This file provides backward compatibility imports with deprecation warnings.
Please update your imports to use the new location.
"""

import warnings
from ..utils.portfolio_utils import default_candidate_weights

warnings.warn(
    "Importing from 'portfolio_backtester.strategies.candidate_weights' is deprecated. "
    "Please use 'portfolio_backtester.utils.portfolio_utils.default_candidate_weights' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["default_candidate_weights"]
