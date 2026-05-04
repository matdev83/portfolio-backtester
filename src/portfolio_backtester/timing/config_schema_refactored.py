"""
Deprecated: use ``portfolio_backtester.timing.config_schema`` instead.

This module remains as a thin re-export for legacy imports only.
"""

from __future__ import annotations

import warnings

from .config_schema import (
    TimingConfigSchema,
    validate_timing_config,
    validate_timing_config_file,
)

warnings.warn(
    "portfolio_backtester.timing.config_schema_refactored is deprecated; "
    "import from portfolio_backtester.timing.config_schema.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "TimingConfigSchema",
    "validate_timing_config",
    "validate_timing_config_file",
]
