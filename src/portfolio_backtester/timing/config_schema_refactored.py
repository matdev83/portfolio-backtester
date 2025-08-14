"""
Refactored timing configuration schema - alias for backward compatibility.

This module provides backward compatibility for tests that import from the
"refactored" module name.
"""

# Specifically import the main classes and functions for clarity
from .config_schema import (
    TimingConfigSchema,
    validate_timing_config,
    validate_timing_config_file,
)

__all__ = [
    "TimingConfigSchema",
    "validate_timing_config",
    "validate_timing_config_file",
]
