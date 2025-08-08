"""
Refactored timing logger - alias for backward compatibility.

This module provides backward compatibility for tests that import from the
"refactored" module name.
"""

# Specifically import the main classes and functions for clarity
from .timing_logger import (
    TimingLogger,
    get_timing_logger,
    configure_timing_logging,
    log_signal_generation,
    log_position_update,
    log_rebalance_event,
)

__all__ = [
    "TimingLogger",
    "get_timing_logger",
    "configure_timing_logging",
    "log_signal_generation",
    "log_position_update",
    "log_rebalance_event",
]
