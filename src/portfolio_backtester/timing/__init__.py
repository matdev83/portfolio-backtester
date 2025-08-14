"""
Timing framework for flexible portfolio rebalancing strategies.

This module provides a modular timing system that allows strategies to use either
traditional time-based rebalancing or custom timing signals based on market conditions.
"""

from .timing_controller import TimingController

# Interface-based imports - ALL NEW CODE MUST USE THESE
from ..interfaces.timing_state_interface import ITimingState, create_timing_state
from ..interfaces.timing_base_interface import create_signal_based_timing
from ..interfaces.time_based_timing_interface import create_time_based_timing

# Factory for creating timing controllers
from .custom_timing_registry import TimingControllerFactory
from .config_validator import TimingConfigValidator
from .config_schema import (
    TimingConfigSchema,
    validate_timing_config,
    validate_timing_config_file,
)
from .timing_logger import TimingLogger, get_timing_logger, configure_timing_logging
from .logging import LogEntryManager, TimingLogEntry, LogExporter, LogAnalyzer
from .state_management import (
    PositionTracker,
    PositionInfo,
    StateStatistics,
    StateSerializer,
)

# Import custom timing registry to ensure built-in controllers are registered
from . import custom_timing_registry  # noqa: F401

__all__ = [
    "TimingController",
    # Interface-based factories - ALL NEW CODE MUST USE THESE
    "ITimingState",
    "create_timing_state",
    "create_signal_based_timing",
    "create_time_based_timing",
    "TimingControllerFactory",
    # Configuration and utilities
    "TimingConfigValidator",
    "TimingConfigSchema",
    "validate_timing_config",
    "validate_timing_config_file",
    "TimingLogger",
    "get_timing_logger",
    "configure_timing_logging",
    "LogEntryManager",
    "TimingLogEntry",
    "LogExporter",
    "LogAnalyzer",
    "PositionTracker",
    "PositionInfo",
    "StateStatistics",
    "StateSerializer",
]
