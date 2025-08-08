"""Shared validation error data structure for the portfolio backtester framework.

This module defines the canonical ValidationError dataclass used throughout
all validation systems (config, timing, etc.) to ensure consistency and
avoid duplication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ValidationError:
    """Represents a configuration validation error or warning.

    This is the canonical ValidationError class used throughout the framework
    to ensure consistency across all validation systems.
    """

    field: str
    value: Any
    message: str
    suggestion: Optional[str] = None
    severity: str = "error"  # "error", "warning", "info"
