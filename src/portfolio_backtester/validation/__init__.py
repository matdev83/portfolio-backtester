"""Shared validation components for the portfolio backtester framework.

This module provides common validation data structures and utilities
used across all validation systems (config, timing, etc.) to ensure
consistency and avoid duplication.
"""

from .validation_error import ValidationError

__all__ = ["ValidationError"]
