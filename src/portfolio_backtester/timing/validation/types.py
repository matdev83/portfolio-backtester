"""
Common types for timing configuration validation.
DEPRECATED: ValidationError now imported from shared validation module.
"""

# Import from the canonical location to avoid duplication
from ...validation import ValidationError

__all__ = ["ValidationError"]
