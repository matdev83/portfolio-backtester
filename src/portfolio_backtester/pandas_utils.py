from __future__ import annotations

from typing import Any, Optional

from .interfaces.scalar_extractor_interface import PolymorphicScalarExtractor

# Global polymorphic extractor instance for efficient reuse
_scalar_extractor = PolymorphicScalarExtractor()


def extract_numeric_scalar(value: Any) -> Optional[float]:
    """
    Best-effort extraction of a numeric float scalar from a pandas DataFrame/Series cell or a plain scalar.

    Rules:
    - If value is a pandas Series/DataFrame with exactly one element, return that element as float if numeric and not NaN.
    - If value is a plain numeric scalar (int/float) and not NaN, return float(value).
    - Otherwise, return None.

    This function is designed to reduce mypy/pandas union typing noise at call sites by centralizing
    narrowing logic in one place.

    Implementation now uses polymorphic extractors following the Open/Closed Principle,
    eliminating isinstance violations for better extensibility.
    """
    return _scalar_extractor.extract_numeric_scalar(value)
