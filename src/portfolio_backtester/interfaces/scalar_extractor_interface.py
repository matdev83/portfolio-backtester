"""
Polymorphic interfaces for scalar extraction from various data types.

This module provides interfaces to replace isinstance violations in pandas_utils.py,
implementing the Open/Closed Principle by allowing extension without modification.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import pandas as pd


class IScalarExtractor(ABC):
    """Interface for extracting numeric scalar values from various data types."""

    @abstractmethod
    def can_extract(self, value: Any) -> bool:
        """Check if this extractor can handle the given value type."""
        pass

    @abstractmethod
    def extract_scalar(self, value: Any) -> Optional[float]:
        """Extract a numeric scalar from the value, returning None if not possible."""
        pass


class PandasScalarExtractor(IScalarExtractor):
    """Extractor for pandas Series and DataFrame objects."""

    def can_extract(self, value: Any) -> bool:
        """Check if value is a pandas Series or DataFrame."""
        return isinstance(value, (pd.Series, pd.DataFrame))

    def extract_scalar(self, value: Any) -> Optional[float]:
        """Extract scalar from pandas object with exactly one element."""
        if not self.can_extract(value):
            return None

        try:
            # Convert to numpy array to avoid ExtensionArray .item() typing issues
            arr = np.asarray(value.values)
            if arr.size == 1:
                item = arr.reshape(-1)[0]
                # Use NumericItemExtractor for polymorphic item handling
                item_extractor = NumericItemExtractor()
                if item_extractor.can_extract(item):
                    return item_extractor.extract_scalar(item)
            return None
        except Exception:
            return None


class NumericScalarExtractor(IScalarExtractor):
    """Extractor for plain numeric scalar values."""

    def can_extract(self, value: Any) -> bool:
        """Check if value is a numeric scalar type."""
        return isinstance(value, (int, float, np.integer, np.floating))

    def extract_scalar(self, value: Any) -> Optional[float]:
        """Extract float from numeric scalar, checking for NaN."""
        if not self.can_extract(value):
            return None

        if not pd.isna(value):
            return float(value)
        return None


class NumericItemExtractor(IScalarExtractor):
    """Extractor for numeric items from numpy arrays (used internally by PandasScalarExtractor)."""

    def can_extract(self, item: Any) -> bool:
        """Check if item is a numeric type suitable for float conversion."""
        return isinstance(item, (np.floating, float, int, np.integer))

    def extract_scalar(self, item: Any) -> Optional[float]:
        """Extract float from numeric item, checking for NaN."""
        if not self.can_extract(item):
            return None

        if not pd.isna(item):
            return float(item)
        return None


class NullScalarExtractor(IScalarExtractor):
    """Default extractor that returns None for unsupported types."""

    def can_extract(self, value: Any) -> bool:
        """Always returns True as this is the fallback extractor."""
        return True

    def extract_scalar(self, value: Any) -> Optional[float]:
        """Always returns None for unsupported value types."""
        return None


class ScalarExtractorFactory:
    """Factory for creating appropriate scalar extractors based on value type."""

    _extractors = [
        PandasScalarExtractor(),
        NumericScalarExtractor(),
        NullScalarExtractor(),  # Must be last as it accepts everything
    ]

    @classmethod
    def get_extractor(cls, value: Any) -> IScalarExtractor:
        """Get the appropriate scalar extractor for the given value type."""
        for extractor in cls._extractors:
            if extractor.can_extract(value):
                return extractor

        # This should never happen due to NullScalarExtractor being last
        return NullScalarExtractor()


class PolymorphicScalarExtractor:
    """Main polymorphic scalar extractor that delegates to type-specific extractors."""

    def __init__(self, factory: Optional[ScalarExtractorFactory] = None):
        """Initialize with optional custom factory."""
        self._factory = factory or ScalarExtractorFactory()

    def extract_numeric_scalar(self, value: Any) -> Optional[float]:
        """
        Best-effort extraction of a numeric float scalar using polymorphic extractors.

        This method replaces isinstance violations with polymorphic dispatch,
        following the Open/Closed Principle.
        """
        extractor = self._factory.get_extractor(value)
        return extractor.extract_scalar(value)
