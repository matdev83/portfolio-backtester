"""
Column type detectors for eliminating isinstance violations.

This module provides polymorphic interfaces for detecting DataFrame column types
and handling different data structures without using isinstance checks.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import pandas as pd


class IColumnTypeDetector(ABC):
    """Interface for detecting DataFrame column types."""

    @abstractmethod
    def is_multiindex(self, columns: Any) -> bool:
        """Check if columns represent a MultiIndex structure."""
        pass

    @abstractmethod
    def validate_multiindex_requirement(self, dataframe: pd.DataFrame) -> None:
        """Validate that DataFrame has MultiIndex columns when required."""
        pass


class PolymorphicColumnTypeDetector(IColumnTypeDetector):
    """Polymorphic column type detector that uses duck typing instead of isinstance."""

    def is_multiindex(self, columns: Any) -> bool:
        """Check if columns is a MultiIndex using duck typing."""
        return (
            hasattr(columns, "nlevels")
            and hasattr(columns, "names")
            and hasattr(columns, "get_level_values")
            and getattr(columns, "nlevels", 1) > 1
        )

    def validate_multiindex_requirement(self, dataframe: pd.DataFrame) -> None:
        """Validate that DataFrame has MultiIndex columns."""
        if not self.is_multiindex(dataframe.columns):
            raise ValueError("MultiIndexColumnHandler requires DataFrame with MultiIndex columns")


class IDataStructureNormalizer(ABC):
    """Interface for normalizing different data structures."""

    @abstractmethod
    def can_normalize(self, data: Any) -> bool:
        """Check if this normalizer can handle the given data type."""
        pass

    @abstractmethod
    def to_dataframe(self, data: Any, name: Optional[str] = None) -> pd.DataFrame:
        """Convert data to DataFrame format."""
        pass


class SeriesNormalizer(IDataStructureNormalizer):
    """Normalizer for Series data structures."""

    def can_normalize(self, data: Any) -> bool:
        """Check if data is a Series using duck typing."""
        return (
            hasattr(data, "to_frame")
            and hasattr(data, "index")
            and hasattr(data, "name")
            and not hasattr(data, "columns")
        )

    def to_dataframe(self, data: Any, name: Optional[str] = None) -> pd.DataFrame:
        """Convert Series to DataFrame."""
        result = data.to_frame(name=name or "value")
        return pd.DataFrame(result)


class DataFrameNormalizer(IDataStructureNormalizer):
    """Normalizer for DataFrame data structures."""

    def can_normalize(self, data: Any) -> bool:
        """Check if data is a DataFrame using duck typing."""
        return hasattr(data, "columns") and hasattr(data, "index") and hasattr(data, "values")

    def to_dataframe(self, data: Any, name: Optional[str] = None) -> pd.DataFrame:
        """Return DataFrame as-is."""
        return pd.DataFrame(data)


class NullNormalizer(IDataStructureNormalizer):
    """Fallback normalizer for unsupported data types."""

    def can_normalize(self, data: Any) -> bool:
        """Always returns True as fallback."""
        return True

    def to_dataframe(self, data: Any, name: Optional[str] = None) -> pd.DataFrame:
        """Raise error for unsupported data types."""
        raise ValueError(f"Cannot normalize data type {type(data)} to DataFrame")


class DataStructureNormalizerFactory:
    """Factory for creating appropriate data structure normalizers."""

    def __init__(self):
        self._normalizers = [
            SeriesNormalizer(),
            DataFrameNormalizer(),
            NullNormalizer(),  # Fallback
        ]

    def get_normalizer(self, data: Any) -> IDataStructureNormalizer:
        """Get the appropriate normalizer for the given data."""
        for normalizer in self._normalizers:
            if normalizer.can_normalize(data):
                return normalizer
        # Should never reach here due to NullNormalizer fallback
        return self._normalizers[-1]


class PolymorphicDataStructureHandler:
    """Polymorphic handler for data structure operations without isinstance."""

    def __init__(self):
        self._normalizer_factory = DataStructureNormalizerFactory()

    def ensure_dataframe(self, data: Any, name: Optional[str] = None) -> pd.DataFrame:
        """
        Ensure data is in DataFrame format using polymorphic normalization.

        Args:
            data: Data to normalize (Series, DataFrame, etc.)
            name: Name to use if converting Series to DataFrame

        Returns:
            DataFrame representation of the data
        """
        normalizer = self._normalizer_factory.get_normalizer(data)
        return normalizer.to_dataframe(data, name)


# Factory functions for easy instantiation
def create_column_type_detector() -> PolymorphicColumnTypeDetector:
    """Create a new polymorphic column type detector instance."""
    return PolymorphicColumnTypeDetector()


def create_data_structure_handler() -> PolymorphicDataStructureHandler:
    """Create a new polymorphic data structure handler instance."""
    return PolymorphicDataStructureHandler()
