"""
Provides interfaces for polymorphic normalization of price data to DataFrame format.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import pandas as pd


class ISeriesNormalizer(ABC):
    """Interface for normalizing price data to DataFrame format."""

    @abstractmethod
    def can_normalize(self, data: Any) -> bool:
        """Returns True if the normalizer can handle the data type."""
        pass

    @abstractmethod
    def normalize(self, data: Any, target_columns: Optional[pd.Index] = None) -> pd.DataFrame:
        """Normalizes data to DataFrame format."""
        pass


class SeriesNormalizer(ISeriesNormalizer):
    """Normalizes pandas Series to DataFrame."""

    def can_normalize(self, data: Any) -> bool:
        return isinstance(data, pd.Series)

    def normalize(self, data: pd.Series, target_columns: Optional[pd.Index] = None) -> pd.DataFrame:
        price_df = data.to_frame()

        if target_columns is not None:
            price_df = price_df.reindex(columns=target_columns)

        return price_df


class DataFrameNormalizer(ISeriesNormalizer):
    """Handles DataFrame input by copying it."""

    def can_normalize(self, data: Any) -> bool:
        return isinstance(data, pd.DataFrame)

    def normalize(
        self, data: pd.DataFrame, target_columns: Optional[pd.Index] = None
    ) -> pd.DataFrame:
        price_df = data.copy()

        if target_columns is not None:
            price_df = price_df.reindex(columns=target_columns)

        return price_df


class SeriesNormalizerFactory:
    """Factory for creating appropriate series normalizers."""

    def __init__(self):
        self._normalizers = [SeriesNormalizer(), DataFrameNormalizer()]

    def get_normalizer(self, data: Any) -> ISeriesNormalizer:
        """Gets the appropriate normalizer for the data."""
        for normalizer in self._normalizers:
            if normalizer.can_normalize(data):
                return normalizer

        # Default behavior: assume it's a DataFrame-like structure
        return DataFrameNormalizer()
