"""
Interfaces for extracting close prices from DataFrames.

This module provides polymorphic interfaces and implementations for extracting
close prices from differently structured DataFrames, replacing isinstance
violations with a more extensible strategy pattern.
"""

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class IClosePriceExtractor(ABC):
    """Interface for extracting close prices from a DataFrame."""

    @abstractmethod
    def can_handle(self, data: pd.DataFrame) -> bool:
        """
        Check if this extractor can handle the given DataFrame format.

        Args:
            data: DataFrame to check

        Returns:
            True if this extractor can handle the DataFrame, False otherwise
        """
        pass

    @abstractmethod
    def extract_close_prices(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Extract close prices from the DataFrame.

        Args:
            data: DataFrame to extract close prices from

        Returns:
            DataFrame with close prices, or None if extraction fails
        """
        pass


class MultiIndexFieldClosePriceExtractor(IClosePriceExtractor):
    """Extractor for DataFrames with 'Close' in the 'Field' level of a MultiIndex."""

    def can_handle(self, data: pd.DataFrame) -> bool:
        """Check for MultiIndex with 'Close' in level 1."""
        return isinstance(data.columns, pd.MultiIndex) and "Close" in data.columns.get_level_values(
            1
        )

    def extract_close_prices(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Extract close prices using xs method."""
        extracted = data.xs("Close", level="Field", axis=1)
        return extracted if isinstance(extracted, pd.DataFrame) else extracted.to_frame()


class SimpleColumnClosePriceExtractor(IClosePriceExtractor):
    """Extractor for DataFrames with simple columns (already close prices)."""

    def can_handle(self, data: pd.DataFrame) -> bool:
        """Check if columns are not MultiIndex."""
        return not isinstance(data.columns, pd.MultiIndex)

    def extract_close_prices(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Return the original DataFrame as is."""
        return data


class MultiIndexLastLevelClosePriceExtractor(IClosePriceExtractor):
    """Extractor for DataFrames with 'Close' in the last level of a MultiIndex."""

    def can_handle(self, data: pd.DataFrame) -> bool:
        """Check for MultiIndex with 'Close' in the last level."""
        return isinstance(data.columns, pd.MultiIndex) and "Close" in data.columns.get_level_values(
            -1
        )

    def extract_close_prices(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Extract close prices from the last level of the MultiIndex."""
        try:
            extracted = data.xs("Close", level=-1, axis=1)
            return extracted if isinstance(extracted, pd.DataFrame) else extracted.to_frame()
        except Exception as e:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(f"Error extracting Close prices from last level: {e}")
            return None


class ClosePriceExtractorFactory:
    """Factory for selecting the appropriate close price extractor."""

    def __init__(self):
        self._extractors = [
            MultiIndexFieldClosePriceExtractor(),
            SimpleColumnClosePriceExtractor(),
            MultiIndexLastLevelClosePriceExtractor(),
        ]

    def get_extractor(self, data: pd.DataFrame) -> Optional[IClosePriceExtractor]:
        """
        Get the appropriate close price extractor for the given DataFrame.

        Args:
            data: DataFrame to find extractor for

        Returns:
            Appropriate close price extractor, or None if no suitable extractor is found
        """
        if data.empty:
            return None

        for extractor in self._extractors:
            if extractor.can_handle(data):
                return extractor

        if logger.isEnabledFor(logging.WARNING):
            logger.warning("Could not find a suitable close price extractor")

        return None
