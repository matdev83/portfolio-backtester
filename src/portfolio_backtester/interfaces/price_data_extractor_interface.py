"""
Price data extractor interface for polymorphic price data processing.

This module provides interfaces for extracting close prices from different
DataFrame formats in a polymorphic way, supporting both simple columns
and multi-level index structures.
"""

from abc import ABC, abstractmethod
import pandas as pd


class IPriceDataExtractor(ABC):
    """Interface for extracting close prices from price data."""

    @abstractmethod
    def extract_close_prices(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract close prices from price data DataFrame.

        Args:
            price_data: DataFrame containing price data

        Returns:
            DataFrame with close prices
        """
        pass


class MultiIndexFieldPriceExtractor(IPriceDataExtractor):
    """Extractor for DataFrames with Field-level MultiIndex columns."""

    def extract_close_prices(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Extract close prices from MultiIndex DataFrame with 'Field' level."""
        result = price_data.xs("Close", level="Field", axis=1)
        # Ensure we return a DataFrame (xs can return Series if only one column)
        if isinstance(result, pd.Series):
            return result.to_frame()
        return result


class SimpleColumnPriceExtractor(IPriceDataExtractor):
    """Extractor for DataFrames with simple column structure."""

    def extract_close_prices(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Return the price data as-is for simple column structure."""
        return price_data


class PriceDataExtractorFactory:
    """Factory for creating appropriate price data extractors."""

    @staticmethod
    def create_extractor(price_data: pd.DataFrame) -> IPriceDataExtractor:
        """
        Create appropriate price data extractor based on DataFrame structure.

        Args:
            price_data: DataFrame to analyze

        Returns:
            Appropriate IPriceDataExtractor implementation
        """
        if isinstance(price_data.columns, pd.MultiIndex):
            # Check if 'Field' level exists
            if "Field" in price_data.columns.names:
                return MultiIndexFieldPriceExtractor()

        # Default to simple column extractor
        return SimpleColumnPriceExtractor()
