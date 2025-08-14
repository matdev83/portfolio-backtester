"""
Price extractor interfaces for data sources to eliminate isinstance violations.

This module provides polymorphic interfaces for extracting price data from
DataFrames with different column structures.
"""

from abc import ABC, abstractmethod
from typing import Any
import pandas as pd


class IPriceExtractor(ABC):
    """Interface for extracting price data from DataFrames."""

    @abstractmethod
    def can_extract(self, columns: Any) -> bool:
        """Check if this extractor can handle the given column structure."""
        pass

    @abstractmethod
    def extract_close_price(self, df: pd.DataFrame, ticker: str) -> pd.Series:
        """Extract close price series for the given ticker."""
        pass


class MultiIndexPriceExtractor(IPriceExtractor):
    """Price extractor for MultiIndex DataFrames."""

    def can_extract(self, columns: Any) -> bool:
        """Check if columns is a MultiIndex using duck typing."""
        return (
            hasattr(columns, "nlevels")
            and hasattr(columns, "names")
            and hasattr(columns, "get_level_values")
            and getattr(columns, "nlevels", 1) > 1
        )

    def extract_close_price(self, df: pd.DataFrame, ticker: str) -> pd.Series:
        """Extract close price from MultiIndex DataFrame."""
        return df[(ticker, "Close")]


class SimplePriceExtractor(IPriceExtractor):
    """Price extractor for simple column DataFrames."""

    def can_extract(self, columns: Any) -> bool:
        """Check if columns is a simple Index using duck typing."""
        return hasattr(columns, "__contains__") and hasattr(columns, "tolist")

    def extract_close_price(self, df: pd.DataFrame, ticker: str) -> pd.Series:
        """Extract close price from simple DataFrame."""
        return df[ticker]


class PriceExtractorFactory:
    """Factory for creating appropriate price extractors."""

    def __init__(self) -> None:
        self._extractors = [
            MultiIndexPriceExtractor(),
            SimplePriceExtractor(),  # Fallback
        ]

    def get_extractor(self, columns: Any) -> IPriceExtractor:
        """Get the appropriate extractor for the given columns."""
        for extractor in self._extractors:
            if extractor.can_extract(columns):
                return extractor
        # Should never reach here due to SimplePriceExtractor fallback
        return self._extractors[-1]


class PolymorphicPriceExtractor:
    """Polymorphic price extractor that eliminates isinstance violations."""

    def __init__(self) -> None:
        self._factory = PriceExtractorFactory()

    def extract_close_price(self, df: pd.DataFrame, ticker: str) -> pd.Series:
        """
        Extract close price series using polymorphic extraction.

        Args:
            df: DataFrame containing price data
            ticker: Ticker symbol to extract

        Returns:
            Series of close prices for the ticker

        Raises:
            KeyError: If ticker or close price data not found
        """
        extractor = self._factory.get_extractor(df.columns)
        return extractor.extract_close_price(df, ticker)


# Factory function for easy instantiation
def create_price_extractor() -> PolymorphicPriceExtractor:
    """Create a new polymorphic price extractor instance."""
    return PolymorphicPriceExtractor()
