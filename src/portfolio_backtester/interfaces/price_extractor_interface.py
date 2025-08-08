"""
Provides interfaces for polymorphic extraction of price series from various data structures.
"""

from abc import ABC, abstractmethod
from typing import Any
import pandas as pd


class IPriceExtractor(ABC):
    """Interface to extract a pandas Series from a given data object."""

    @abstractmethod
    def can_handle(self, data: Any) -> bool:
        """Returns True if the extractor can handle the data type."""
        pass

    @abstractmethod
    def extract(self, data: Any, universe_tickers: pd.Index) -> pd.Series:
        """Extracts a pandas Series from the data."""
        pass


class DataFrameExtractor(IPriceExtractor):
    """Extracts a Series from a DataFrame."""

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, pd.DataFrame)

    def extract(self, data: pd.DataFrame, universe_tickers: pd.Index) -> pd.Series:
        if data.shape[1] == 1:
            series = data.iloc[:, 0]
        elif "Close" in data.columns:
            series = data["Close"]
        else:
            series = data.iloc[:, 0]
        return series


class SeriesExtractor(IPriceExtractor):
    """Returns the Series if the input is a Series."""

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, pd.Series)

    def extract(self, data: pd.Series, universe_tickers: pd.Index) -> pd.Series:
        return data


class ScalarExtractor(IPriceExtractor):
    """Handles scalar values."""

    def can_handle(self, data: Any) -> bool:
        return not isinstance(data, (pd.DataFrame, pd.Series))

    def extract(self, data: Any, universe_tickers: pd.Index) -> pd.Series:
        if not universe_tickers.empty:
            return pd.Series([float(data)], index=[universe_tickers[0]])
        return pd.Series(dtype=float)


class PriceExtractorFactory:
    """Factory to create a price extractor."""

    def __init__(self):
        # Order is important, ScalarExtractor should be last as a catch-all.
        self._extractors = [DataFrameExtractor(), SeriesExtractor(), ScalarExtractor()]

    def get_extractor(self, data: Any) -> IPriceExtractor:
        """Gets the appropriate extractor for the data."""
        for extractor in self._extractors:
            if extractor.can_handle(data):
                return extractor
        # This should not be reached given the catch-all ScalarExtractor
        raise TypeError(f"No extractor found for data of type {type(data)}")
