"""
OHLCNormalizer interface and implementations for polymorphic OHLC data handling.

Replaces isinstance checks for DataFrame column structure with proper polymorphic behavior.
"""

from abc import ABC, abstractmethod
from typing import List, Set
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class IOHLCNormalizer(ABC):
    """Abstract interface for normalizing and extracting information from OHLC data."""

    @abstractmethod
    def get_available_tickers(self, df: pd.DataFrame) -> Set[str]:
        """
        Extract available ticker symbols from DataFrame columns.

        Args:
            df: DataFrame with price data

        Returns:
            Set of available ticker symbols
        """
        pass

    @abstractmethod
    def extract_ticker_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Extract data for a specific ticker from the DataFrame.

        Args:
            df: DataFrame with price data
            ticker: Ticker symbol to extract

        Returns:
            DataFrame with ticker-specific data

        Raises:
            KeyError: If ticker is not available in the DataFrame
        """
        pass

    @abstractmethod
    def validate_ticker_presence(self, df: pd.DataFrame, tickers: List[str]) -> List[str]:
        """
        Validate which tickers are present in the DataFrame.

        Args:
            df: DataFrame with price data
            tickers: List of ticker symbols to check

        Returns:
            List of tickers that are missing from the DataFrame
        """
        pass


class MultiIndexOHLCNormalizer(IOHLCNormalizer):
    """Handles OHLC data with MultiIndex columns (Ticker, OHLC)."""

    def get_available_tickers(self, df: pd.DataFrame) -> Set[str]:
        """Get available tickers from MultiIndex columns."""
        if not isinstance(df.columns, pd.MultiIndex):
            raise ValueError("Expected MultiIndex columns, got regular Index")

        return set(df.columns.get_level_values("Ticker").unique())

    def extract_ticker_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Extract ticker data from MultiIndex DataFrame."""
        if not isinstance(df.columns, pd.MultiIndex):
            raise ValueError("Expected MultiIndex columns, got regular Index")

        try:
            result = df.xs(ticker, level="Ticker", axis=1, drop_level=False)
            if isinstance(result, pd.DataFrame):
                return result
            else:
                # Convert Series to DataFrame if needed
                return pd.DataFrame(result)
        except KeyError:
            raise KeyError(f"Ticker '{ticker}' not found in MultiIndex DataFrame")

    def validate_ticker_presence(self, df: pd.DataFrame, tickers: List[str]) -> List[str]:
        """Validate ticker presence in MultiIndex DataFrame."""
        available_tickers = self.get_available_tickers(df)
        return [ticker for ticker in tickers if ticker not in available_tickers]


class FlatOHLCNormalizer(IOHLCNormalizer):
    """Handles OHLC data with flat column structure (ticker names as columns)."""

    def get_available_tickers(self, df: pd.DataFrame) -> Set[str]:
        """Get available tickers from flat column structure."""
        if isinstance(df.columns, pd.MultiIndex):
            raise ValueError("Expected flat columns, got MultiIndex")

        return set(df.columns)

    def extract_ticker_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Extract ticker data from flat DataFrame."""
        if isinstance(df.columns, pd.MultiIndex):
            raise ValueError("Expected flat columns, got MultiIndex")

        if ticker not in df.columns:
            raise KeyError(f"Ticker '{ticker}' not found in flat DataFrame")

        return df[[ticker]]

    def validate_ticker_presence(self, df: pd.DataFrame, tickers: List[str]) -> List[str]:
        """Validate ticker presence in flat DataFrame."""
        available_tickers = self.get_available_tickers(df)
        return [ticker for ticker in tickers if ticker not in available_tickers]


class OHLCNormalizerFactory:
    """Factory for creating appropriate OHLC normalizers based on DataFrame structure."""

    @staticmethod
    def create_normalizer(df: pd.DataFrame) -> IOHLCNormalizer:
        """
        Create appropriate OHLC normalizer based on DataFrame column structure.

        Args:
            df: DataFrame to analyze

        Returns:
            Appropriate IOHLCNormalizer implementation
        """
        if isinstance(df.columns, pd.MultiIndex):
            return MultiIndexOHLCNormalizer()
        else:
            return FlatOHLCNormalizer()
