"""
Column Handler Interface

This module provides polymorphic interfaces for handling different DataFrame column types,
replacing isinstance checks with extensible strategy pattern implementations.

Key interfaces:
- IColumnHandler: Core interface for column type detection and operations
- IColumnExtractor: Interface for extracting specific data from columns
- IUniverseExtractor: Interface for extracting universe tickers from columns

This eliminates isinstance violations while maintaining full backward compatibility.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class IColumnHandler(ABC):
    """
    Interface for handling DataFrame column operations.

    Replaces isinstance checks for MultiIndex vs single-level columns with
    polymorphic behavior based on the Strategy pattern.
    """

    @abstractmethod
    def is_multiindex(self) -> bool:
        """
        Check if columns are MultiIndex type.

        Returns:
            True if columns are MultiIndex, False otherwise
        """
        pass

    @abstractmethod
    def extract_close_prices(
        self, dataframe: pd.DataFrame, price_column: str = "Close"
    ) -> pd.DataFrame:
        """
        Extract close prices from the DataFrame based on column structure.

        Args:
            dataframe: DataFrame to extract from
            price_column: Name of price column to extract

        Returns:
            DataFrame with close prices
        """
        pass

    @abstractmethod
    def get_universe_tickers(self, dataframe: pd.DataFrame) -> List[str]:
        """
        Extract universe ticker symbols from DataFrame columns.

        Args:
            dataframe: DataFrame to extract tickers from

        Returns:
            List of ticker symbols
        """
        pass

    @abstractmethod
    def extract_ticker_data(
        self, dataframe: pd.DataFrame, ticker: str, field: str = "Close"
    ) -> Optional[pd.Series]:
        """
        Extract data for a specific ticker and field.

        Args:
            dataframe: DataFrame to extract from
            ticker: Ticker symbol to extract
            field: Data field to extract (default: "Close")

        Returns:
            Series with ticker data, or None if not found
        """
        pass

    @abstractmethod
    def validate_ticker_availability(self, dataframe: pd.DataFrame, ticker: str) -> bool:
        """
        Check if a ticker is available in the DataFrame columns.

        Args:
            dataframe: DataFrame to check
            ticker: Ticker symbol to validate

        Returns:
            True if ticker is available, False otherwise
        """
        pass


class MultiIndexColumnHandler(IColumnHandler):
    """Handler for DataFrames with MultiIndex columns."""

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize handler with DataFrame.

        Args:
            dataframe: DataFrame with MultiIndex columns
        """
        self.dataframe = dataframe
        # Use polymorphic column type detector to eliminate isinstance violations
        from .column_type_detector import create_column_type_detector

        detector = create_column_type_detector()
        detector.validate_multiindex_requirement(dataframe)

    def is_multiindex(self) -> bool:
        """Always returns True for MultiIndex handler."""
        return True

    def extract_close_prices(
        self, dataframe: pd.DataFrame, price_column: str = "Close"
    ) -> pd.DataFrame:
        """Extract close prices from MultiIndex DataFrame."""
        try:
            # Use xs to extract the price column across all tickers
            result = dataframe.xs(price_column, level="Field", axis=1)
            # Ensure we always return a DataFrame using polymorphic normalization
            from .column_type_detector import create_data_structure_handler

            handler = create_data_structure_handler()
            return handler.ensure_dataframe(result, name=price_column)
        except KeyError:
            # If "Field" level doesn't exist, try direct access
            close_data = {}
            for ticker in self.get_universe_tickers(dataframe):
                if (ticker, price_column) in dataframe.columns:
                    close_data[ticker] = dataframe[(ticker, price_column)]
            return pd.DataFrame(close_data)

    def get_universe_tickers(self, dataframe: pd.DataFrame) -> List[str]:
        """Extract tickers from MultiIndex columns."""
        try:
            # Try to get from "Ticker" level first
            return dataframe.columns.get_level_values("Ticker").unique().tolist()
        except KeyError:
            # If no "Ticker" level, get from level 0
            return dataframe.columns.get_level_values(0).unique().tolist()

    def extract_ticker_data(
        self, dataframe: pd.DataFrame, ticker: str, field: str = "Close"
    ) -> Optional[pd.Series]:
        """Extract specific ticker data from MultiIndex DataFrame."""
        try:
            # Try direct column access first
            if (ticker, field) in dataframe.columns:
                return dataframe[(ticker, field)]

            # Try ticker-level access with xs
            if ticker in dataframe.columns.get_level_values(0):
                ticker_data = dataframe.xs(ticker, level=0, axis=1)
                if field in ticker_data.columns:
                    return ticker_data[field]

            logger.warning(f"Ticker {ticker} with field {field} not found in MultiIndex columns")
            return None

        except Exception as e:
            logger.error(f"Error extracting {ticker} data: {e}")
            return None

    def validate_ticker_availability(self, dataframe: pd.DataFrame, ticker: str) -> bool:
        """Check if ticker exists in MultiIndex columns."""
        try:
            # Check level 0 for ticker names
            level_0_values = dataframe.columns.get_level_values(0)
            return ticker in level_0_values
        except Exception:
            return False


class SingleIndexColumnHandler(IColumnHandler):
    """Handler for DataFrames with single-level columns."""

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize handler with DataFrame.

        Args:
            dataframe: DataFrame with single-level columns
        """
        self.dataframe = dataframe
        if isinstance(dataframe.columns, pd.MultiIndex):
            raise ValueError(
                "SingleIndexColumnHandler requires DataFrame with single-level columns"
            )

    def is_multiindex(self) -> bool:
        """Always returns False for single-level handler."""
        return False

    def extract_close_prices(
        self, dataframe: pd.DataFrame, price_column: str = "Close"
    ) -> pd.DataFrame:
        """For single-level columns, assume columns are already close prices."""
        return dataframe

    def get_universe_tickers(self, dataframe: pd.DataFrame) -> List[str]:
        """Get column names as ticker symbols."""
        return dataframe.columns.tolist()

    def extract_ticker_data(
        self, dataframe: pd.DataFrame, ticker: str, field: str = "Close"
    ) -> Optional[pd.Series]:
        """Extract ticker data from single-level DataFrame."""
        if ticker in dataframe.columns:
            return dataframe[ticker]
        else:
            logger.warning(f"Ticker {ticker} not found in single-level columns")
            return None

    def validate_ticker_availability(self, dataframe: pd.DataFrame, ticker: str) -> bool:
        """Check if ticker exists in single-level columns."""
        return ticker in dataframe.columns


class IColumnExtractor(ABC):
    """Interface for extracting specific column data based on column type."""

    @abstractmethod
    def extract_field(
        self, dataframe: pd.DataFrame, ticker: str, field: str
    ) -> Optional[pd.Series]:
        """
        Extract field data for a specific ticker.

        Args:
            dataframe: Source DataFrame
            ticker: Ticker symbol
            field: Field name to extract

        Returns:
            Series with field data or None if not found
        """
        pass

    @abstractmethod
    def extract_all_tickers_field(self, dataframe: pd.DataFrame, field: str) -> pd.DataFrame:
        """
        Extract field data for all tickers.

        Args:
            dataframe: Source DataFrame
            field: Field name to extract

        Returns:
            DataFrame with field data for all tickers
        """
        pass


class ColumnHandlerFactory:
    """Factory for creating appropriate column handlers based on DataFrame structure."""

    @staticmethod
    def create(dataframe: pd.DataFrame) -> IColumnHandler:
        """
        Create appropriate column handler based on DataFrame column structure.

        Args:
            dataframe: DataFrame to analyze

        Returns:
            Appropriate column handler instance
        """
        if isinstance(dataframe.columns, pd.MultiIndex):
            return MultiIndexColumnHandler(dataframe)
        else:
            return SingleIndexColumnHandler(dataframe)

    @staticmethod
    def detect_column_type(dataframe: pd.DataFrame) -> str:
        """
        Detect column type for logging/debugging purposes.

        Args:
            dataframe: DataFrame to analyze

        Returns:
            String description of column type
        """
        if isinstance(dataframe.columns, pd.MultiIndex):
            return f"MultiIndex with {dataframe.columns.nlevels} levels"
        else:
            return f"SingleIndex with {len(dataframe.columns)} columns"


class IUniverseExtractor(ABC):
    """Interface for extracting universe information from DataFrames."""

    @abstractmethod
    def extract_universe(self, dataframe: pd.DataFrame) -> List[str]:
        """
        Extract universe ticker symbols from DataFrame.

        Args:
            dataframe: DataFrame to extract from

        Returns:
            List of universe ticker symbols
        """
        pass

    @abstractmethod
    def validate_universe_data(
        self, dataframe: pd.DataFrame, required_tickers: List[str]
    ) -> tuple[bool, List[str]]:
        """
        Validate that all required tickers are available.

        Args:
            dataframe: DataFrame to validate
            required_tickers: List of required ticker symbols

        Returns:
            Tuple of (is_valid, missing_tickers)
        """
        pass


class UniverseExtractor(IUniverseExtractor):
    """Standard implementation of universe extraction."""

    def __init__(self, column_handler: IColumnHandler):
        """
        Initialize with column handler.

        Args:
            column_handler: Column handler for the DataFrame
        """
        self.column_handler = column_handler

    def extract_universe(self, dataframe: pd.DataFrame) -> List[str]:
        """Extract universe using the column handler."""
        return self.column_handler.get_universe_tickers(dataframe)

    def validate_universe_data(
        self, dataframe: pd.DataFrame, required_tickers: List[str]
    ) -> tuple[bool, List[str]]:
        """Validate universe data availability."""
        available_tickers = self.extract_universe(dataframe)
        missing_tickers = [ticker for ticker in required_tickers if ticker not in available_tickers]
        return len(missing_tickers) == 0, missing_tickers


class UniverseExtractorFactory:
    """Factory for creating universe extractors."""

    @staticmethod
    def create(dataframe: pd.DataFrame) -> IUniverseExtractor:
        """
        Create universe extractor for DataFrame.

        Args:
            dataframe: DataFrame to create extractor for

        Returns:
            Universe extractor instance
        """
        column_handler = ColumnHandlerFactory.create(dataframe)
        return UniverseExtractor(column_handler)
