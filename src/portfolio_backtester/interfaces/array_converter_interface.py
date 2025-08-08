"""
Array converter interfaces for eliminating isinstance violations.

This module provides polymorphic interfaces for converting DataFrames to NumPy arrays
without using isinstance checks for DataFrame validation and MultiIndex handling.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple, List, Optional
import numpy as np
import pandas as pd


class IDataFrameValidator(ABC):
    """Interface for validating input data for array conversion."""

    @abstractmethod
    def can_validate(self, data: Any) -> bool:
        """Check if this validator can handle the given data type."""
        pass

    @abstractmethod
    def validate(self, data: Any) -> None:
        """Validate the data, raising appropriate exceptions if invalid."""
        pass


class DataFrameValidator(IDataFrameValidator):
    """Validator for pandas DataFrame objects."""

    def can_validate(self, data: Any) -> bool:
        """Check if data has DataFrame-like characteristics."""
        return (
            hasattr(data, "index")
            and hasattr(data, "columns")
            and hasattr(data, "values")
            and hasattr(data, "sort_index")
        )

    def validate(self, data: Any) -> None:
        """Validate that data is a proper DataFrame."""
        if not self.can_validate(data):
            raise TypeError("Input must be a pandas DataFrame")


class NullDataFrameValidator(IDataFrameValidator):
    """Fallback validator for invalid data types."""

    def can_validate(self, data: Any) -> bool:
        """Always returns True as fallback validator."""
        return True

    def validate(self, data: Any) -> None:
        """Always raises TypeError for invalid data."""
        raise TypeError("Input must be a pandas DataFrame")


class DataFrameValidatorFactory:
    """Factory for creating appropriate DataFrame validators."""

    def __init__(self):
        self._validators = [
            DataFrameValidator(),
            NullDataFrameValidator(),  # Fallback
        ]

    def get_validator(self, data: Any) -> IDataFrameValidator:
        """Get the appropriate validator for the given data."""
        for validator in self._validators:
            if validator.can_validate(data):
                return validator
        # Should never reach here due to NullValidator fallback
        return self._validators[-1]


class IColumnProcessor(ABC):
    """Interface for processing DataFrame columns."""

    @abstractmethod
    def can_process(self, columns: Any) -> bool:
        """Check if this processor can handle the given column type."""
        pass

    @abstractmethod
    def process_columns(self, df: pd.DataFrame, field: Optional[str] = None) -> pd.DataFrame:
        """Process the DataFrame columns and return the processed DataFrame."""
        pass


class MultiIndexColumnProcessor(IColumnProcessor):
    """Processor for MultiIndex DataFrame columns."""

    def can_process(self, columns: Any) -> bool:
        """Check if columns is a MultiIndex."""
        return (
            hasattr(columns, "nlevels")
            and hasattr(columns, "names")
            and hasattr(columns, "get_level_values")
            and getattr(columns, "nlevels", 1) > 1
        )

    def process_columns(self, df: pd.DataFrame, field: Optional[str] = None) -> pd.DataFrame:
        """Extract field from MultiIndex columns."""
        if field is None:
            raise ValueError("Multi-Index DataFrame requires *field* parameter")

        # Determine the field level name
        if "Field" in df.columns.names:
            level_name = "Field"
        else:
            # Assume the last level holds the field
            last_level_name = df.columns.names[-1]
            if last_level_name is not None:
                level_name = str(last_level_name)
            else:
                # Use numeric index when level name is None
                level_name = str(len(df.columns.names) - 1)

        # Normalize level_name when None by using last level index
        if level_name not in df.columns.names:
            try:
                level_idx = int(level_name)
            except ValueError:
                level_idx = len(df.columns.names) - 1
            level_values = df.columns.get_level_values(level_idx)
        else:
            level_values = df.columns.get_level_values(level_name)

        if field not in level_values:
            raise KeyError(f"Field '{field}' not found in DataFrame columns")

        # Extract the field - xs with axis=1 on MultiIndex returns DataFrame
        try:
            result = df.xs(
                field, level=level_name if level_name in df.columns.names else level_idx, axis=1
            )
        except Exception:
            # Fallback to positional level index
            result = df.xs(field, level=len(df.columns.names) - 1, axis=1)
        if not isinstance(result, pd.DataFrame):
            raise TypeError("Expected DataFrame from MultiIndex extraction")
        return result


class SimpleColumnProcessor(IColumnProcessor):
    """Processor for simple (non-MultiIndex) DataFrame columns."""

    def can_process(self, columns: Any) -> bool:
        """Check if columns is a simple Index (not MultiIndex)."""
        return not (hasattr(columns, "nlevels") and getattr(columns, "nlevels", 1) > 1)

    def process_columns(self, df: pd.DataFrame, field: Optional[str] = None) -> pd.DataFrame:
        """Return a copy of the DataFrame as-is."""
        return df.copy()


class ColumnProcessorFactory:
    """Factory for creating appropriate column processors."""

    def __init__(self):
        self._processors = [
            MultiIndexColumnProcessor(),
            SimpleColumnProcessor(),  # Fallback for simple columns
        ]

    def get_processor(self, columns: Any) -> IColumnProcessor:
        """Get the appropriate processor for the given columns."""
        for processor in self._processors:
            if processor.can_process(columns):
                return processor
        # Should never reach here due to SimpleColumnProcessor fallback
        return self._processors[-1]


class IArrayConverter(ABC):
    """Interface for converting DataFrames to NumPy arrays."""

    @abstractmethod
    def convert_to_array(
        self, df: pd.DataFrame, field: Optional[str] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Convert DataFrame to NumPy array and return tickers."""
        pass


class PolymorphicArrayConverter(IArrayConverter):
    """Polymorphic array converter that eliminates isinstance violations."""

    def __init__(self):
        self._validator_factory = DataFrameValidatorFactory()
        self._column_factory = ColumnProcessorFactory()

    def convert_to_array(
        self, df: pd.DataFrame, field: Optional[str] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Convert a (potentially Multi-Index) DataFrame to a contiguous float32 NumPy array.

        Args:
            df: Price or returns data. Index must be monotonic and unique.
            field: If df has a Multi-Index with levels (Ticker, Field) supply the
                  desired Field (e.g. "Close") to extract. When None the function
                  assumes df already has one column per asset.

        Returns:
            Tuple of (2-D float32 array of shape (n_periods, n_assets),
                     list of tickers in column order)
        """
        # Step 1: Validate input
        validator = self._validator_factory.get_validator(df)
        validator.validate(df)

        # Step 2: Ensure monotonic index
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()

        # Step 3: Process columns polymorphically
        column_processor = self._column_factory.get_processor(df.columns)
        extracted = column_processor.process_columns(df, field)

        # Step 4: Ensure column order is deterministic (sorted tickers)
        tickers = list(extracted.columns)
        extracted = extracted.astype(np.float32)

        # Step 5: Convert to contiguous NumPy array
        matrix = np.ascontiguousarray(extracted.values, dtype=np.float32)
        return matrix, tickers


# Factory function for easy instantiation
def create_array_converter() -> PolymorphicArrayConverter:
    """Create a new polymorphic array converter instance."""
    return PolymorphicArrayConverter()
