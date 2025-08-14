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
        """Check if data is a pandas DataFrame."""
        return isinstance(data, pd.DataFrame)

    def validate(self, data: Any) -> None:
        """Validate that data is a pandas DataFrame."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Input must be a pandas DataFrame, got {type(data)}")


class MultiIndexDataFrameValidator(IDataFrameValidator):
    """Validator for pandas DataFrame objects with MultiIndex columns."""

    def can_validate(self, data: Any) -> bool:
        """Check if data is a pandas DataFrame with MultiIndex columns."""
        return isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex)

    def validate(self, data: Any) -> None:
        """Validate that data is a pandas DataFrame with MultiIndex columns."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Input must be a pandas DataFrame, got {type(data)}")
        if not isinstance(data.columns, pd.MultiIndex):
            raise TypeError("DataFrame must have MultiIndex columns")


class SimpleDataFrameValidator(IDataFrameValidator):
    """Validator for pandas DataFrame objects with simple columns."""

    def can_validate(self, data: Any) -> bool:
        """Check if data is a pandas DataFrame with simple columns."""
        return isinstance(data, pd.DataFrame) and not isinstance(data.columns, pd.MultiIndex)

    def validate(self, data: Any) -> None:
        """Validate that data is a pandas DataFrame with simple columns."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Input must be a pandas DataFrame, got {type(data)}")
        if isinstance(data.columns, pd.MultiIndex):
            raise TypeError("DataFrame must not have MultiIndex columns")


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
                field,
                level=level_name if level_name in df.columns.names else level_idx,
                axis=1,
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


class UniversalColumnProcessor(IColumnProcessor):
    """Universal processor that can handle any column type."""

    def can_process(self, columns: Any) -> bool:
        """Always returns True - universal fallback processor."""
        return True

    def process_columns(self, df: pd.DataFrame, field: Optional[str] = None) -> pd.DataFrame:
        """Process columns using duck typing approach."""
        # For MultiIndex columns, try to extract field if provided
        if hasattr(df.columns, "nlevels") and getattr(df.columns, "nlevels", 1) > 1:
            if field is not None:
                try:
                    # Try to extract the field using xs
                    result = df.xs(field, level=-1, axis=1)
                    if isinstance(result, pd.DataFrame):
                        return result
                except Exception:
                    pass

        # For simple columns or when field extraction fails, return a copy
        return df.copy()


class ColumnProcessorFactory:
    """Factory for creating appropriate column processors."""

    def __init__(self) -> None:
        self._processors = [
            MultiIndexColumnProcessor(),
            SimpleColumnProcessor(),  # Fallback for simple column structures
            UniversalColumnProcessor(),  # Universal fallback
        ]

    def get_processor(self, columns: Any) -> IColumnProcessor:
        """Get the appropriate processor for the given columns."""
        for processor in self._processors:
            if processor.can_process(columns):
                return processor
        # Should never reach here due to UniversalColumnProcessor fallback
        return self._processors[-1]


class IArrayConverter(ABC):
    """Interface for converting DataFrames to NumPy arrays."""

    @abstractmethod
    def convert_to_array(
        self, df: pd.DataFrame, field: Optional[str] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Convert DataFrame to NumPy array and return tickers."""
        pass


class PolymorphicArrayConverter:
    """Polymorphic array converter that eliminates isinstance violations."""

    def __init__(self) -> None:
        # Initialize factories with dependency injection
        self._validator = DataFrameValidator()
        self._processor_factory = ColumnProcessorFactory()

    def convert_to_array(
        self, data: Any, column_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Convert data to numpy array using polymorphic validation and processing."""
        # Step 1: Validate data type using polymorphic validator
        self._validator.validate(data)

        # Step 2: Process columns using polymorphic processor
        if isinstance(data, pd.DataFrame):
            if not data.index.is_monotonic_increasing:
                data = data.sort_index()
            processor = self._processor_factory.get_processor(data.columns)
            # Convert column_names to a single field if needed
            field = column_names[0] if column_names and len(column_names) > 0 else None
            processed_data = processor.process_columns(data, field)
        else:
            processed_data = data

        # Step 3: Convert to numpy array and extract tickers
        if isinstance(processed_data, pd.DataFrame):
            return processed_data.values.astype(np.float32), list(processed_data.columns)
        elif isinstance(processed_data, pd.Series):
            # For Series, use the name as the ticker or default to "series"
            ticker = processed_data.name if processed_data.name is not None else "series"
            return processed_data.values.reshape(-1, 1).astype(np.float32), [str(ticker)]
        elif isinstance(processed_data, np.ndarray):
            # For arrays, create dummy tickers
            if processed_data.ndim == 1:
                return processed_data.reshape(-1, 1).astype(np.float32), ["asset_0"]
            else:
                n_assets = processed_data.shape[1] if processed_data.ndim > 1 else 1
                return processed_data.astype(np.float32), [
                    f"asset_{i}" for i in range(n_assets)
                ]
        else:
            # For scalar values or other types, convert to array
            array_data = np.array(processed_data)
            if array_data.ndim == 0:
                return array_data.reshape(1, 1).astype(np.float32), ["scalar"]
            elif array_data.ndim == 1:
                return array_data.reshape(-1, 1).astype(np.float32), ["asset_0"]
            else:
                n_assets = array_data.shape[1] if array_data.ndim > 1 else 1
                return array_data.astype(np.float32), [
                    f"asset_{i}" for i in range(n_assets)
                ]

    def create_converter(self) -> "PolymorphicArrayConverter":
        """Create a new converter instance."""
        return PolymorphicArrayConverter()


# Factory function for easy instantiation
def create_array_converter() -> PolymorphicArrayConverter:
    """Create a new polymorphic array converter instance."""
    return PolymorphicArrayConverter()
