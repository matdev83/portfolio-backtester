"""
Data hashing interfaces for polymorphic value handling.

This module provides interfaces and implementations to replace isinstance
violations in data hashing operations, enabling extensible and testable
value conversion strategies.
"""

from abc import ABC, abstractmethod
from typing import Any
import pandas as pd


class IValueConverter(ABC):
    """Interface for converting values to string representation for hashing."""

    @abstractmethod
    def can_handle(self, value: Any) -> bool:
        """
        Check if this converter can handle the given value type.

        Args:
            value: Value to check

        Returns:
            True if this converter can handle the value, False otherwise
        """
        pass

    @abstractmethod
    def convert_to_string(self, value: Any) -> str:
        """
        Convert value to string representation for hashing.

        Args:
            value: Value to convert

        Returns:
            String representation of the value
        """
        pass


class BytesValueConverter(IValueConverter):
    """Converter for bytes and bytearray values."""

    def can_handle(self, value: Any) -> bool:
        """Check if value is bytes or bytearray."""
        return isinstance(value, (bytes, bytearray))

    def convert_to_string(self, value: Any) -> str:
        """Convert bytes/bytearray to string with error handling."""
        if isinstance(value, (bytes, bytearray)):
            return value.decode(errors="ignore")
        # Fallback to str() for type safety (should not happen in practice)
        return str(value)


class DefaultValueConverter(IValueConverter):
    """Default converter for all other value types."""

    def can_handle(self, value: Any) -> bool:
        """Can handle any value type as fallback."""
        return True

    def convert_to_string(self, value: Any) -> str:
        """Convert value to string using str() function."""
        return str(value)


class ValueConverterFactory:
    """Factory class for selecting appropriate value converter."""

    def __init__(self):
        self._converters = [
            BytesValueConverter(),
            DefaultValueConverter(),  # Must be last as it handles all types
        ]

    def get_converter(self, value: Any) -> IValueConverter:
        """
        Get appropriate converter for the given value.

        Args:
            value: Value to find converter for

        Returns:
            Appropriate value converter
        """
        for converter in self._converters:
            if converter.can_handle(value):
                return converter

        # This should never happen since DefaultValueConverter handles all types
        return self._converters[-1]


class IDataHasher(ABC):
    """Interface for generating hash keys from data."""

    @abstractmethod
    def get_data_hash(self, data: pd.DataFrame, identifier: str) -> str:
        """
        Generate a hash for DataFrame to use as cache key.

        Args:
            data: DataFrame to hash
            identifier: Additional identifier for uniqueness

        Returns:
            Hash string for the data
        """
        pass

    @abstractmethod
    def get_window_hash(
        self, data: pd.DataFrame, window_start: pd.Timestamp, window_end: pd.Timestamp
    ) -> str:
        """
        Generate a hash for window-specific data.

        Args:
            data: DataFrame to hash
            window_start: Start of the window
            window_end: End of the window

        Returns:
            Hash string for the windowed data
        """
        pass


class PolymorphicDataHasher(IDataHasher):
    """
    Polymorphic implementation of data hasher using value converter strategy.

    This implementation replaces isinstance violations with polymorphic
    value conversion strategies.
    """

    def __init__(self):
        self._converter_factory = ValueConverterFactory()

    def _safe_value_to_string(self, value: Any) -> str:
        """
        Safely convert a value to string for hashing using polymorphic strategy.

        Args:
            value: Value to convert to string

        Returns:
            String representation of the value
        """
        converter = self._converter_factory.get_converter(value)
        return converter.convert_to_string(value)

    def get_data_hash(self, data: pd.DataFrame, identifier: str) -> str:
        """
        Generate a hash for DataFrame to use as cache key.

        Args:
            data: DataFrame to hash
            identifier: Additional identifier for uniqueness

        Returns:
            Hash string for the data
        """
        import hashlib

        # Create a hash based on data shape, index, columns, and sample values
        hash_input = f"{identifier}_{data.shape}_{data.index.min()}_{data.index.max()}"

        first_val = data.iloc[0, 0] if not data.empty else "empty"
        first_val_str = self._safe_value_to_string(first_val)
        hash_input += f"_{list(data.columns)}_{first_val_str}"

        if len(data) > 1:
            last_val = data.iloc[-1, 0]
            last_val_str = self._safe_value_to_string(last_val)
            hash_input += f"_{last_val_str}"

        return hashlib.md5(hash_input.encode()).hexdigest()

    def get_window_hash(
        self, data: pd.DataFrame, window_start: pd.Timestamp, window_end: pd.Timestamp
    ) -> str:
        """
        Generate a hash for window-specific data.

        Args:
            data: DataFrame to hash
            window_start: Start of the window
            window_end: End of the window

        Returns:
            Hash string for the windowed data
        """
        import hashlib

        hash_input = f"window_{window_start}_{window_end}_{data.shape}"

        first_val = data.iloc[0, 0] if not data.empty else "empty"
        first_val_str = self._safe_value_to_string(first_val)
        hash_input += f"_{list(data.columns)}_{first_val_str}"

        if len(data) > 1:
            last_val = data.iloc[-1, 0]
            last_val_str = self._safe_value_to_string(last_val)
            hash_input += f"_{last_val_str}"

        return hashlib.md5(hash_input.encode()).hexdigest()


class DataHasherFactory:
    """Factory for creating appropriate data hasher implementation."""

    @staticmethod
    def create_hasher() -> IDataHasher:
        """
        Create appropriate data hasher implementation.

        Returns:
            Data hasher instance
        """
        return PolymorphicDataHasher()
