"""
Data hashing utilities for cache key generation.

This module provides hash generation functionality for DataFrames to create
unique cache keys for data preprocessing operations.
"""

import pandas as pd
from typing import Any
import hashlib

from ..interfaces.data_hasher_interface import ValueConverterFactory


class DataHasher:
    """
    Generates hash keys for DataFrame data to use in caching operations.

    This class now uses polymorphic interfaces instead of isinstance violations,
    maintaining backward compatibility while improving extensibility.
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
        hash_input = f"window_{window_start}_{window_end}_{data.shape}"

        first_val = data.iloc[0, 0] if not data.empty else "empty"
        first_val_str = self._safe_value_to_string(first_val)
        hash_input += f"_{list(data.columns)}_{first_val_str}"

        if len(data) > 1:
            last_val = data.iloc[-1, 0]
            last_val_str = self._safe_value_to_string(last_val)
            hash_input += f"_{last_val_str}"

        return hashlib.md5(hash_input.encode()).hexdigest()
