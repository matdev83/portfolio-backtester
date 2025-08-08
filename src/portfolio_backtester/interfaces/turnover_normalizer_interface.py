"""
Turnover normalizer interface for polymorphic turnover value processing.

This module provides interfaces for normalizing turnover values from different
data types (Series, DataFrame, scalar) in a polymorphic way.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Union


class ITurnoverNormalizer(ABC):
    """Interface for normalizing turnover values."""

    @abstractmethod
    def normalize_turnover_value(
        self, turnover_value: Union[float, pd.Series, pd.DataFrame]
    ) -> float:
        """
        Normalize turnover value to a scalar float.

        Args:
            turnover_value: Turnover value in various formats

        Returns:
            Normalized scalar turnover value
        """
        pass


class SeriesTurnoverNormalizer(ITurnoverNormalizer):
    """Normalizer for pandas Series turnover values."""

    def normalize_turnover_value(
        self, turnover_value: Union[float, pd.Series, pd.DataFrame]
    ) -> float:
        """Sum Series values to get total turnover."""
        if isinstance(turnover_value, pd.Series):
            return float(turnover_value.sum())
        elif isinstance(turnover_value, (int, float)):
            return float(turnover_value)
        else:
            # Handle unexpected types gracefully
            return 0.0


class DataFrameTurnoverNormalizer(ITurnoverNormalizer):
    """Normalizer for pandas DataFrame turnover values."""

    def normalize_turnover_value(
        self, turnover_value: Union[float, pd.Series, pd.DataFrame]
    ) -> float:
        """Sum DataFrame values to get total turnover."""
        if isinstance(turnover_value, pd.DataFrame):
            return float(turnover_value.sum().sum())
        elif isinstance(turnover_value, (int, float)):
            return float(turnover_value)
        else:
            # Handle unexpected types gracefully
            return 0.0


class ScalarTurnoverNormalizer(ITurnoverNormalizer):
    """Normalizer for scalar turnover values."""

    def normalize_turnover_value(
        self, turnover_value: Union[float, pd.Series, pd.DataFrame]
    ) -> float:
        """Return scalar value as-is."""
        if isinstance(turnover_value, (int, float)):
            return float(turnover_value)
        else:
            # Handle unexpected types gracefully
            return 0.0


class TurnoverNormalizerFactory:
    """Factory for creating appropriate turnover normalizers."""

    @staticmethod
    def create_normalizer(
        turnover_value: Union[float, pd.Series, pd.DataFrame],
    ) -> ITurnoverNormalizer:
        """
        Create appropriate turnover normalizer based on value type.

        Args:
            turnover_value: Turnover value to analyze

        Returns:
            Appropriate ITurnoverNormalizer implementation
        """
        if isinstance(turnover_value, pd.Series):
            return SeriesTurnoverNormalizer()
        elif isinstance(turnover_value, pd.DataFrame):
            return DataFrameTurnoverNormalizer()
        else:
            return ScalarTurnoverNormalizer()
