"""
Commission parameter handler interface and implementations.

This module provides interfaces and implementations for handling
commission parameters in different formats for backward compatibility.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Union


class ICommissionParameterHandler(ABC):
    """Interface for handling commission parameters."""

    @abstractmethod
    def normalize_commissions(
        self,
        commissions: Union[Dict[str, float], float, int],
        new_weights: pd.Series,
        existing_positions: Dict[str, Any],
    ) -> Dict[str, float]:
        """Normalize commission parameters to a dictionary format."""
        pass


class DictCommissionParameterHandler(ICommissionParameterHandler):
    """Handler for dictionary commission parameters."""

    def normalize_commissions(
        self,
        commissions: Union[Dict[str, float], float, int],
        new_weights: pd.Series,
        existing_positions: Dict[str, Any],
    ) -> Dict[str, float]:
        """Return the dictionary as-is or empty dict if None."""
        if isinstance(commissions, dict):
            return commissions
        return {}


class NumericCommissionParameterHandler(ICommissionParameterHandler):
    """Handler for numeric commission parameters (int/float)."""

    def normalize_commissions(
        self,
        commissions: Union[Dict[str, float], float, int],
        new_weights: pd.Series,
        existing_positions: Dict[str, Any],
    ) -> Dict[str, float]:
        """Convert single commission value to dict for all tickers."""
        if isinstance(commissions, (int, float)):
            commission_value = float(commissions)
            all_tickers = set(new_weights.keys()) | set(existing_positions.keys())
            return {ticker: commission_value for ticker in all_tickers}
        return {}


class CommissionParameterHandlerFactory:
    """Factory for creating appropriate commission parameter handlers."""

    @staticmethod
    def create_handler(
        commissions: Union[Dict[str, float], float, int],
    ) -> ICommissionParameterHandler:
        """Create appropriate commission parameter handler based on parameter type."""
        if isinstance(commissions, (int, float)):
            return NumericCommissionParameterHandler()
        else:
            return DictCommissionParameterHandler()
