"""
Holding processor interfaces for individual holding item processing.

This module provides polymorphic interfaces to replace isinstance violations
in holding item processing for better adherence to the Open/Closed Principle.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple
import pandas as pd
import re
import logging
from .attribute_accessor_interface import (
    IObjectFieldAccessor,
    create_object_field_accessor,
)

logger = logging.getLogger(__name__)


class IHoldingProcessor(ABC):
    """Interface for processing individual holding items."""

    @abstractmethod
    def process_holding(
        self, item: Any, date: pd.Timestamp
    ) -> Optional[Tuple[pd.Timestamp, str, Optional[float], Optional[float], Optional[float]]]:
        """
        Process a single holding item.

        Parameters
        ----------
        item : Any
            Holding item to process
        date : pd.Timestamp
            Date associated with the holding

        Returns
        -------
        Optional[Tuple[pd.Timestamp, str, Optional[float], Optional[float], Optional[float]]]
            Tuple of (date, ticker, weight_pct, shares, market_value) or None if processing fails
        """
        pass

    @abstractmethod
    def can_process(self, item: Any) -> bool:
        """
        Check if this processor can handle the item format.

        Parameters
        ----------
        item : Any
            Item to check

        Returns
        -------
        bool
            True if this processor can handle the item, False otherwise
        """
        pass


class ModernEdgarHoldingProcessor(IHoldingProcessor):
    """Processor for modern EdgarTools format (InvestmentOrSecurity objects)."""

    def __init__(
        self,
        cusip_to_ticker_func,
        field_accessor: Optional[IObjectFieldAccessor] = None,
    ):
        """
        Initialize with CUSIP to ticker conversion function.

        Parameters
        ----------
        cusip_to_ticker_func : callable
            Function that converts CUSIP to ticker symbol
        field_accessor : IObjectFieldAccessor, optional
            Injected accessor for object field access (DIP)
        """
        self._cusip_to_ticker = cusip_to_ticker_func
        # Dependency injection for field access (DIP)
        self._field_accessor = field_accessor or create_object_field_accessor()

    def process_holding(
        self, item: Any, date: pd.Timestamp
    ) -> Optional[Tuple[pd.Timestamp, str, Optional[float], Optional[float], Optional[float]]]:
        """Process modern EdgarTools holding item."""
        if not self.can_process(item):
            return None

        ticker = None
        pct_nav = None
        shares = None
        value = None

        raw_ticker = self._field_accessor.get_field_value(item, "ticker", None)
        cusip = self._field_accessor.get_field_value(item, "cusip", None)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"ModernEdgarHoldingProcessor: raw_ticker={raw_ticker}, cusip={cusip}")

        resolved_ticker = None
        if cusip:
            resolved_ticker = self._cusip_to_ticker(cusip)
            if resolved_ticker == cusip and raw_ticker:  # CUSIP not mapped, but raw_ticker exists
                resolved_ticker = raw_ticker  # Fallback to raw_ticker
        elif raw_ticker:
            resolved_ticker = raw_ticker

        # Fallback to name-based heuristic if no ticker/CUSIP resolution yet
        if not resolved_ticker and hasattr(item, "name") and item.name:
            resolved_ticker = item.name.replace(" ", "").upper()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"ModernEdgarHoldingProcessor: No ticker/CUSIP, falling back to name-based heuristic: {resolved_ticker}"
                )

        # Ensure resolved_ticker is cleaned
        if resolved_ticker:
            ticker = re.sub(r"[^a-zA-Z0-9]", "", str(resolved_ticker)).upper()
        else:
            ticker = None  # No valid ticker found

        pct_nav = (
            float(self._field_accessor.get_field_value(item, "pct_value", 0))
            if self._field_accessor.get_field_value(item, "pct_value", None) is not None
            else None
        )
        shares = (
            float(self._field_accessor.get_field_value(item, "balance", 0))
            if self._field_accessor.get_field_value(item, "balance", None) is not None
            else None
        )
        value = (
            float(self._field_accessor.get_field_value(item, "value_usd", 0))
            if self._field_accessor.get_field_value(item, "value_usd", None) is not None
            else None
        )
        asset_category = self._field_accessor.get_field_value(item, "asset_category", "").upper()

        if ticker and asset_category == "EC":  # Equity Common stock
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"ModernEdgarHoldingProcessor: Returning: date={date}, ticker={ticker}, weight_pct={pct_nav}, shares={shares}, market_value={value}"
                )
            return date, ticker, pct_nav, shares, value

        return None

    def can_process(self, item: Any) -> bool:
        """Check if item has model_dump method (modern EdgarTools format)."""
        return hasattr(item, "model_dump")


class DictHoldingProcessor(IHoldingProcessor):
    """Processor for dictionary-formatted holding items (old EdgarTools format)."""

    def __init__(self, cusip_to_ticker_func):
        """
        Initialize with CUSIP to ticker conversion function.

        Parameters
        ----------
        cusip_to_ticker_func : callable
            Function that converts CUSIP to ticker symbol
        """
        self._cusip_to_ticker = cusip_to_ticker_func

    def process_holding(
        self, item: Any, date: pd.Timestamp
    ) -> Optional[Tuple[pd.Timestamp, str, Optional[float], Optional[float], Optional[float]]]:
        """Process dictionary holding item."""
        if not self.can_process(item):
            return None

        security_type = item.get("security_type", "").lower()
        if security_type == "common stock":
            identifier = str(item.get("identifier", "")).upper()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"DictHoldingProcessor: identifier={identifier}")

            resolved_ticker = self._cusip_to_ticker(identifier)
            # Clean up any remaining non-alphanumeric characters from the resolved ticker
            if resolved_ticker:
                ticker_to_use = re.sub(r"[^a-zA-Z0-9]", "", str(resolved_ticker)).upper()
            else:
                ticker_to_use = None

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"DictHoldingProcessor: ticker_to_use after _cusip_to_ticker={ticker_to_use}"
                )

            if ticker_to_use:
                pct_nav = (
                    item.get("pct_nav")
                    if item.get("pct_nav") is not None
                    else item.get("pct_value")
                )
                shares = item.get("shares")
                value = item.get("value")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"DictHoldingProcessor: Returning: date={date}, ticker={ticker_to_use.upper()}, weight_pct={pct_nav}, shares={shares}, market_value={value}"
                    )
                return date, ticker_to_use.upper(), pct_nav, shares, value

        return None

    def can_process(self, item: Any) -> bool:
        """Check if item is a dictionary."""
        return isinstance(item, dict)


class CompositeHoldingProcessor(IHoldingProcessor):
    """Composite processor that tries multiple processing strategies."""

    def __init__(self, cusip_to_ticker_func):
        """
        Initialize with CUSIP to ticker conversion function.

        Parameters
        ----------
        cusip_to_ticker_func : callable
            Function that converts CUSIP to ticker symbol
        """
        self._processors = [
            ModernEdgarHoldingProcessor(cusip_to_ticker_func),
            DictHoldingProcessor(cusip_to_ticker_func),
        ]

    def process_holding(
        self, item: Any, date: pd.Timestamp
    ) -> Optional[Tuple[pd.Timestamp, str, Optional[float], Optional[float], Optional[float]]]:
        """Try each processor until one succeeds."""
        for processor in self._processors:
            if processor.can_process(item):
                return processor.process_holding(item, date)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"CompositeHoldingProcessor: No valid item processed for date {date}")
        return None

    def can_process(self, item: Any) -> bool:
        """Check if any processor can handle the item."""
        return any(processor.can_process(item) for processor in self._processors)


class HoldingProcessorFactory:
    """Factory for creating holding processor instances."""

    @staticmethod
    def create_holding_processor(cusip_to_ticker_func) -> IHoldingProcessor:
        """Create a composite holding processor."""
        return CompositeHoldingProcessor(cusip_to_ticker_func)
