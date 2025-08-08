"""
Holdings extraction interfaces for SEC filing data processing.

This module provides polymorphic interfaces to replace isinstance violations
in SEC holdings extraction for better adherence to the Open/Closed Principle.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, cast
import logging
from .attribute_accessor_interface import IObjectFieldAccessor, create_object_field_accessor

logger = logging.getLogger(__name__)


# Type guard for FundReport - avoid importing to prevent circular dependencies
def _is_fund_report(obj: Any) -> bool:
    """Check if object is a FundReport using duck typing."""
    # Import FundReport dynamically to avoid dependency issues
    try:
        from edgar.objects import FundReport  # type: ignore[import-untyped]

        return FundReport is not None and isinstance(obj, FundReport)
    except Exception:
        return False


class IHoldingsExtractor(ABC):
    """Interface for extracting holdings from SEC filing objects."""

    @abstractmethod
    def extract_holdings(self, obj: Any, filing_accession_no: str) -> List[Any]:
        """
        Extract holdings items from a parsed SEC filing object.

        Parameters
        ----------
        obj : Any
            Parsed SEC filing object
        filing_accession_no : str
            Filing accession number for logging

        Returns
        -------
        List[Any]
            List of holding items
        """
        pass

    @abstractmethod
    def can_extract(self, obj: Any) -> bool:
        """
        Check if this extractor can handle the object format.

        Parameters
        ----------
        obj : Any
            Object to check

        Returns
        -------
        bool
            True if this extractor can handle the object, False otherwise
        """
        pass


class FundReportHoldingsExtractor(IHoldingsExtractor):
    """Extractor for FundReport objects (edgartools ≥4)."""

    def __init__(self, field_accessor: Optional[IObjectFieldAccessor] = None):
        """
        Initialize FundReport holdings extractor.

        Args:
            field_accessor: Injected accessor for object field access (DIP)
        """
        # Dependency injection for field access (DIP)
        self._field_accessor = field_accessor or create_object_field_accessor()

    def extract_holdings(self, obj: Any, filing_accession_no: str) -> List[Any]:
        """Extract holdings from FundReport object."""
        if not self.can_extract(obj):
            return []

        return cast(List[Any], self._field_accessor.get_field_value(obj.portfolio, "holdings", []))

    def can_extract(self, obj: Any) -> bool:
        """Check if object is a FundReport."""
        return _is_fund_report(obj)


class DictHoldingsExtractor(IHoldingsExtractor):
    """Extractor for dictionary-formatted SEC filing objects."""

    def extract_holdings(self, obj: Any, filing_accession_no: str) -> List[Any]:
        """Extract holdings from dictionary object."""
        if not self.can_extract(obj):
            return []

        portfolio = obj.get("portfolio", [])
        if isinstance(portfolio, list):
            return portfolio
        return []

    def can_extract(self, obj: Any) -> bool:
        """Check if object is a dictionary."""
        return isinstance(obj, dict)


class PortfolioAttributeHoldingsExtractor(IHoldingsExtractor):
    """Extractor for objects with portfolio attribute."""

    def extract_holdings(self, obj: Any, filing_accession_no: str) -> List[Any]:
        """Extract holdings from object with portfolio attribute."""
        if not self.can_extract(obj):
            return []

        port = obj.portfolio
        if isinstance(port, list):
            return port
        if hasattr(port, "holdings"):
            return port.holdings or []
        return []

    def can_extract(self, obj: Any) -> bool:
        """Check if object has portfolio attribute."""
        return hasattr(obj, "portfolio")


class InvestmentsAttributeHoldingsExtractor(IHoldingsExtractor):
    """Extractor for objects with investments attribute (edgar 4.3+)."""

    def extract_holdings(self, obj: Any, filing_accession_no: str) -> List[Any]:
        """Extract holdings from object with investments attribute."""
        if not self.can_extract(obj):
            return []

        return obj.investments or []

    def can_extract(self, obj: Any) -> bool:
        """Check if object has investments attribute."""
        return hasattr(obj, "investments")


class InvestmentDataAttributeHoldingsExtractor(IHoldingsExtractor):
    """Extractor for objects with investment_data attribute."""

    def extract_holdings(self, obj: Any, filing_accession_no: str) -> List[Any]:
        """Extract holdings from object with investment_data attribute."""
        if not self.can_extract(obj):
            return []

        return obj.investment_data or []

    def can_extract(self, obj: Any) -> bool:
        """Check if object has investment_data attribute."""
        return hasattr(obj, "investment_data")


class CompositeHoldingsExtractor(IHoldingsExtractor):
    """Composite extractor that tries multiple extraction strategies."""

    def __init__(self, field_accessor: Optional[IObjectFieldAccessor] = None):
        """
        Initialize composite holdings extractor.

        Args:
            field_accessor: Injected accessor for object field access (DIP)
        """
        # Order matters - try most specific extractors first
        self._extractors = [
            FundReportHoldingsExtractor(field_accessor),  # Pass DIP dependency
            DictHoldingsExtractor(),
            PortfolioAttributeHoldingsExtractor(),
            InvestmentsAttributeHoldingsExtractor(),
            InvestmentDataAttributeHoldingsExtractor(),
        ]

    def extract_holdings(self, obj: Any, filing_accession_no: str) -> List[Any]:
        """Try each extractor until one succeeds."""
        for extractor in self._extractors:
            if extractor.can_extract(obj):
                return extractor.extract_holdings(obj, filing_accession_no)

        # Log warning if no extractor could handle the object
        if logger.isEnabledFor(logging.WARNING):
            logger.warning(
                f"Skipping filing {filing_accession_no}: Unexpected obj type {type(obj)} or no holdings data found."
            )

        return []

    def can_extract(self, obj: Any) -> bool:
        """Check if any extractor can handle the object."""
        return any(extractor.can_extract(obj) for extractor in self._extractors)


class HoldingsExtractorFactory:
    """Factory for creating holdings extractor instances."""

    @staticmethod
    def create_holdings_extractor(
        field_accessor: Optional[IObjectFieldAccessor] = None,
    ) -> IHoldingsExtractor:
        """Create a composite holdings extractor with optional dependency injection."""
        return CompositeHoldingsExtractor(field_accessor)
