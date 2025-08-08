"""
Filing date extraction interfaces for SEC filing data processing.

This module provides polymorphic interfaces to replace isinstance violations
in SEC filing date extraction for better adherence to the Open/Closed Principle.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import datetime as dt
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class IFilingDateExtractor(ABC):
    """Interface for extracting dates from SEC filing objects."""

    @abstractmethod
    def extract_date(self, filing: Any) -> Optional[pd.Timestamp]:
        """
        Extract and validate the period_of_report date from a filing.

        Parameters
        ----------
        filing : Any
            Filing object with period_of_report attribute

        Returns
        -------
        Optional[pd.Timestamp]
            Extracted timestamp or None if extraction fails
        """
        pass

    @abstractmethod
    def can_extract(self, filing: Any) -> bool:
        """
        Check if this extractor can handle the filing format.

        Parameters
        ----------
        filing : Any
            Filing object to check

        Returns
        -------
        bool
            True if this extractor can handle the filing, False otherwise
        """
        pass


class StringFilingDateExtractor(IFilingDateExtractor):
    """Extractor for string date formats in filings."""

    def extract_date(self, filing: Any) -> Optional[pd.Timestamp]:
        """Extract date from string period_of_report."""
        if not self.can_extract(filing):
            return None

        period_of_report = filing.period_of_report
        try:
            return pd.Timestamp(period_of_report)
        except ValueError:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    f"Skipping filing {filing.accession_no}: Invalid date string '{period_of_report}'"
                )
            return None

    def can_extract(self, filing: Any) -> bool:
        """Check if filing has string period_of_report."""
        return hasattr(filing, "period_of_report") and isinstance(filing.period_of_report, str)


class DateObjectFilingDateExtractor(IFilingDateExtractor):
    """Extractor for datetime.date formats in filings."""

    def extract_date(self, filing: Any) -> Optional[pd.Timestamp]:
        """Extract date from datetime.date period_of_report."""
        if not self.can_extract(filing):
            return None

        return pd.Timestamp(filing.period_of_report)

    def can_extract(self, filing: Any) -> bool:
        """Check if filing has datetime.date period_of_report."""
        return hasattr(filing, "period_of_report") and isinstance(filing.period_of_report, dt.date)


class CompositeFilingDateExtractor(IFilingDateExtractor):
    """Composite extractor that tries multiple extraction strategies."""

    def __init__(self):
        self._extractors = [
            StringFilingDateExtractor(),
            DateObjectFilingDateExtractor(),
        ]

    def extract_date(self, filing: Any) -> Optional[pd.Timestamp]:
        """Try each extractor until one succeeds."""
        for extractor in self._extractors:
            if extractor.can_extract(filing):
                return extractor.extract_date(filing)

        # Log warning if no extractor could handle the filing
        if hasattr(filing, "accession_no") and hasattr(filing, "period_of_report"):
            period_type = type(filing.period_of_report)
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    f"Skipping filing {filing.accession_no}: Invalid period_of_report type {period_type}."
                )

        return None

    def can_extract(self, filing: Any) -> bool:
        """Check if any extractor can handle the filing."""
        return any(extractor.can_extract(filing) for extractor in self._extractors)


class FilingDateExtractorFactory:
    """Factory for creating filing date extractor instances."""

    @staticmethod
    def create_filing_date_extractor() -> IFilingDateExtractor:
        """Create a composite filing date extractor."""
        return CompositeFilingDateExtractor()
