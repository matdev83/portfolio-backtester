"""
TickerCollector interface and implementations for polymorphic ticker collection.

Replaces isinstance checks in universe handling with proper polymorphic behavior.
"""

from abc import ABC, abstractmethod
from typing import List, Any
import logging
import warnings

logger = logging.getLogger(__name__)


# Suppress deprecation warning for internal usage
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"DEPRECATED: resolve_universe_config is deprecated",
)


class ITickerCollector(ABC):
    """Abstract interface for collecting tickers from universe configurations."""

    @abstractmethod
    def collect_tickers(self, universe_config: Any) -> List[str]:
        """
        Collect ticker symbols from universe configuration.

        Args:
            universe_config: Universe configuration (list, dict, or string)

        Returns:
            List of ticker symbols

        Raises:
            ValueError: If universe configuration is invalid
        """
        pass


class ListTickerCollector(ITickerCollector):
    """Collects tickers from list-based universe configurations."""

    def collect_tickers(self, universe_config: Any) -> List[str]:
        """
        Collect tickers from a list configuration.

        Args:
            universe_config: List of ticker symbols

        Returns:
            List of ticker symbols

        Raises:
            ValueError: If universe_config is not a list
        """
        if not isinstance(universe_config, list):
            raise ValueError(f"Expected list, got {type(universe_config).__name__}")

        return list(universe_config)


class ConfigTickerCollector(ITickerCollector):
    """Collects tickers from config-based universe configurations."""

    def collect_tickers(self, universe_config: Any) -> List[str]:
        """
        Collect tickers from a configuration that needs resolution.

        Args:
            universe_config: Configuration that needs to be resolved

        Returns:
            List of ticker symbols

        Raises:
            ValueError: If universe configuration cannot be resolved
        """
        try:
            from ..universe_resolver import resolve_universe_config

            return list(resolve_universe_config(universe_config))
        except Exception as e:
            logger.error(f"Failed to resolve universe config: {e}")
            raise ValueError(f"Cannot resolve universe config: {e}")


class TickerCollectorFactory:
    """Factory for creating appropriate ticker collectors based on configuration type."""

    @staticmethod
    def create_collector(universe_config: Any) -> ITickerCollector:
        """
        Create appropriate ticker collector based on configuration type.

        Args:
            universe_config: Universe configuration

        Returns:
            Appropriate ITickerCollector implementation

        Raises:
            ValueError: If configuration type is not supported
        """
        if isinstance(universe_config, list):
            return ListTickerCollector()
        else:
            # String, dict, or any other configuration that needs resolution
            return ConfigTickerCollector()
