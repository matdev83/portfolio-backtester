"""
Interface for data source dependencies implementing Dependency Inversion Principle.

This module provides abstractions for data source creation and management,
enabling dependency inversion for backtester components.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, cast

import pandas as pd
from ..data_sources.base_data_source import BaseDataSource

logger = logging.getLogger(__name__)


class IDataSource(BaseDataSource):
    """
    Abstract interface for data sources extending BaseDataSource.

    This interface defines the contract that all data source implementations
    must follow, enabling dependency inversion for backtester components.
    This inherits from BaseDataSource to ensure compatibility with existing implementations.
    """

    @abstractmethod
    def get_data(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch price data for the given tickers and date range.

        Args:
            tickers: List of ticker symbols to fetch data for
            start_date: Start date in string format
            end_date: End date in string format

        Returns:
            DataFrame containing price data for the requested tickers
        """
        pass


class IDataSourceFactory(ABC):
    """
    Abstract factory interface for creating data source instances.
    """

    @abstractmethod
    def create_data_source(self, global_config: Dict[str, Any]) -> IDataSource:
        """
        Create a data source instance based on configuration.

        Args:
            global_config: Global configuration dictionary

        Returns:
            Data source instance implementing IDataSource
        """
        pass


class ConcreteDataSourceFactory(IDataSourceFactory):
    """
    Concrete implementation of data source factory.

    This factory creates appropriate data source instances based on
    configuration without exposing concrete implementation details.
    """

    def create_data_source(self, global_config: Dict[str, Any]) -> IDataSource:
        """
        Create a data source instance based on configuration.

        Args:
            global_config: Global configuration dictionary

        Returns:
            Data source instance implementing IDataSource
        """
        from ..data_sources.memory_data_source import MemoryDataSource

        ds_name = global_config.get("data_source", "mdmp").lower()

        # Handle memory/test data source
        if ds_name in ("memory", "test"):
            return cast(
                IDataSource,
                MemoryDataSource(global_config.get("data_source_config", {})),
            )

        # Handle MDMP data source (default)
        if ds_name in ("mdmp", "market-data-multi-provider"):
            try:
                from ..data_sources.mdmp_data_source import MarketDataMultiProviderDataSource

                min_coverage_ratio = global_config.get("min_coverage_ratio")
                data_source_config = global_config.get("data_source_config", {}) or {}
                allow_fallbacks = bool(data_source_config.get("allow_fallbacks", True))
                cache_only = bool(data_source_config.get("cache_only", False))
                preferred_provider = data_source_config.get("preferred_provider")
                cache_max_age_seconds = data_source_config.get("cache_max_age_seconds", 14400)
                max_workers = data_source_config.get("max_workers")
                logger.info(
                    "MDMP effective data_source_config (reproducibility): "
                    "preferred_provider=%r, allow_fallbacks=%s, cache_only=%s, "
                    "cache_max_age_seconds=%s, max_workers=%s",
                    preferred_provider,
                    allow_fallbacks,
                    cache_only,
                    cache_max_age_seconds,
                    max_workers,
                )
                return cast(
                    IDataSource,
                    MarketDataMultiProviderDataSource(
                        data_dir=None,
                        min_coverage_ratio=min_coverage_ratio,
                        preferred_provider=preferred_provider,
                        allow_fallbacks=allow_fallbacks,
                        max_workers=max_workers,
                        cache_only=cache_only,
                        cache_max_age_seconds=cache_max_age_seconds,
                    ),
                )
            except ImportError as e:
                raise ImportError(
                    f"Cannot use MDMP data source: {e}. "
                    "Install market-data-multi-provider with: "
                    "pip install -e ../market-data-multi-provider"
                ) from e

        # Unsupported data source
        raise ValueError(
            f"Unsupported data source: {ds_name}. "
            f"Valid options: 'mdmp', 'market-data-multi-provider', 'memory', 'test'"
        )


# Factory instance for dependency injection
def create_data_source_factory() -> IDataSourceFactory:
    """
    Create a data source factory instance.

    Returns:
        Data source factory implementing IDataSourceFactory
    """
    return ConcreteDataSourceFactory()


def create_data_source(global_config: Dict[str, Any]) -> IDataSource:
    """
    Create a data source instance using the factory.

    Args:
        global_config: Global configuration dictionary

    Returns:
        Data source instance implementing IDataSource
    """
    factory = create_data_source_factory()
    return factory.create_data_source(global_config)
