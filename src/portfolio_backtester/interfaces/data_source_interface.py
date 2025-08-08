"""
Interface for data source dependencies implementing Dependency Inversion Principle.

This module provides abstractions for data source creation and management,
enabling dependency inversion for backtester components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, cast

import pandas as pd
from ..data_sources.base_data_source import BaseDataSource


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
        # Import here to avoid circular dependencies
        from ..data_sources.stooq_data_source import StooqDataSource
        from ..data_sources.yfinance_data_source import YFinanceDataSource
        from ..data_sources.hybrid_data_source import HybridDataSource
        from ..data_sources.memory_data_source import MemoryDataSource

        data_source_map = {
            "stooq": StooqDataSource,
            "yfinance": YFinanceDataSource,
            "hybrid": HybridDataSource,
            "memory": MemoryDataSource,
            "test": MemoryDataSource,  # Test data source for API stability tests
        }

        ds_name = global_config.get("data_source", "hybrid").lower()
        data_source_class = data_source_map.get(ds_name)

        if data_source_class:
            if ds_name == "hybrid":
                prefer_stooq = global_config.get("prefer_stooq", True)
                return cast(
                    IDataSource,
                    HybridDataSource(
                        cache_expiry_hours=24,
                        prefer_stooq=prefer_stooq,
                        negative_cache_timeout_hours=4,
                    ),
                )
            elif ds_name == "memory" or ds_name == "test":
                return cast(
                    IDataSource, MemoryDataSource(global_config.get("data_source_config", {}))
                )
            else:
                return cast(IDataSource, data_source_class())
        else:
            raise ValueError(f"Unsupported data source: {ds_name}")


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
