"""
Interface for data source dependencies implementing Dependency Inversion Principle.

This module provides abstractions for data source creation and management,
enabling dependency inversion for backtester components.
"""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Union, cast

import pandas as pd
from ..data_sources.base_data_source import BaseDataSource

logger = logging.getLogger(__name__)


def _resolve_mdmp_data_dir(mdmp_data_dir: Union[str, os.PathLike[str]]) -> str:
    """Resolve MDMP disk root for ``MarketDataClient(data_dir=...)``.

    Absolute paths are normalized with :func:`Path.resolve`. Relative paths are
    resolved from the **portfolio-backtester repository root** (the directory
    that contains ``src/``), so a sibling checkout
    ``../market-data-multi-provider/data`` works without duplicating parquet under
    this repo.

    Args:
        mdmp_data_dir: Path from ``parameters.yaml`` or ``MDMP_DATA_DIR``.

    Returns:
        Absolute string path for MDMP.
    """
    p = Path(mdmp_data_dir).expanduser()
    if p.is_absolute():
        return str(p.resolve())
    repo_root = Path(__file__).resolve().parents[3]
    return str((repo_root / p).resolve())


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


# Alias for type hints and imports expecting the historical ``PriceDataSource`` name.
PriceDataSource = IDataSource


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
                if "cache_max_age_seconds" in data_source_config:
                    cms_val = data_source_config["cache_max_age_seconds"]
                    cache_max_age_seconds = None if cms_val is None else int(cms_val)
                else:
                    cache_max_age_seconds = 14400
                max_workers = data_source_config.get("max_workers")
                mdmp_data_dir = (
                    data_source_config.get("mdmp_data_dir")
                    or data_source_config.get("data_dir")
                    or os.environ.get("MDMP_DATA_DIR")
                )
                if mdmp_data_dir in ("", None):
                    mdmp_data_dir = None
                else:
                    mdmp_data_dir = _resolve_mdmp_data_dir(str(mdmp_data_dir))
                logger.info(
                    "MDMP effective data_source_config (reproducibility): "
                    "preferred_provider=%r, allow_fallbacks=%s, cache_only=%s, "
                    "cache_max_age_seconds=%s, max_workers=%s, mdmp_data_dir=%s",
                    preferred_provider,
                    allow_fallbacks,
                    cache_only,
                    cache_max_age_seconds,
                    max_workers,
                    str(mdmp_data_dir) if mdmp_data_dir else "MDMP default",
                )
                return cast(
                    IDataSource,
                    MarketDataMultiProviderDataSource(
                        data_dir=mdmp_data_dir,
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
