"""Tests for data source factory wiring of reproducibility-sensitive MDMP options."""

from typing import cast
from unittest.mock import patch

import pytest

from portfolio_backtester.data_sources.mdmp_data_source import MarketDataMultiProviderDataSource
from portfolio_backtester.interfaces.data_source_interface import ConcreteDataSourceFactory


def test_factory_passes_pinned_provider_cache_only_no_fallback() -> None:
    with patch("portfolio_backtester.data_sources.mdmp_data_source.MarketDataClient"):
        fac = ConcreteDataSourceFactory()
        ds = cast(
            MarketDataMultiProviderDataSource,
            fac.create_data_source(
                {
                    "data_source": "mdmp",
                    "data_source_config": {
                        "preferred_provider": "yfinance",
                        "allow_fallbacks": False,
                        "cache_only": True,
                        "cache_max_age_seconds": 7200,
                    },
                }
            ),
        )

    assert ds._preferred_provider == "yfinance"
    assert ds._allow_fallbacks is False
    assert ds._cache_only is True
    assert ds._cache_max_age_seconds == 7200
    assert ds.data_dir is None


def test_factory_defaults_allow_fallbacks_true_when_omitted() -> None:
    with patch("portfolio_backtester.data_sources.mdmp_data_source.MarketDataClient"):
        fac = ConcreteDataSourceFactory()
        ds = cast(
            MarketDataMultiProviderDataSource,
            fac.create_data_source({"data_source": "mdmp", "data_source_config": {}}),
        )

    assert ds._allow_fallbacks is True
    assert ds.data_dir is None


def test_factory_logs_effective_mdmp_config(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    caplog.set_level(logging.INFO, logger="portfolio_backtester.interfaces.data_source_interface")
    with patch("portfolio_backtester.data_sources.mdmp_data_source.MarketDataClient"):
        ConcreteDataSourceFactory().create_data_source(
            {
                "data_source": "mdmp",
                "data_source_config": {
                    "preferred_provider": "stooq",
                    "allow_fallbacks": False,
                    "cache_only": True,
                },
            }
        )
    msgs = "\n".join(r.message for r in caplog.records)
    assert "MDMP effective data_source_config" in msgs
    assert "stooq" in msgs
    assert "allow_fallbacks=False" in msgs
