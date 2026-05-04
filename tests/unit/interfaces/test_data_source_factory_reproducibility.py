"""Tests for data source factory wiring of reproducibility-sensitive MDMP options."""

from typing import cast
from unittest.mock import patch

from pathlib import Path

import pytest

from portfolio_backtester.data_sources.mdmp_data_source import MarketDataMultiProviderDataSource
from portfolio_backtester.interfaces.data_source_interface import (
    ConcreteDataSourceFactory,
    _resolve_mdmp_data_dir,
)


def test_resolve_mdmp_data_dir_absolute(tmp_path) -> None:
    d = tmp_path / "canonical_root"
    d.mkdir()
    out = _resolve_mdmp_data_dir(str(d))
    assert Path(out) == d.resolve()


def test_resolve_mdmp_data_dir_relative_points_under_repo() -> None:
    out = _resolve_mdmp_data_dir("../market-data-multi-provider/data")
    assert Path(out).is_absolute()
    assert out.replace("\\", "/").endswith("market-data-multi-provider/data")


def test_factory_passes_resolved_relative_mdmp_data_dir() -> None:
    with patch(
        "portfolio_backtester.data_sources.mdmp_data_source.MarketDataClient"
    ) as mock_client:
        ConcreteDataSourceFactory().create_data_source(
            {
                "data_source": "mdmp",
                "data_source_config": {
                    "mdmp_data_dir": "../market-data-multi-provider/data",
                },
            }
        )
    passed = mock_client.call_args.kwargs.get("data_dir")
    assert passed is not None
    assert Path(passed).is_absolute()
    assert "market-data-multi-provider" in str(passed)


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


def test_factory_passes_mdmp_data_dir_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    root = str(tmp_path / "mdmp_cache_pb")
    monkeypatch.setenv("MDMP_DATA_DIR", root)
    with patch(
        "portfolio_backtester.data_sources.mdmp_data_source.MarketDataClient"
    ) as mock_client:
        ConcreteDataSourceFactory().create_data_source(
            {"data_source": "mdmp", "data_source_config": {"preferred_provider": "stooq"}},
        )
    mock_client.assert_called_once()
    assert mock_client.call_args.kwargs.get("data_dir") == root


def test_factory_cache_max_age_none_when_explicit() -> None:
    with patch("portfolio_backtester.data_sources.mdmp_data_source.MarketDataClient"):
        ds = cast(
            MarketDataMultiProviderDataSource,
            ConcreteDataSourceFactory().create_data_source(
                {
                    "data_source": "mdmp",
                    "data_source_config": {"cache_max_age_seconds": None},
                }
            ),
        )
    assert ds._cache_max_age_seconds is None
