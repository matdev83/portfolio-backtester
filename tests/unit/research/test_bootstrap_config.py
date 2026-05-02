"""Tests for research protocol bootstrap configuration parsing."""

from __future__ import annotations

import pytest

from portfolio_backtester.research.protocol_config import (
    BootstrapConfig,
    ResearchProtocolConfigError,
    parse_double_oos_wfo_protocol,
)

from tests.unit.research.test_protocol_config import _minimal_primary_inner


def test_bootstrap_defaults_disabled() -> None:
    cfg = parse_double_oos_wfo_protocol({"research_protocol": _minimal_primary_inner()})
    bs = cfg.bootstrap
    assert isinstance(bs, BootstrapConfig)
    assert bs.enabled is False
    assert bs.n_samples == 200
    assert bs.random_seed == 42
    assert bs.random_wfo_architecture.enabled is False
    assert bs.block_shuffled_returns.enabled is False
    assert bs.block_shuffled_returns.block_size_days == 20
    assert bs.block_shuffled_positions.enabled is False
    assert bs.block_shuffled_positions.block_size_days == 20
    assert bs.random_strategy_parameters.enabled is False
    assert bs.random_strategy_parameters.sample_size == 100
    assert bs.persist_distribution_samples is False


def test_bootstrap_parse_full_block() -> None:
    inner = _minimal_primary_inner()
    inner["bootstrap"] = {
        "enabled": True,
        "n_samples": 200,
        "random_seed": 42,
        "random_wfo_architecture": {"enabled": True},
        "block_shuffled_returns": {"enabled": True, "block_size_days": 20},
        "block_shuffled_positions": {"enabled": True, "block_size_days": 15},
        "random_strategy_parameters": {"enabled": True, "sample_size": 50},
    }
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    bs = cfg.bootstrap
    assert bs.enabled is True
    assert bs.n_samples == 200
    assert bs.random_seed == 42
    assert bs.random_wfo_architecture.enabled is True
    assert bs.block_shuffled_returns.enabled is True
    assert bs.block_shuffled_returns.block_size_days == 20
    assert bs.block_shuffled_positions.enabled is True
    assert bs.block_shuffled_positions.block_size_days == 15
    assert bs.random_strategy_parameters.enabled is True
    assert bs.random_strategy_parameters.sample_size == 50


def test_bootstrap_enabled_non_mapping_rejected() -> None:
    inner = _minimal_primary_inner()
    inner["bootstrap"] = "yes"
    with pytest.raises(ResearchProtocolConfigError, match="bootstrap"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_bootstrap_enabled_n_samples_non_positive_rejected() -> None:
    inner = _minimal_primary_inner()
    inner["bootstrap"] = {"enabled": True, "n_samples": 0}
    with pytest.raises(ResearchProtocolConfigError, match="n_samples"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_bootstrap_block_size_non_positive_when_enabled_rejected() -> None:
    inner = _minimal_primary_inner()
    inner["bootstrap"] = {
        "enabled": True,
        "n_samples": 10,
        "block_shuffled_returns": {"enabled": True, "block_size_days": 0},
    }
    with pytest.raises(ResearchProtocolConfigError, match="block_size"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_bootstrap_block_shuffled_positions_block_size_non_positive_when_enabled_rejected() -> None:
    inner = _minimal_primary_inner()
    inner["bootstrap"] = {
        "enabled": True,
        "n_samples": 10,
        "block_shuffled_positions": {"enabled": True, "block_size_days": 0},
    }
    with pytest.raises(ResearchProtocolConfigError, match="block_shuffled_positions"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})


def test_bootstrap_random_strategy_parameters_default_disabled() -> None:
    inner = _minimal_primary_inner()
    inner["bootstrap"] = {"enabled": True, "n_samples": 10}
    cfg = parse_double_oos_wfo_protocol({"research_protocol": inner})
    rsp = cfg.bootstrap.random_strategy_parameters
    assert rsp.enabled is False
    assert rsp.sample_size == 100


def test_bootstrap_random_strategy_parameters_sample_size_non_positive_rejected() -> None:
    inner = _minimal_primary_inner()
    inner["bootstrap"] = {
        "enabled": True,
        "n_samples": 10,
        "random_strategy_parameters": {"enabled": True, "sample_size": 0},
    }
    with pytest.raises(ResearchProtocolConfigError, match="sample_size"):
        parse_double_oos_wfo_protocol({"research_protocol": inner})
