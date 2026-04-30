"""Validation tests for ``timing_config.trade_execution_timing`` (TDD)."""

from __future__ import annotations

from portfolio_backtester.timing.config_validator import TimingConfigValidator


def test_validate_trade_execution_timing_accepts_bar_close() -> None:
    assert TimingConfigValidator.validate_trade_execution_timing("bar_close") == []


def test_validate_trade_execution_timing_accepts_next_bar_open() -> None:
    assert TimingConfigValidator.validate_trade_execution_timing("next_bar_open") == []


def test_validate_trade_execution_timing_none_means_omit() -> None:
    assert TimingConfigValidator.validate_trade_execution_timing(None) == []


def test_validate_trade_execution_timing_rejects_invalid_string() -> None:
    errors = TimingConfigValidator.validate_trade_execution_timing("at_the_open")
    assert errors
    assert "trade_execution_timing" in errors[0]


def test_validate_trade_execution_timing_rejects_non_string() -> None:
    errors = TimingConfigValidator.validate_trade_execution_timing(1)
    assert errors
    assert "string" in errors[0].lower()


def test_validate_config_surfaces_trade_execution_timing_errors() -> None:
    errors = TimingConfigValidator.validate_config(
        {
            "mode": "time_based",
            "rebalance_frequency": "M",
            "trade_execution_timing": "invalid",
        }
    )
    assert errors
    assert any("trade_execution_timing" in msg for msg in errors)


def test_validate_config_allows_valid_trade_execution_timing() -> None:
    errors = TimingConfigValidator.validate_config(
        {
            "mode": "time_based",
            "rebalance_frequency": "M",
            "trade_execution_timing": "next_bar_open",
        }
    )
    assert errors == []
