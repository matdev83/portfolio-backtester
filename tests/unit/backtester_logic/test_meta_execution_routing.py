"""Tests for explicit meta vs canonical portfolio execution routing."""

from __future__ import annotations

from unittest.mock import MagicMock


from portfolio_backtester.backtester_logic.meta_execution import (
    MetaExecutionMode,
    portfolio_execution_mode_for_strategy,
)
from portfolio_backtester.interfaces.strategy_resolver import StrategyResolverFactory


class _PlainMomentumStrategy:
    """Name intentionally avoids meta substring (non-meta)."""


class _DemoMetaStrategy:
    """Suffix matches resolver meta detection (MetaStrategy in name)."""


def test_portfolio_execution_mode_none_strategy_is_canonical() -> None:
    assert (
        portfolio_execution_mode_for_strategy(None)
        is MetaExecutionMode.CANONICAL_SHARE_CASH_SIMULATION
    )


def test_portfolio_execution_mode_plain_class_is_canonical() -> None:
    s = _PlainMomentumStrategy()
    assert (
        portfolio_execution_mode_for_strategy(s)
        is MetaExecutionMode.CANONICAL_SHARE_CASH_SIMULATION
    )


def test_portfolio_execution_mode_meta_suffix_is_trade_aggregation() -> None:
    s = _DemoMetaStrategy()
    assert portfolio_execution_mode_for_strategy(s) is MetaExecutionMode.TRADE_AGGREGATION


def test_portfolio_execution_mode_respects_explicit_resolver() -> None:
    mock = MagicMock()
    mock.is_meta_strategy.return_value = True
    s = _PlainMomentumStrategy()
    assert (
        portfolio_execution_mode_for_strategy(s, strategy_resolver=mock)
        is MetaExecutionMode.TRADE_AGGREGATION
    )
    mock.is_meta_strategy.assert_called_once_with(type(s))


def test_resolver_factory_strategy_type_meta_vs_signal() -> None:
    resolver = StrategyResolverFactory.create()
    assert resolver.is_meta_strategy(_DemoMetaStrategy) is True
    assert resolver.is_meta_strategy(_PlainMomentumStrategy) is False


def test_meta_execution_mode_enum_documented_trade_aggregation_member() -> None:
    assert MetaExecutionMode.TRADE_AGGREGATION.value == "trade_aggregation"
