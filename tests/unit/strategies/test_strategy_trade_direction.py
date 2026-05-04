"""Unit tests for strategy_trade_direction helpers."""

import pandas as pd
import pytest

from portfolio_backtester.strategies._core.base.base.strategy_trade_direction import (
    TradeDirectionConfigurationError,
    TradeDirectionViolationError,
    validate_signals_trade_direction,
    validate_trade_direction_configuration,
)


def test_validate_trade_direction_both_false() -> None:
    with pytest.raises(TradeDirectionConfigurationError, match="Both trade_longs"):
        validate_trade_direction_configuration("S", False, False)


def test_validate_signals_long_forbidden() -> None:
    sig = pd.DataFrame([[0.5]], columns=["A"])
    with pytest.raises(TradeDirectionViolationError, match="positive"):
        validate_signals_trade_direction("S", False, True, sig)
