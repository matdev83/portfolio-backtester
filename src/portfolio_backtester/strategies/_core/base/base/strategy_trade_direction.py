"""Trade direction configuration and signal validation for strategies."""

from __future__ import annotations

from typing import Any

import pandas as pd


class TradeDirectionConfigurationError(ValueError):
    """Raised when trade direction configuration is invalid."""

    def __init__(self, strategy_class, trade_longs, trade_shorts, details):
        self.strategy_class = strategy_class
        self.trade_longs = trade_longs
        self.trade_shorts = trade_shorts
        self.details = details
        message = f"Invalid trade direction configuration in {strategy_class}: {details}"
        super().__init__(message)


class TradeDirectionViolationError(ValueError):
    """Raised when a strategy violates its trade direction constraints."""

    def __init__(
        self,
        strategy_class,
        trade_longs,
        trade_shorts,
        violation_details,
        violated_signals=None,
    ):
        self.strategy_class = strategy_class
        self.trade_longs = trade_longs
        self.trade_shorts = trade_shorts
        self.violation_details = violation_details
        self.violated_signals = violated_signals
        message = f"Trade direction violation in {strategy_class}: {violation_details}"
        super().__init__(message)


def validate_trade_direction_configuration(
    strategy_class_name: str, trade_longs: Any, trade_shorts: Any
) -> None:
    """Validate trade_longs / trade_shorts; raises TradeDirectionConfigurationError if invalid."""
    if not trade_longs and not trade_shorts:
        raise TradeDirectionConfigurationError(
            strategy_class=strategy_class_name,
            trade_longs=trade_longs,
            trade_shorts=trade_shorts,
            details="Both trade_longs and trade_shorts are False - strategy cannot trade!",
        )

    if not isinstance(trade_longs, bool):
        raise TradeDirectionConfigurationError(
            strategy_class=strategy_class_name,
            trade_longs=trade_longs,
            trade_shorts=trade_shorts,
            details=f"trade_longs must be boolean, got {type(trade_longs).__name__}",
        )

    if not isinstance(trade_shorts, bool):
        raise TradeDirectionConfigurationError(
            strategy_class=strategy_class_name,
            trade_longs=trade_longs,
            trade_shorts=trade_shorts,
            details=f"trade_shorts must be boolean, got {type(trade_shorts).__name__}",
        )


def validate_signals_trade_direction(
    strategy_class_name: str,
    trade_longs: bool,
    trade_shorts: bool,
    signals: pd.DataFrame,
) -> None:
    """Ensure signal weights respect trade_longs / trade_shorts."""
    if signals.empty:
        return

    if not trade_longs:
        positive_mask = signals > 0
        if positive_mask.any().any():
            positive_signals = signals[positive_mask]
            violation_count = positive_mask.sum().sum()
            raise TradeDirectionViolationError(
                strategy_class=strategy_class_name,
                trade_longs=trade_longs,
                trade_shorts=trade_shorts,
                violation_details=(
                    f"Generated {violation_count} positive (long) signals when trade_longs=False"
                ),
                violated_signals=positive_signals,
            )

    if not trade_shorts:
        negative_mask = signals < 0
        if negative_mask.any().any():
            negative_signals = signals[negative_mask]
            violation_count = negative_mask.sum().sum()
            raise TradeDirectionViolationError(
                strategy_class=strategy_class_name,
                trade_longs=trade_longs,
                trade_shorts=trade_shorts,
                violation_details=(
                    f"Generated {violation_count} negative (short) signals when trade_shorts=False"
                ),
                violated_signals=negative_signals,
            )
