"""Trade record data structures for meta strategy trade tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum

import pandas as pd


class TradeSide(Enum):
    """Enumeration for trade sides."""

    BUY = "buy"
    SELL = "sell"


@dataclass
class TradeRecord:
    """
    Individual trade record for meta strategy trade tracking.

    This represents a single trade executed by a sub-strategy within a meta strategy,
    scaled by the capital allocation of that sub-strategy.
    """

    # Core trade information
    date: pd.Timestamp
    asset: str
    quantity: float  # Number of shares/units (positive for buy, negative for sell)
    price: float  # Price per share/unit
    side: TradeSide  # Buy or sell

    # Meta strategy context
    strategy_id: str  # ID of the sub-strategy that generated this trade
    allocated_capital: float  # Capital allocated to the sub-strategy at trade time

    # Cost and metadata
    transaction_cost: float = 0.0
    trade_value: Optional[float] = None  # Will be calculated if not provided

    # Additional metadata
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Calculate derived fields after initialization."""
        if self.trade_value is None:
            self.trade_value = abs(self.quantity * self.price)

        # Ensure quantity sign matches trade side
        if self.side == TradeSide.BUY and self.quantity < 0:
            self.quantity = abs(self.quantity)
        elif self.side == TradeSide.SELL and self.quantity > 0:
            self.quantity = -abs(self.quantity)

    @property
    def net_value(self) -> float:
        """Net value of the trade including transaction costs."""
        base_value = self.quantity * self.price
        if self.side == TradeSide.BUY:
            return base_value + self.transaction_cost  # Cost increases for buys
        else:
            return base_value - self.transaction_cost  # Cost reduces proceeds for sells

    @property
    def is_buy(self) -> bool:
        """True if this is a buy trade."""
        return self.side == TradeSide.BUY

    @property
    def is_sell(self) -> bool:
        """True if this is a sell trade."""
        return self.side == TradeSide.SELL

    def to_dict(self) -> Dict[str, Any]:
        """Convert trade record to dictionary for serialization."""
        return {
            "date": self.date,
            "asset": self.asset,
            "quantity": self.quantity,
            "price": self.price,
            "side": self.side.value,
            "strategy_id": self.strategy_id,
            "allocated_capital": self.allocated_capital,
            "transaction_cost": self.transaction_cost,
            "trade_value": self.trade_value,
            "net_value": self.net_value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TradeRecord:
        """Create trade record from dictionary."""
        return cls(
            date=pd.Timestamp(data["date"]),
            asset=data["asset"],
            quantity=data["quantity"],
            price=data["price"],
            side=TradeSide(data["side"]),
            strategy_id=data["strategy_id"],
            allocated_capital=data["allocated_capital"],
            transaction_cost=data.get("transaction_cost", 0.0),
            trade_value=data.get("trade_value"),
            metadata=data.get("metadata"),
        )

    def __str__(self) -> str:
        """String representation of the trade record."""
        return (
            f"TradeRecord({self.date.date()}, {self.asset}, "
            f"{self.side.value.upper()} {abs(self.quantity):.2f} @ ${self.price:.2f}, "
            f"Strategy: {self.strategy_id}, Value: ${self.trade_value:.2f})"
        )

    def __repr__(self) -> str:
        """Detailed representation of the trade record."""
        return (
            f"TradeRecord(date={self.date}, asset='{self.asset}', "
            f"quantity={self.quantity}, price={self.price}, side={self.side}, "
            f"strategy_id='{self.strategy_id}', allocated_capital={self.allocated_capital}, "
            f"transaction_cost={self.transaction_cost})"
        )


@dataclass
class PositionRecord:
    """
    Current position record for an asset within a meta strategy.

    This tracks the current holdings of an asset across all sub-strategies.
    """

    asset: str
    quantity: float  # Total quantity held (positive for long, negative for short)
    average_price: float  # Volume-weighted average price
    last_update: pd.Timestamp

    # Attribution to sub-strategies
    strategy_contributions: Dict[str, float]  # strategy_id -> quantity contributed

    @property
    def market_value(self) -> float:
        """Current market value of the position (requires current price)."""
        # Note: This would need current market price to calculate
        # For now, return book value
        return self.quantity * self.average_price

    @property
    def is_long(self) -> bool:
        """True if this is a long position."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """True if this is a short position."""
        return self.quantity < 0

    @property
    def is_flat(self) -> bool:
        """True if there is no position."""
        return abs(self.quantity) < 1e-8  # Account for floating point precision

    def add_trade(self, trade: TradeRecord) -> None:
        """
        Update position with a new trade.

        Args:
            trade: TradeRecord to add to this position
        """
        if trade.asset != self.asset:
            raise ValueError(f"Trade asset {trade.asset} doesn't match position asset {self.asset}")

        old_quantity = self.quantity
        new_quantity = old_quantity + trade.quantity

        # Handle average price calculation
        if abs(old_quantity) < 1e-8:
            # Starting from zero position - use trade price
            self.average_price = trade.price
        elif abs(new_quantity) < 1e-8:
            # Position closed - reset average price
            self.average_price = 0.0
        elif (old_quantity > 0 and trade.quantity > 0) or (old_quantity < 0 and trade.quantity < 0):
            # Adding to existing position in same direction - recalculate average
            old_value = old_quantity * self.average_price
            new_value = old_value + (trade.quantity * trade.price)
            self.average_price = new_value / new_quantity
        else:
            # Reducing position (opposite direction) - keep original average price
            # This is correct for partial closes
            pass

        self.quantity = new_quantity
        self.last_update = trade.date

        # Update strategy contributions
        if trade.strategy_id not in self.strategy_contributions:
            self.strategy_contributions[trade.strategy_id] = 0.0
        self.strategy_contributions[trade.strategy_id] += trade.quantity

    def to_dict(self) -> Dict[str, Any]:
        """Convert position record to dictionary."""
        return {
            "asset": self.asset,
            "quantity": self.quantity,
            "average_price": self.average_price,
            "last_update": self.last_update,
            "strategy_contributions": self.strategy_contributions,
            "market_value": self.market_value,
            "is_long": self.is_long,
            "is_short": self.is_short,
            "is_flat": self.is_flat,
        }

    def __str__(self) -> str:
        """String representation of the position."""
        position_type = "LONG" if self.is_long else "SHORT" if self.is_short else "FLAT"
        return (
            f"Position({self.asset}: {position_type} {abs(self.quantity):.2f} "
            f"@ ${self.average_price:.2f}, Value: ${self.market_value:.2f})"
        )
