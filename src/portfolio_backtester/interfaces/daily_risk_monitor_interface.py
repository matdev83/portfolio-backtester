"""
Daily Risk Monitor Interface

Defines interfaces for daily risk management monitoring services.
This provides abstractions for stop loss and take profit monitoring that operate
independently of strategy rebalancing schedules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..backtesting.position_tracker import PositionTracker
    from ..risk_management.stop_loss_handlers import BaseStopLoss
    from ..risk_management.take_profit_handlers import BaseTakeProfit


class IDailyRiskMonitor(ABC):
    """
    Interface for daily risk management monitoring services.

    This interface defines the contract for monitoring positions daily
    for risk management conditions, independent of strategy rebalancing schedules.
    """

    @abstractmethod
    def reset_monitoring_state(self) -> None:
        """Reset monitoring state (useful for testing or new backtests)."""
        pass


class IDailyStopLossMonitor(IDailyRiskMonitor):
    """
    Interface for daily stop loss monitoring services.

    Monitors positions daily for stop loss conditions and generates
    liquidation signals when conditions are met.
    """

    @abstractmethod
    def check_positions_for_stop_loss(
        self,
        current_date: pd.Timestamp,
        position_tracker: PositionTracker,
        current_prices: pd.Series,
        stop_loss_handler: BaseStopLoss,
        historical_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Check all current positions for stop loss conditions and generate liquidation signals.

        Args:
            current_date: Current evaluation date
            position_tracker: Position tracker with current positions
            current_prices: Current asset prices for stop loss evaluation
            stop_loss_handler: Strategy's stop loss handler
            historical_data: Historical OHLC data for stop loss calculations

        Returns:
            DataFrame with liquidation signals for positions that triggered stop loss.
            Empty DataFrame if no positions need liquidation.
        """
        pass


class IDailyTakeProfitMonitor(IDailyRiskMonitor):
    """
    Interface for daily take profit monitoring services.

    Monitors positions daily for take profit conditions and generates
    liquidation signals when conditions are met.
    """

    @abstractmethod
    def check_positions_for_take_profit(
        self,
        current_date: pd.Timestamp,
        position_tracker: PositionTracker,
        current_prices: pd.Series,
        take_profit_handler: BaseTakeProfit,
        historical_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Check all current positions for take profit conditions and generate liquidation signals.

        Args:
            current_date: Current evaluation date
            position_tracker: Position tracker with current positions
            current_prices: Current asset prices for take profit evaluation
            take_profit_handler: Strategy's take profit handler
            historical_data: Historical OHLC data for take profit calculations

        Returns:
            DataFrame with liquidation signals for positions that triggered take profit.
            Empty DataFrame if no positions need liquidation.
        """
        pass


class IDailyRiskMonitorFactory(ABC):
    """
    Factory interface for creating daily risk monitors.

    Provides dependency injection for risk monitoring services.
    """

    @abstractmethod
    def create_stop_loss_monitor(self) -> IDailyStopLossMonitor:
        """Create a stop loss monitor instance."""
        pass

    @abstractmethod
    def create_take_profit_monitor(self) -> IDailyTakeProfitMonitor:
        """Create a take profit monitor instance."""
        pass


class DefaultDailyRiskMonitorFactory(IDailyRiskMonitorFactory):
    """
    Default implementation of daily risk monitor factory.

    Creates standard daily risk monitors for production use.
    """

    def create_stop_loss_monitor(self) -> IDailyStopLossMonitor:
        """Create a default stop loss monitor instance."""
        from ..risk_management.daily_stop_loss_monitor import DailyStopLossMonitor

        return DailyStopLossMonitor()

    def create_take_profit_monitor(self) -> IDailyTakeProfitMonitor:
        """Create a default take profit monitor instance."""
        from ..risk_management.daily_take_profit_monitor import DailyTakeProfitMonitor

        return DailyTakeProfitMonitor()
