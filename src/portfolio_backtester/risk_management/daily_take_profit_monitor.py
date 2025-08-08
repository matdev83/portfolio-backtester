"""
Daily Take Profit Monitoring Service

This service provides schedule-independent daily monitoring of positions for take profit conditions.
It operates independently of strategy rebalancing schedules, ensuring positions are monitored
and liquidated daily when take profit conditions are met.

Key Features:
- Works with any strategy type (signal-based or portfolio-based)
- Operates regardless of strategy rebalance schedule (monthly, quarterly, etc.)
- Uses strategy's configured take profit handler
- Generates liquidation signals for triggered positions
- Monitors for favorable price excursions (profits)
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, TYPE_CHECKING

import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from ..backtesting.position_tracker import Position, PositionTracker
    from ..risk_management.take_profit_handlers import BaseTakeProfit

from ..interfaces.daily_risk_monitor_interface import IDailyTakeProfitMonitor

logger = logging.getLogger(__name__)


class DailyTakeProfitMonitor(IDailyTakeProfitMonitor):
    """
    Service for monitoring positions daily for take profit conditions.

    This service operates independently of strategy rebalancing schedules,
    ensuring positions are checked daily regardless of when the strategy
    last rebalanced or will next rebalance.

    Take profit monitors favorable price movements and closes positions
    when profit targets are achieved.
    """

    def __init__(self):
        """Initialize the daily take profit monitor."""
        self.last_check_date: Optional[pd.Timestamp] = None
        self.triggered_positions: Dict[str, pd.Timestamp] = (
            {}
        )  # Track when positions were closed for profit
        self.debug_enabled = logger.isEnabledFor(logging.DEBUG)

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
        # Check if position tracker has valid current positions
        try:
            positions = getattr(position_tracker, "current_positions", None)
            if not positions or not isinstance(positions, dict):
                self._log_debug(f"No positions to monitor on {current_date.date()}")
                return pd.DataFrame()
        except (AttributeError, TypeError):
            self._log_debug(f"No positions to monitor on {current_date.date()}")
            return pd.DataFrame()
        current_weights = self._extract_current_weights(positions)
        entry_prices = self._extract_entry_prices(positions)

        if current_weights.empty or entry_prices.empty:
            self._log_debug(f"No valid positions with entry prices on {current_date.date()}")
            return pd.DataFrame()

        # Calculate take profit levels using strategy's take profit handler
        try:
            take_profit_levels = take_profit_handler.calculate_take_profit_levels(
                current_date=current_date,
                asset_ohlc_history=historical_data,
                current_weights=current_weights,
                entry_prices=entry_prices,
            )
        except Exception as e:
            logger.error(f"Error calculating take profit levels on {current_date.date()}: {e}")
            return pd.DataFrame()

        # Apply take profit logic to determine which positions to liquidate
        try:
            weights_after_take_profit = take_profit_handler.apply_take_profit(
                current_date=current_date,
                current_asset_prices=current_prices,
                target_weights=current_weights,
                entry_prices=entry_prices,
                take_profit_levels=take_profit_levels,
            )
        except Exception as e:
            logger.error(f"Error applying take profit on {current_date.date()}: {e}")
            return pd.DataFrame()

        # Identify positions that were closed for profit
        liquidated_positions = self._identify_liquidated_positions(
            current_weights, weights_after_take_profit, current_date
        )

        # Generate liquidation signals
        if not liquidated_positions.empty:
            liquidation_signals = self._generate_liquidation_signals(
                liquidated_positions, current_date
            )
            self._log_take_profit_triggers(
                liquidated_positions, current_date, current_prices, take_profit_levels
            )
            return liquidation_signals
        else:
            self._log_debug(f"No take profit triggers on {current_date.date()}")
            return pd.DataFrame()

    def _extract_current_weights(self, positions: Dict[str, Position]) -> pd.Series:
        """Extract current weights from position tracker positions."""
        if not positions:
            return pd.Series(dtype=float)

        weights_dict = {ticker: pos.weight for ticker, pos in positions.items()}
        return pd.Series(weights_dict, dtype=float)

    def _extract_entry_prices(self, positions: Dict[str, Position]) -> pd.Series:
        """Extract entry prices from position tracker positions."""
        if not positions:
            return pd.Series(dtype=float)

        entry_prices_dict = {}
        for ticker, pos in positions.items():
            if pos.entry_price is not None:
                entry_prices_dict[ticker] = pos.entry_price
            else:
                # If no entry price available, use NaN (take profit handler should handle this)
                entry_prices_dict[ticker] = np.nan

        return pd.Series(entry_prices_dict, dtype=float)

    def _identify_liquidated_positions(
        self,
        original_weights: pd.Series,
        weights_after_take_profit: pd.Series,
        current_date: pd.Timestamp,
    ) -> pd.Series:
        """Identify positions that were liquidated due to take profit."""
        # Find positions that went from non-zero to zero (take profit triggered)
        was_open = (original_weights != 0) & (~pd.isna(original_weights))
        is_closed = (weights_after_take_profit == 0) | pd.isna(weights_after_take_profit)

        liquidated_mask = was_open & is_closed
        liquidated_positions = original_weights[liquidated_mask]

        # Track liquidated positions to avoid duplicate processing
        for ticker in liquidated_positions.index:
            self.triggered_positions[ticker] = current_date

        return liquidated_positions

    def _generate_liquidation_signals(
        self, liquidated_positions: pd.Series, current_date: pd.Timestamp
    ) -> pd.DataFrame:
        """Generate signals to liquidate positions that triggered take profit."""
        if liquidated_positions.empty:
            return pd.DataFrame()

        # Create liquidation signals (set all triggered positions to zero weight)
        liquidation_weights = pd.Series(0.0, index=liquidated_positions.index)

        # Return as DataFrame with current date as index (matching strategy signal format)
        return pd.DataFrame([liquidation_weights], index=[current_date])

    def _log_take_profit_triggers(
        self,
        liquidated_positions: pd.Series,
        current_date: pd.Timestamp,
        current_prices: pd.Series,
        take_profit_levels: pd.Series,
    ) -> None:
        """Log take profit trigger events for debugging and monitoring."""
        for ticker in liquidated_positions.index:
            current_price = current_prices.get(ticker, "N/A")
            take_profit_level = take_profit_levels.get(ticker, "N/A")
            original_weight = liquidated_positions[ticker]

            logger.info(
                f"TAKE PROFIT TRIGGERED: {ticker} on {current_date.date()} "
                f"(weight: {original_weight:.4f}, price: {current_price}, target: {take_profit_level})"
            )

    def _log_debug(self, message: str) -> None:
        """Log debug message if debug logging is enabled."""
        if self.debug_enabled:
            logger.debug(f"DailyTakeProfitMonitor: {message}")

    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get statistics about take profit monitoring activity."""
        return {
            "last_check_date": self.last_check_date,
            "total_triggered_positions": len(self.triggered_positions),
            "triggered_positions": dict(self.triggered_positions),
        }

    def reset_monitoring_state(self) -> None:
        """Reset monitoring state (useful for testing or new backtests)."""
        self.last_check_date = None
        self.triggered_positions.clear()
        logger.debug("DailyTakeProfitMonitor: Reset monitoring state")
