"""
Daily Stop Loss Monitoring Service

This service provides schedule-independent daily monitoring of positions for stop loss conditions.
It operates independently of strategy rebalancing schedules, ensuring positions are monitored
and liquidated daily when stop loss conditions are met.

Key Features:
- Works with any strategy type (signal-based or portfolio-based)
- Operates regardless of strategy rebalance schedule (monthly, quarterly, etc.)
- Uses strategy's configured stop loss handler
- Generates liquidation signals for triggered positions
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, TYPE_CHECKING

import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from ..backtesting.position_tracker import Position, PositionTracker
    from ..risk_management.stop_loss_handlers import BaseStopLoss

from ..interfaces.daily_risk_monitor_interface import IDailyStopLossMonitor

logger = logging.getLogger(__name__)


class DailyStopLossMonitor(IDailyStopLossMonitor):
    """
    Service for monitoring positions daily for stop loss conditions.

    This service operates independently of strategy rebalancing schedules,
    ensuring positions are checked daily regardless of when the strategy
    last rebalanced or will next rebalance.
    """

    def __init__(self):
        """Initialize the daily stop loss monitor."""
        self.last_check_date: Optional[pd.Timestamp] = None
        self.triggered_positions: Dict[str, pd.Timestamp] = (
            {}
        )  # Track when positions were stopped out
        self.debug_enabled = logger.isEnabledFor(logging.DEBUG)

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
        if (
            not hasattr(position_tracker, "current_positions")
            or not position_tracker.current_positions
        ):
            self._log_debug(f"No positions to monitor on {current_date.date()}")
            return pd.DataFrame()

        # Get current positions and their details
        positions = position_tracker.current_positions
        current_weights = self._extract_current_weights(positions)
        entry_prices = self._extract_entry_prices(positions)

        if current_weights.empty or entry_prices.empty:
            self._log_debug(f"No valid positions with entry prices on {current_date.date()}")
            return pd.DataFrame()

        # Calculate stop levels using strategy's stop loss handler
        try:
            stop_levels = stop_loss_handler.calculate_stop_levels(
                current_date=current_date,
                asset_ohlc_history=historical_data,
                current_weights=current_weights,
                entry_prices=entry_prices,
            )
        except Exception as e:
            logger.error(f"Error calculating stop levels on {current_date.date()}: {e}")
            return pd.DataFrame()

        # Apply stop loss logic to determine which positions to liquidate
        try:
            weights_after_stop_loss = stop_loss_handler.apply_stop_loss(
                current_date=current_date,
                current_asset_prices=current_prices,
                target_weights=current_weights,
                entry_prices=entry_prices,
                stop_levels=stop_levels,
            )
        except Exception as e:
            logger.error(f"Error applying stop loss on {current_date.date()}: {e}")
            return pd.DataFrame()

        # Identify positions that were stopped out
        liquidated_positions = self._identify_liquidated_positions(
            current_weights, weights_after_stop_loss, current_date
        )

        # Generate liquidation signals
        if not liquidated_positions.empty:
            liquidation_signals = self._generate_liquidation_signals(
                liquidated_positions, current_date
            )
            self._log_stop_loss_triggers(
                liquidated_positions, current_date, current_prices, stop_levels
            )
            return liquidation_signals
        else:
            self._log_debug(f"No stop loss triggers on {current_date.date()}")
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
                # If no entry price available, use NaN (stop loss handler should handle this)
                entry_prices_dict[ticker] = np.nan

        return pd.Series(entry_prices_dict, dtype=float)

    def _identify_liquidated_positions(
        self,
        original_weights: pd.Series,
        weights_after_stop_loss: pd.Series,
        current_date: pd.Timestamp,
    ) -> pd.Series:
        """Identify positions that were liquidated due to stop loss."""
        was_open = (original_weights != 0) & (~pd.isna(original_weights))
        is_closed = (weights_after_stop_loss == 0) | pd.isna(weights_after_stop_loss)

        # Align indices to avoid label mismatches during comparison
        was_open, is_closed = was_open.align(is_closed, join="outer", fill_value=False)
        liquidated_mask = was_open & is_closed

        # Align original_weights to mask index
        original_weights_aligned = original_weights.reindex(liquidated_mask.index).fillna(0.0)
        liquidated_positions = original_weights_aligned[liquidated_mask]

        for ticker in liquidated_positions.index:
            self.triggered_positions[ticker] = current_date

        return liquidated_positions

    def _generate_liquidation_signals(
        self, liquidated_positions: pd.Series, current_date: pd.Timestamp
    ) -> pd.DataFrame:
        """Generate signals to liquidate positions that triggered stop loss."""
        if liquidated_positions.empty:
            return pd.DataFrame()

        liquidation_weights = pd.Series(0.0, index=liquidated_positions.index)
        return pd.DataFrame([liquidation_weights], index=[current_date])

    def _log_stop_loss_triggers(
        self,
        liquidated_positions: pd.Series,
        current_date: pd.Timestamp,
        current_prices: pd.Series,
        stop_levels: pd.Series,
    ) -> None:
        """Log stop loss trigger events for debugging and monitoring."""
        for ticker in liquidated_positions.index:
            current_price = current_prices.get(ticker, "N/A")
            stop_level = stop_levels.get(ticker, "N/A")
            original_weight = liquidated_positions[ticker]

            logger.info(
                f"STOP LOSS TRIGGERED: {ticker} on {current_date.date()} "
                f"(weight: {original_weight:.4f}, price: {current_price}, stop: {stop_level})"
            )

    def _log_debug(self, message: str) -> None:
        """Log debug message if debug logging is enabled."""
        if self.debug_enabled:
            logger.debug(f"DailyStopLossMonitor: {message}")

    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get statistics about stop loss monitoring activity."""
        return {
            "last_check_date": self.last_check_date,
            "total_triggered_positions": len(self.triggered_positions),
            "triggered_positions": dict(self.triggered_positions),
        }

    def reset_monitoring_state(self) -> None:
        """Reset monitoring state (useful for testing or new backtests)."""
        self.last_check_date = None
        self.triggered_positions.clear()
        logger.debug("DailyStopLossMonitor: Reset monitoring state")
