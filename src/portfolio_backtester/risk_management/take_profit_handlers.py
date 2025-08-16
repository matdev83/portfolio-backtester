"""
Take profit handlers for portfolio backtesting.

This module provides various take profit mechanisms that can be applied
to trading positions to lock in profits at predetermined levels.
Optimized to use shared ATR service for performance.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from .atr_service import calculate_atr_fast


class BaseTakeProfit(ABC):
    """
    Abstract base class for take profit handlers.

    All take profit implementations should inherit from this class
    and implement the required abstract methods.
    """

    def __init__(self, strategy_config: dict, take_profit_specific_config: dict):
        self.strategy_config = strategy_config
        self.take_profit_specific_config = take_profit_specific_config

    @abstractmethod
    def calculate_take_profit_levels(
        self,
        current_date: pd.Timestamp,
        asset_ohlc_history: pd.DataFrame,
        current_weights: pd.Series,
        entry_prices: pd.Series,
    ) -> pd.Series:
        """
        Calculate take profit levels for current positions.

        Args:
            current_date: Current evaluation date
            asset_ohlc_history: Historical OHLC data for assets
            current_weights: Current position weights
            entry_prices: Entry prices for current positions

        Returns:
            Series containing take profit levels for each asset
        """
        pass

    @abstractmethod
    def apply_take_profit(
        self,
        current_date: pd.Timestamp,
        current_asset_prices: pd.Series,
        target_weights: pd.Series,
        entry_prices: pd.Series,
        take_profit_levels: pd.Series,
    ) -> pd.Series:
        """
        Apply take profit logic and return adjusted weights.

        Args:
            current_date: Current evaluation date
            current_asset_prices: Current asset prices
            target_weights: Target position weights
            entry_prices: Entry prices for positions
            take_profit_levels: Pre-calculated take profit levels

        Returns:
            Series containing adjusted weights after take profit application
        """
        pass


class NoTakeProfit(BaseTakeProfit):
    """
    A take-profit handler that does nothing (no take profit applied).

    This is useful for strategies that don't want any take profit mechanism.
    """

    def calculate_take_profit_levels(
        self,
        current_date: pd.Timestamp,
        asset_ohlc_history: pd.DataFrame,
        current_weights: pd.Series,
        entry_prices: pd.Series,
    ) -> pd.Series:
        return pd.Series(np.nan, index=current_weights.index)

    def apply_take_profit(
        self,
        current_date: pd.Timestamp,
        current_asset_prices: pd.Series,
        target_weights: pd.Series,
        entry_prices: pd.Series,
        take_profit_levels: pd.Series,
    ) -> pd.Series:
        return target_weights


class AtrBasedTakeProfit(BaseTakeProfit):
    """
    A take-profit handler that uses Average True Range (ATR) to set take-profit levels.

    For long positions: Take profit when price rises above entry_price + (ATR * atr_multiple)
    For short positions: Take profit when price falls below entry_price - (ATR * atr_multiple)
    """

    def __init__(self, strategy_config: dict, take_profit_specific_config: dict):
        super().__init__(strategy_config, take_profit_specific_config)
        self.atr_length = self.take_profit_specific_config.get("atr_length", 14)
        self.atr_multiple = self.take_profit_specific_config.get("atr_multiple", 2.0)

    def calculate_take_profit_levels(
        self,
        current_date: pd.Timestamp,
        asset_ohlc_history: pd.DataFrame,
        current_weights: pd.Series,
        entry_prices: pd.Series,
    ) -> pd.Series:
        # Fast ATR calculation using optimized service
        atr_values_for_date = calculate_atr_fast(asset_ohlc_history, current_date, self.atr_length)

        # Align all series to a common index
        idx = current_weights.index.union(entry_prices.index)
        if isinstance(atr_values_for_date, pd.Series):
            idx = idx.union(atr_values_for_date.index)

        cw = current_weights.reindex(idx)
        ep = entry_prices.reindex(idx)
        atr = (
            atr_values_for_date.reindex(idx)
            if isinstance(atr_values_for_date, pd.Series)
            else pd.Series(atr_values_for_date, index=idx)
        )
        
        # For assets with positions and valid entry prices but missing ATR,
        # use a fallback ATR value based on entry price volatility assumption
        fallback_mask = (cw != 0) & (~pd.isna(ep)) & pd.isna(atr)
        if fallback_mask.any():
            # Default to 2% of entry price as fallback ATR
            fallback_atr = ep * 0.02
            atr[fallback_mask] = fallback_atr[fallback_mask]

        take_profit_levels = pd.Series(np.nan, index=cw.index)

        # Only consider assets with open positions and valid entry prices
        # After fallback ATR calculation, any asset with position and valid entry price should have ATR
        open_mask = (cw != 0) & (~pd.isna(ep))
        long_mask = open_mask & (cw > 0)
        short_mask = open_mask & (cw < 0)

        # Vectorized take profit level calculation (opposite logic from stop loss)
        # For long positions: take profit when price goes UP by ATR multiple
        take_profit_levels[long_mask] = ep[long_mask] + (atr[long_mask] * self.atr_multiple)
        # For short positions: take profit when price goes DOWN by ATR multiple
        take_profit_levels[short_mask] = ep[short_mask] - (atr[short_mask] * self.atr_multiple)

        return take_profit_levels.reindex(current_weights.index)

    def apply_take_profit(
        self,
        current_date: pd.Timestamp,
        current_asset_prices: pd.Series,
        target_weights: pd.Series,
        entry_prices: pd.Series,
        take_profit_levels: pd.Series,
    ) -> pd.Series:
        # Vectorized take profit application
        adjusted_weights = target_weights.copy()
        valid_mask = (
            (~pd.isna(take_profit_levels))
            & (target_weights != 0)
            & (~pd.isna(current_asset_prices))
        )

        if not valid_mask.any():
            return adjusted_weights

        # Check for take profit triggers
        long_positions = target_weights > 0
        short_positions = target_weights < 0

        # For long positions: close if current price >= take profit level
        long_take_profit_trigger = (
            valid_mask & long_positions & (current_asset_prices >= take_profit_levels)
        )

        # For short positions: close if current price <= take profit level
        short_take_profit_trigger = (
            valid_mask & short_positions & (current_asset_prices <= take_profit_levels)
        )

        # Close positions that hit take profit
        take_profit_trigger = long_take_profit_trigger | short_take_profit_trigger
        adjusted_weights[take_profit_trigger] = 0.0

        return adjusted_weights


# Factory function to create take profit handlers
def create_take_profit_handler(strategy_config: dict) -> BaseTakeProfit:
    """
    Factory function to create appropriate take profit handler based on configuration.

    Args:
        strategy_config: Strategy configuration dictionary

    Returns:
        Appropriate take profit handler instance
    """
    take_profit_config = strategy_config.get("take_profit_config", {})
    take_profit_type = take_profit_config.get("type", "NoTakeProfit")

    if take_profit_type == "NoTakeProfit" or take_profit_type is None:
        return NoTakeProfit(strategy_config, take_profit_config)
    elif take_profit_type == "AtrBasedTakeProfit":
        return AtrBasedTakeProfit(strategy_config, take_profit_config)
    else:
        raise ValueError(f"Unknown take profit type: {take_profit_type}")
