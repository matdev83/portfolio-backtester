from __future__ import annotations
from .atr_service import calculate_atr_fast

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    # from ..features.atr import ATRFeature # No longer needed
    pass  # For type hinting if needed


class BaseStopLoss(ABC):
    """
    Abstract base class for stop-loss handlers.
    """

    def __init__(
        self, strategy_config: Dict[str, Any], stop_loss_specific_config: Dict[str, Any]
    ):  # Updated type hints
        """
        Initializes the stop-loss handler.

        Args:
            strategy_config: The overall configuration for the strategy.
            stop_loss_specific_config: Configuration specific to this stop-loss handler,
                                       e.g., {"type": "AtrBasedStopLoss", "atr_length": 14, ...}.
        """
        self.strategy_config = strategy_config
        self.stop_loss_specific_config = stop_loss_specific_config

    # Removed get_required_features
    # @abstractmethod
    # def get_required_features(self) -> Set[Feature]:
    #     """
    #     Returns a set of features required by this stop-loss handler.
    #     """
    #     pass

    @abstractmethod
    def calculate_stop_levels(
        self,
        current_date: pd.Timestamp,
        asset_ohlc_history: pd.DataFrame,  # Changed from prices_history, removed features
        current_weights: pd.Series,
        entry_prices: pd.Series,
    ) -> pd.Series:
        """
        Calculates the stop-loss price levels for each asset.

        Args:
            current_date: The current date for which to calculate stop levels.
            asset_ohlc_history: DataFrame of historical OHLCV data for all assets up to current_date.
            current_weights: Series of current weights of assets in the portfolio.
                             Positive for long, negative for short, zero if no position.
            entry_prices: Series of entry prices for the current positions. NaN if no position
                          or entry price is not tracked.

        Returns:
            A pandas Series with asset tickers as index and stop-loss price levels as values.
            NaN if no stop-loss is applicable for an asset.
        """
        pass

    @abstractmethod
    def apply_stop_loss(
        self,
        current_date: pd.Timestamp,
        current_asset_prices: pd.Series,  # Changed from prices_for_current_date (DataFrame) to Series
        target_weights: pd.Series,
        entry_prices: pd.Series,
        stop_levels: pd.Series,
    ) -> pd.Series:
        """
        Adjusts target weights if stop-loss conditions are met.

        Args:
            current_date: The specific date for which the stop loss is being checked.
            current_asset_prices: Series of current prices (e.g., 'Close') for assets, for current_date.
            target_weights: The target weights proposed by the main strategy logic.
            entry_prices: Series of entry prices for current positions.
            stop_levels: Series of stop-loss price levels calculated by `calculate_stop_levels`.

        Returns:
            A pandas Series with adjusted weights after applying stop-loss logic.
            Assets hitting their stop-loss should have their weights set to 0.
        """
        pass


class NoStopLoss(BaseStopLoss):
    """
    A stop-loss handler that does not implement any stop-loss logic.
    """

    # Removed get_required_features
    # def get_required_features(self) -> Set[Feature]:
    #     return set()

    def calculate_stop_levels(
        self,
        current_date: pd.Timestamp,
        asset_ohlc_history: pd.DataFrame,  # Updated signature
        current_weights: pd.Series,
        entry_prices: pd.Series,
    ) -> pd.Series:
        return pd.Series(np.nan, index=current_weights.index)  # No features dict needed

    def apply_stop_loss(
        self,
        current_date: pd.Timestamp,
        current_asset_prices: pd.Series,  # Updated signature
        target_weights: pd.Series,
        entry_prices: pd.Series,
        stop_levels: pd.Series,
    ) -> pd.Series:
        return target_weights


class AtrBasedStopLoss(BaseStopLoss):
    """
    A stop-loss handler that uses Average True Range (ATR) to set stop-loss levels.
    """

    def __init__(self, strategy_config: dict, stop_loss_specific_config: dict):
        super().__init__(strategy_config, stop_loss_specific_config)
        self.atr_length = self.stop_loss_specific_config.get("atr_length", 14)
        self.atr_multiple = self.stop_loss_specific_config.get("atr_multiple", 2.5)

    def get_required_features(self):
        from ..features.atr import (
            ATRFeature,
        )  # Import here to avoid circular dependency at module load time

        return {ATRFeature(atr_period=self.atr_length)}

    def calculate_stop_levels(
        self,
        current_date: pd.Timestamp,
        asset_ohlc_history: pd.DataFrame,  # Updated to match base class signature
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

        stop_levels = pd.Series(np.nan, index=cw.index)

        # Only consider assets with open positions and valid entry prices
        # After fallback ATR calculation, any asset with position and valid entry price should have ATR
        open_mask = (cw != 0) & (~pd.isna(ep))

        long_mask = open_mask & (cw > 0)
        short_mask = open_mask & (cw < 0)

        # Vectorized stop level calculation
        stop_levels[long_mask] = ep[long_mask] - (atr[long_mask] * self.atr_multiple)
        stop_levels[short_mask] = ep[short_mask] + (atr[short_mask] * self.atr_multiple)

        return stop_levels.reindex(current_weights.index)

    def apply_stop_loss(
        self,
        current_date: pd.Timestamp,
        current_asset_prices: pd.Series,
        target_weights: pd.Series,
        entry_prices: pd.Series,
        stop_levels: pd.Series,
    ) -> pd.Series:
        # Align all inputs to a common index to avoid label mismatches
        idx = target_weights.index.union(current_asset_prices.index).union(stop_levels.index)
        tw = target_weights.reindex(idx)
        prices = current_asset_prices.reindex(idx)
        stops = stop_levels.reindex(idx)

        # Vectorized stop loss application on aligned data
        adjusted_weights = tw.copy()
        valid_mask = (~pd.isna(stops)) & (tw != 0) & (~pd.isna(prices))
        long_mask = valid_mask & (tw > 0) & (prices <= stops)
        short_mask = valid_mask & (tw < 0) & (prices >= stops)
        adjusted_weights[long_mask | short_mask] = 0.0

        # Return only for original target_weights index
        return adjusted_weights.reindex(target_weights.index).fillna(target_weights)


# Example of how ATRFeature might be defined (will be in feature.py)
# from ..features.base import Feature
# class ATRFeature(Feature):
#     def __init__(self, atr_period: int):
#         super().__init__(params={"atr_period": atr_period})
#         self.atr_period = atr_period
#         self.name = f"atr_{self.atr_period}"

#     def compute(self, data: pd.DataFrame, benchmark_data: pd.Series | None = None) -> pd.DataFrame:
#         # data is expected to be a DataFrame with 'High', 'Low', 'Close' columns for each asset
#         # This computation needs to be adapted based on how data is provided (single asset or multi-asset)
#         # For multi-asset, it should iterate over columns if they are assets, or expect multi-index.
# NOTE: ATRFeature implementation has been moved to src/portfolio_backtester/features/atr.py
# This commented code block was removed as part of code cleanup.
