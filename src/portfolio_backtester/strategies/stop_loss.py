from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Set

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    # from ..features.atr import ATRFeature # No longer needed
    pass # For type hinting if needed

class BaseStopLoss(ABC):
    """
    Abstract base class for stop-loss handlers.
    """
    def __init__(self, strategy_config: Dict[str, Any], stop_loss_specific_config: Dict[str, Any]): # Updated type hints
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
        asset_ohlc_history: pd.DataFrame, # Changed from prices_history, removed features
        current_weights: pd.Series,
        entry_prices: pd.Series
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
        current_asset_prices: pd.Series, # Changed from prices_for_current_date (DataFrame) to Series
        target_weights: pd.Series,
        entry_prices: pd.Series,
        stop_levels: pd.Series
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
        asset_ohlc_history: pd.DataFrame, # Updated signature
        current_weights: pd.Series,
        entry_prices: pd.Series
    ) -> pd.Series:
        return pd.Series(np.nan, index=current_weights.index) # No features dict needed

    def apply_stop_loss(
        self,
        current_date: pd.Timestamp,
        current_asset_prices: pd.Series, # Updated signature
        target_weights: pd.Series,
        entry_prices: pd.Series,
        stop_levels: pd.Series
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
        self._atr_cache = {}  # Cache for ATR calculations

    def get_required_features(self):
        from ..features.atr import ATRFeature # Import here to avoid circular dependency at module load time
        return {ATRFeature(atr_period=self.atr_length)}

    def calculate_stop_levels(
        self,
        current_date: pd.Timestamp,
        asset_ohlc_history: pd.DataFrame, # Updated to match base class signature
        current_weights: pd.Series,
        entry_prices: pd.Series
    ) -> pd.Series:
        # Vectorized ATR calculation
        atr_values_for_date = self._calculate_atr(asset_ohlc_history, current_date)
        stop_levels = pd.Series(np.nan, index=current_weights.index)
        # Only consider assets with open positions and valid entry prices
        open_mask = (current_weights != 0) & (~pd.isna(entry_prices)) & (~pd.isna(atr_values_for_date))
        long_mask = open_mask & (current_weights > 0)
        short_mask = open_mask & (current_weights < 0)
        # Vectorized stop level calculation
        stop_levels[long_mask] = entry_prices[long_mask] - (atr_values_for_date[long_mask] * self.atr_multiple)
        stop_levels[short_mask] = entry_prices[short_mask] + (atr_values_for_date[short_mask] * self.atr_multiple)
        return stop_levels

    def _calculate_atr(self, asset_ohlc_history: pd.DataFrame, current_date: pd.Timestamp) -> pd.Series:
        """Vectorized ATR calculation for all assets as of current_date."""
        if asset_ohlc_history is None or asset_ohlc_history.empty:
            tickers = []
            if hasattr(asset_ohlc_history, 'columns'):
                if isinstance(asset_ohlc_history.columns, pd.MultiIndex) and 'Ticker' in asset_ohlc_history.columns.names:
                    tickers = asset_ohlc_history.columns.get_level_values('Ticker').unique()
                else:
                    tickers = asset_ohlc_history.columns
            return pd.Series(np.nan, index=tickers)

        cache_key = (str(current_date), self.atr_length, len(asset_ohlc_history))
        if cache_key in self._atr_cache:
            return self._atr_cache[cache_key]
        ohlc_data = asset_ohlc_history[asset_ohlc_history.index <= current_date]
        if len(ohlc_data) < self.atr_length:
            if hasattr(asset_ohlc_history, 'columns'):
                if isinstance(asset_ohlc_history.columns, pd.MultiIndex) and 'Ticker' in asset_ohlc_history.columns.names:
                    tickers = asset_ohlc_history.columns.get_level_values('Ticker').unique()
                else:
                    tickers = asset_ohlc_history.columns
            else:
                tickers = []
            result = pd.Series(np.nan, index=tickers)
            self._atr_cache[cache_key] = result
            return result
        if isinstance(ohlc_data.columns, pd.MultiIndex) and 'Field' in ohlc_data.columns.names:
            tickers = ohlc_data.columns.get_level_values('Ticker').unique()
            highs = ohlc_data.xs('High', level='Field', axis=1)
            lows = ohlc_data.xs('Low', level='Field', axis=1)
            closes = ohlc_data.xs('Close', level='Field', axis=1)
            prev_closes = closes.shift(1)
            tr1 = highs - lows
            tr2 = (highs - prev_closes).abs()
            tr3 = (lows - prev_closes).abs()
            true_range = np.fmax(tr1, np.fmax(tr2, tr3))
            atr = pd.DataFrame(true_range, index=highs.index, columns=highs.columns).rolling(window=self.atr_length, min_periods=self.atr_length).mean()
            if current_date in atr.index:
                atr_today = atr.loc[current_date]
                if isinstance(atr_today, pd.Series):
                    result = atr_today
                else:
                    result = pd.Series(np.nan, index=tickers)
            else:
                result = pd.Series(np.nan, index=tickers)
        else:
            closes = ohlc_data
            returns = closes.pct_change(fill_method=None)
            rolling_std = returns.rolling(window=self.atr_length, min_periods=self.atr_length).std()
            if current_date in closes.index and current_date in rolling_std.index:
                current_prices = closes.loc[current_date]
                std_today = rolling_std.loc[current_date]
                atr_today = current_prices * std_today
                result = atr_today
            else:
                result = pd.Series(np.nan, index=closes.columns)
        # Ensure result is always a Series, never a DataFrame
        if isinstance(result, pd.DataFrame):
            # Return a Series of NaN with the expected index
            if hasattr(result, 'columns'):
                return pd.Series(np.nan, index=result.columns)
            return pd.Series(np.nan)
        self._atr_cache[cache_key] = result
        return result.astype(float)
    def apply_stop_loss(
        self,
        current_date: pd.Timestamp,
        current_asset_prices: pd.Series,
        target_weights: pd.Series,
        entry_prices: pd.Series,
        stop_levels: pd.Series
    ) -> pd.Series:
        # Vectorized stop loss application
        adjusted_weights = target_weights.copy()
        valid_mask = (~pd.isna(stop_levels)) & (target_weights != 0) & (~pd.isna(current_asset_prices))
        long_mask = valid_mask & (target_weights > 0) & (current_asset_prices <= stop_levels)
        short_mask = valid_mask & (target_weights < 0) & (current_asset_prices >= stop_levels)
        adjusted_weights[long_mask | short_mask] = 0.0
        return adjusted_weights

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
