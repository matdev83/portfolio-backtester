from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Any
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    # BaseStrategy import is not strictly needed here anymore for type hinting if methods are self-contained
    pass # from ..strategies.base_strategy import BaseStrategy

class BaseStopLoss(ABC):
    """
    Abstract base class for stop-loss handlers.
    """
    def __init__(self, strategy_config: Dict[str, Any], stop_loss_specific_config: Dict[str, Any]):
        self.strategy_config = strategy_config
        self.stop_loss_specific_config = stop_loss_specific_config

    @abstractmethod
    def calculate_stop_levels(
        self,
        current_date: pd.Timestamp,
        asset_ohlc_history: pd.DataFrame,
        current_weights: pd.Series,
        entry_prices: pd.Series
    ) -> pd.Series:
        pass

    @abstractmethod
    def apply_stop_loss(
        self,
        current_date: pd.Timestamp,
        current_asset_prices: pd.Series,
        target_weights: pd.Series,
        entry_prices: pd.Series, # entry_prices might not be used by all stop-loss types directly in apply
        stop_levels: pd.Series
    ) -> pd.Series:
        pass


class NoStopLoss(BaseStopLoss):
    """
    A stop-loss handler that does not implement any stop-loss logic.
    """
    def calculate_stop_levels(
        self,
        current_date: pd.Timestamp,
        asset_ohlc_history: pd.DataFrame,
        current_weights: pd.Series,
        entry_prices: pd.Series
    ) -> pd.Series:
        return pd.Series(np.nan, index=current_weights.index)

    def apply_stop_loss(
        self,
        current_date: pd.Timestamp,
        current_asset_prices: pd.Series,
        target_weights: pd.Series,
        entry_prices: pd.Series,
        stop_levels: pd.Series
    ) -> pd.Series:
        return target_weights


class AtrBasedStopLoss(BaseStopLoss):
    """
    A stop-loss handler that uses Average True Range (ATR) to set stop-loss levels.
    ATR is calculated internally using the provided OHLC history.
    """
    def __init__(self, strategy_config: dict, stop_loss_specific_config: dict):
        super().__init__(strategy_config, stop_loss_specific_config)
        self.atr_length = self.stop_loss_specific_config.get("atr_length", 14)
        self.atr_multiple = self.stop_loss_specific_config.get("atr_multiple", 2.5)

    def _calculate_single_atr(self, asset_ohlc_data: pd.DataFrame) -> pd.Series:
        """
        Calculates ATR for a single asset's OHLC data.
        asset_ohlc_data: DataFrame with 'High', 'Low', 'Close' columns for one asset, indexed by date.
        Returns a Series of ATR values, indexed by date.
        """
        if not all(col in asset_ohlc_data.columns for col in ['High', 'Low', 'Close']):
            # If essential columns are missing, return NaNs for ATR
            return pd.Series(np.nan, index=asset_ohlc_data.index)

        high_low = asset_ohlc_data['High'] - asset_ohlc_data['Low']
        high_close_prev = np.abs(asset_ohlc_data['High'] - asset_ohlc_data['Close'].shift(1))
        low_close_prev = np.abs(asset_ohlc_data['Low'] - asset_ohlc_data['Close'].shift(1))

        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_length, min_periods=self.atr_length).mean()
        return atr

    def calculate_stop_levels(
        self,
        current_date: pd.Timestamp,
        asset_ohlc_history: pd.DataFrame, # Expects MultiIndex columns (Asset, OHLCV) or wide format (Asset_OHLCV)
        current_weights: pd.Series,
        entry_prices: pd.Series
    ) -> pd.Series:
        stop_levels = pd.Series(np.nan, index=current_weights.index)

        # Assuming asset_ohlc_history columns are MultiIndex: level 0 for asset, level 1 for OHLC type
        # Or, if it's wide format, e.g., 'AAPL_High', 'AAPL_Low', 'AAPL_Close', etc.
        # For this implementation, let's assume MultiIndex for easier iteration.
        # If not, this part needs adjustment.

        # A robust way is to iterate through assets that have positions.
        for asset in current_weights.index:
            weight = current_weights.get(asset, 0)
            entry_price = entry_prices.get(asset)

            if pd.isna(entry_price) or weight == 0:
                continue

            # Extract OHLC data for the specific asset
            try:
                if isinstance(asset_ohlc_history.columns, pd.MultiIndex):
                    asset_data = asset_ohlc_history[asset] # Selects columns 'High', 'Low', 'Close' for 'asset'
                else: # Attempt wide format: asset_High, asset_Low, asset_Close
                    asset_data = asset_ohlc_history[[f'{asset}_High', f'{asset}_Low', f'{asset}_Close']].rename(
                        columns={f'{asset}_High':'High', f'{asset}_Low':'Low', f'{asset}_Close':'Close'}
                    )
                # Ensure data is up to current_date for ATR calculation
                asset_data_for_atr = asset_data[asset_data.index <= current_date]
            except KeyError:
                # print(f"Warning: OHLC data for asset {asset} not found in expected format. Skipping ATR stop for this asset.")
                continue

            if len(asset_data_for_atr) < self.atr_length:
                # print(f"Warning: Not enough data for asset {asset} to calculate ATR (length {self.atr_length}). Skipping.")
                continue

            atr_series = self._calculate_single_atr(asset_data_for_atr)

            if atr_series.empty or pd.isna(atr_series.iloc[-1]):
                # print(f"Warning: ATR value for asset {asset} on date {current_date} is NaN. Skipping stop for this asset.")
                continue

            atr_value = atr_series.iloc[-1] # Get the ATR for the current_date (or latest available)

            if weight > 0:  # Long position
                stop_levels.loc[asset] = entry_price - (atr_value * self.atr_multiple)
            elif weight < 0:  # Short position
                stop_levels.loc[asset] = entry_price + (atr_value * self.atr_multiple)

        return stop_levels

    def apply_stop_loss(
        self,
        current_date: pd.Timestamp,
        current_asset_prices: pd.Series, # Series of current CLOSE prices, indexed by asset
        target_weights: pd.Series,
        entry_prices: pd.Series,
        stop_levels: pd.Series
    ) -> pd.Series:
        adjusted_weights = target_weights.copy()

        for asset in target_weights.index:
            if pd.isna(stop_levels.get(asset)) or target_weights.get(asset, 0) == 0:
                continue

            current_price_for_asset = current_asset_prices.get(asset)

            if current_price_for_asset is None or pd.isna(current_price_for_asset):
                continue

            asset_stop_level = stop_levels.loc[asset]
            asset_target_weight = target_weights.loc[asset]

            if asset_target_weight > 0:  # Long position
                if current_price_for_asset <= asset_stop_level:
                    adjusted_weights.loc[asset] = 0.0
            elif asset_target_weight < 0:  # Short position
                if current_price_for_asset >= asset_stop_level:
                    adjusted_weights.loc[asset] = 0.0
        return adjusted_weights

# Removed commented-out ATRFeature class definition
