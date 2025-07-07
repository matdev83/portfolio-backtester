from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Set, TYPE_CHECKING
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from ..features.base import Feature
    from ..features.atr import ATRFeature # Forward reference for ATRFeature
    from ..strategies.base_strategy import BaseStrategy # For type hinting if needed

class BaseStopLoss(ABC):
    """
    Abstract base class for stop-loss handlers.
    """
    def __init__(self, strategy_config: dict, stop_loss_specific_config: dict):
        """
        Initializes the stop-loss handler.

        Args:
            strategy_config: The overall configuration for the strategy.
            stop_loss_specific_config: Configuration specific to this stop-loss handler,
                                       e.g., {"type": "AtrBasedStopLoss", "atr_length": 14, ...}.
        """
        self.strategy_config = strategy_config
        self.stop_loss_specific_config = stop_loss_specific_config

    @abstractmethod
    def get_required_features(self) -> Set[Feature]:
        """
        Returns a set of features required by this stop-loss handler.
        """
        pass

    @abstractmethod
    def calculate_stop_levels(
        self,
        current_date: pd.Timestamp,
        prices_history: pd.DataFrame, # Historical prices up to current_date for context
        features: dict,
        current_weights: pd.Series,
        entry_prices: pd.Series
    ) -> pd.Series:
        """
        Calculates the stop-loss price levels for each asset.

        Args:
            current_date: The current date for which to calculate stop levels.
            prices_history: DataFrame of historical prices (OHLC) for all assets.
                            The stop_loss module might use this for its calculations (e.g. ATR).
            features: Dictionary of pre-calculated feature data.
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
        current_date: pd.Timestamp, # The date for which signals are being generated / stops checked
        prices_for_current_date: pd.DataFrame, # Daily OHLC data for the current_date
        target_weights: pd.Series, # The proposed target weights by the strategy before stop loss
        entry_prices: pd.Series,
        stop_levels: pd.Series
    ) -> pd.Series:
        """
        Adjusts target weights if stop-loss conditions are met.

        Args:
            current_date: The specific date for which the stop loss is being checked.
            prices_for_current_date: DataFrame containing at least 'Low' and 'High' columns
                                     for the `current_date` for all relevant assets.
                                     This is used to check if stop levels were breached.
                                     For strategies generating signals on e.g. month-end,
                                     this might be the month-end prices.
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
    def get_required_features(self) -> Set[Feature]:
        return set()

    def calculate_stop_levels(
        self,
        current_date: pd.Timestamp,
        prices_history: pd.DataFrame,
        features: dict,
        current_weights: pd.Series,
        entry_prices: pd.Series
    ) -> pd.Series:
        return pd.Series(np.nan, index=current_weights.index)

    def apply_stop_loss(
        self,
        current_date: pd.Timestamp,
        prices_for_current_date: pd.DataFrame,
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

    def get_required_features(self) -> Set[Feature]:
        from ..features.atr import ATRFeature # Import here to avoid circular dependency at module load time
        return {ATRFeature(atr_period=self.atr_length)}

    def calculate_stop_levels(
        self,
        current_date: pd.Timestamp,
        prices_history: pd.DataFrame, # Unused in this version, ATR comes from features
        features: dict,
        current_weights: pd.Series,
        entry_prices: pd.Series
    ) -> pd.Series:
        stop_levels = pd.Series(np.nan, index=current_weights.index)
        atr_feature_name = f"atr_{self.atr_length}" # Matches default ATRFeature naming

        if atr_feature_name not in features:
            # Log or raise error if ATR feature is missing
            # For now, return NaNs, effectively disabling stops
            print(f"Warning: ATR feature '{atr_feature_name}' not found for date {current_date}. ATR Stop Loss disabled for this period.")
            return stop_levels

        atr_values_all_assets = features[atr_feature_name]

        # Ensure atr_values_for_date is a Series
        if isinstance(atr_values_all_assets, pd.DataFrame):
            if current_date in atr_values_all_assets.index:
                atr_values_for_date = atr_values_all_assets.loc[current_date]
            else:
                # If current_date is not in features (e.g., features are on month ends, current_date is intraday)
                # try to get the latest available ATR. This might need adjustment based on how features are indexed.
                # For now, we assume current_date will be in features index.
                print(f"Warning: ATR values for date {current_date} not found. ATR Stop Loss might be impacted.")
                return stop_levels # Or use ffill, but that implies stale data
        elif isinstance(atr_values_all_assets, pd.Series): # Should ideally be a DataFrame (features over time)
             # This case implies features[atr_feature_name] is a single series for one date, which is unusual
             # Or it's a multi-indexed series. For now, let's assume it's NOT what we expect for time-series features.
             # This part of the logic might need refinement based on actual `features` structure.
             # For now, if it's a series, we assume it's for the current_date, which is unlikely.
             # A robust solution would be to ensure features are DataFrames [date, asset_ticker].
             # Given the current plan, features are computed on monthly data, so atr_values_all_assets
             # will be a DataFrame indexed by month-end dates, with columns for each asset.
            print(f"Warning: ATR feature '{atr_feature_name}' is a Series, expected DataFrame. Processing may be incorrect.")
            # This path is less likely if features are correctly structured as DataFrame[date, ticker]
            # If it's a series, it must be indexed by asset for the current_date.
            atr_values_for_date = atr_values_all_assets
        else:
            print(f"Warning: ATR feature '{atr_feature_name}' has unexpected type: {type(atr_values_all_assets)}. ATR Stop Loss disabled.")
            return stop_levels


        for asset in current_weights.index:
            weight = current_weights.loc[asset]
            entry_price = entry_prices.loc[asset]

            if pd.isna(entry_price) or weight == 0:
                continue

            # Get ATR for the specific asset on the current_date
            # This expects atr_values_for_date to be a Series indexed by asset tickers
            if asset in atr_values_for_date.index:
                atr_value = atr_values_for_date.loc[asset]
            else:
                # print(f"Warning: ATR value for asset {asset} on date {current_date} not found. Skipping stop for this asset.")
                # This can happen if the ATR feature doesn't cover all assets in current_weights,
                # e.g. if an asset is new to the universe and doesn't have enough history for ATR.
                atr_value = np.nan # Default to NaN if specific asset ATR is missing

            if pd.isna(atr_value):
                # print(f"Warning: ATR value for asset {asset} on date {current_date} is NaN. Skipping stop for this asset.")
                continue

            if weight > 0:  # Long position
                stop_levels.loc[asset] = entry_price - (atr_value * self.atr_multiple)
            elif weight < 0:  # Short position
                stop_levels.loc[asset] = entry_price + (atr_value * self.atr_multiple)

        return stop_levels

    def apply_stop_loss(
        self,
        current_date: pd.Timestamp,
        prices_for_current_date: pd.DataFrame, # Expects columns like 'Low', 'High', 'Close'
        target_weights: pd.Series,
        entry_prices: pd.Series, # Unused here, but part of the interface
        stop_levels: pd.Series
    ) -> pd.Series:
        adjusted_weights = target_weights.copy()

        # Ensure prices_for_current_date is a Series for the specific date if it's a DataFrame row
        # The plan is that BaseStrategy will pass prices.loc[[date]], which is a DataFrame with one row.
        # We need to ensure we are accessing the values correctly.
        if isinstance(prices_for_current_date, pd.DataFrame):
            if not prices_for_current_date.empty and current_date in prices_for_current_date.index:
                # If it's a DataFrame with the current_date as index (e.g. from prices.loc[[date]])
                # then for each asset, we access its low/high.
                # This assumes prices_for_current_date columns are asset tickers, and it contains Low/High for that date.
                # This interpretation is WRONG. prices_for_current_date should be like:
                #            AAPL   MSFT   GOOG
                # Open      150.0  250.0  1000.0
                # High      152.0  252.0  1002.0
                # Low       149.0  249.0   998.0
                # Close     151.0  251.0  1001.0
                # This structure is not what BaseStrategy.generate_signals currently receives for `prices`.
                # `prices` in generate_signals is DataFrame [date, asset_ticker] of CLOSE prices.
                #
                # REVISED PLAN for apply_stop_loss:
                # The `BaseStrategy` will need to be modified to pass the *correct* price data
                # for the stop check. For the initial implementation as per plan step 3's revision:
                # "The stop loss will be checked based on the *previous period's close* against the calculated stop level."
                # So, `prices_for_current_date` will effectively be a Series of close prices for `current_date`.
                # Let's assume `prices_for_current_date` is a pd.Series of CLOSE prices for current_date, indexed by asset.
                pass # Logic below handles this
            else:
                # print(f"Warning: Price data for current_date {current_date} not available in expected format for stop-loss check.")
                return adjusted_weights # Cannot check stops

        for asset in target_weights.index:
            if pd.isna(stop_levels.loc[asset]) or target_weights.loc[asset] == 0:
                continue

            # As per revised plan, using close prices from `prices_for_current_date` (assumed Series of closes)
            # This means `prices_for_current_date` should be `prices.loc[current_date]` from BaseStrategy.
            current_price_for_asset = prices_for_current_date.get(asset) # Get close price for the asset

            if current_price_for_asset is None or pd.isna(current_price_for_asset):
                # print(f"Warning: Close price for asset {asset} on {current_date} not found. Cannot apply stop loss.")
                continue

            if target_weights.loc[asset] > 0:  # Long position
                # If current period's price (e.g. month-end close) is below stop level
                if current_price_for_asset <= stop_levels.loc[asset]:
                    adjusted_weights.loc[asset] = 0.0
                    # print(f"Stop-loss triggered for LONG {asset} on {current_date}. Price: {current_price_for_asset}, Stop: {stop_levels.loc[asset]}")
            elif target_weights.loc[asset] < 0:  # Short position
                # If current period's price (e.g. month-end close) is above stop level
                if current_price_for_asset >= stop_levels.loc[asset]:
                    adjusted_weights.loc[asset] = 0.0
                    # print(f"Stop-loss triggered for SHORT {asset} on {current_date}. Price: {current_price_for_asset}, Stop: {stop_levels.loc[asset]}")

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
#         # Placeholder:
#         # import pandas_ta as ta
#         # atr_df = pd.DataFrame(index=data.index)
#         # for asset_col in data.columns.get_level_values(0).unique(): # Assuming multi-index [Asset, OHLC]
#         #     asset_data = data[asset_col] # This would be a DataFrame with HLC columns
#         #     atr_df[asset_col] = ta.atr(high=asset_data['High'], low=asset_data['Low'], close=asset_data['Close'], length=self.atr_period)
#         # return atr_df
#         raise NotImplementedError("ATRFeature compute method needs to be implemented correctly.")

#     def __hash__(self):
#         return hash((self.name, self.atr_period))

#     def __eq__(self, other):
#         if not isinstance(other, ATRFeature):
#             return False
#         return self.name == other.name and self.atr_period == other.atr_period
