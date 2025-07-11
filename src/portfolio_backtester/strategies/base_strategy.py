from __future__ import annotations

import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Removed Feature and BenchmarkSMA imports as features are now internal
# from ..features.base import Feature
# from ..features.benchmark_sma import BenchmarkSMA
from ..portfolio.position_sizer import get_position_sizer
# Removed BaseSignalGenerator as it's being phased out
# from ..signal_generators import BaseSignalGenerator
from ..roro_signals import BaseRoRoSignal
from .stop_loss import BaseStopLoss, NoStopLoss, AtrBasedStopLoss


class BaseStrategy(ABC):
    """Base class for trading strategies."""

    # Removed signal_generator_class as signal generation is now internal
    # signal_generator_class: type[BaseSignalGenerator] | None = None

    #: class attribute specifying which RoRo signal generator to use
    roro_signal_class: type[BaseRoRoSignal] | None = None
    #: class attribute specifying which Stop Loss handler to use by default
    stop_loss_handler_class: type[BaseStopLoss] = NoStopLoss


    def __init__(self, strategy_config: dict):
        self.strategy_config = strategy_config
        self._roro_signal_instance: BaseRoRoSignal | None = None
        self._stop_loss_handler_instance: BaseStopLoss | None = None
        self.entry_prices: pd.Series | None = None


    # ------------------------------------------------------------------ #
    # Hooks to override in subclasses
    # ------------------------------------------------------------------ #

    # Removed get_signal_generator as it's no longer needed
    # def get_signal_generator(self) -> BaseSignalGenerator:
    #     if self.signal_generator_class is None:
    #         raise NotImplementedError("signal_generator_class must be set")
    #     return self.signal_generator_class(self.strategy_config)

    def get_roro_signal(self) -> BaseRoRoSignal | None:
        if self.roro_signal_class is None:
            return None
        if self._roro_signal_instance is None:
            roro_config = self.strategy_config.get("roro_signal_params", self.strategy_config)
            self._roro_signal_instance = self.roro_signal_class(roro_config)
        return self._roro_signal_instance

    def get_stop_loss_handler(self) -> BaseStopLoss:
        if self._stop_loss_handler_instance is None:
            sl_config = self.strategy_config.get("stop_loss_config", {})
            sl_type_name = sl_config.get("type", "NoStopLoss")

            handler_class = NoStopLoss
            if sl_type_name == "AtrBasedStopLoss":
                handler_class = AtrBasedStopLoss

            self._stop_loss_handler_instance = handler_class(self.strategy_config, sl_config)
        return self._stop_loss_handler_instance

    def get_position_sizer(self) -> Callable[[pd.DataFrame], pd.DataFrame]:
        name = self.strategy_config.get("position_sizer", "equal_weight")
        return get_position_sizer(name)

    # Removed get_required_features as features are now internal to strategies
    # @classmethod
    # def get_required_features(cls, strategy_config: dict) -> Set[Feature]:
    #     features: Set[Feature] = set()
    #     # ... (old logic removed) ...
    #     return features

    # ------------------------------------------------------------------ #
    # Universe helper
    # ------------------------------------------------------------------ #
    def get_universe(self, global_config: dict) -> list[tuple[str, float]]:
        default_universe = global_config.get("universe", [])
        return [(ticker, 1.0) for ticker in default_universe]


    # ------------------------------------------------------------------ #
    # Optimiser-introspection hook                                       #
    # ------------------------------------------------------------------ #
    @classmethod
    def tunable_parameters(cls) -> set[str]:
        """Names of hyper-parameters this strategy understands."""
        return set()

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #
    def _calculate_candidate_weights(self, look: pd.Series) -> pd.Series:
        num_holdings = self.strategy_config.get("num_holdings")
        if num_holdings is not None and num_holdings > 0:
            nh = int(num_holdings)
        else:
            nh = max(
                int(
                    np.ceil(
                        self.strategy_config.get("top_decile_fraction", 0.1)
                        * look.count()
                    )
                ),
                1,
            )

        winners = look.nlargest(nh).index
        losers = look.nsmallest(nh).index

        cand = pd.Series(index=look.index, dtype=float).fillna(0.0)
        if len(winners) > 0:
            cand[winners] = 1 / len(winners)
        if not self.strategy_config.get("long_only", True) and len(losers) > 0:
            cand[losers] = -1 / len(losers)
        return cand

    def _apply_leverage_and_smoothing(
        self, cand: pd.Series, w_prev: pd.Series
    ) -> pd.Series:
        leverage = self.strategy_config.get("leverage", 1.0)
        smoothing_lambda = self.strategy_config.get("smoothing_lambda", 0.5)

        w_new = smoothing_lambda * w_prev + (1 - smoothing_lambda) * cand

        if cand.abs().sum() > 1e-9:
            long_lev = w_new[w_new > 0].sum()
            short_lev = -w_new[w_new < 0].sum()

            if long_lev > leverage:
                w_new[w_new > 0] *= leverage / long_lev
            if short_lev > leverage:
                w_new[w_new < 0] *= leverage / short_lev

        return w_new

    # ------------------------------------------------------------------ #
    # Default signal generation pipeline (Abstract method to be implemented by subclasses)
    # ------------------------------------------------------------------ #
    @abstractmethod
    def generate_signals(
        self,
        all_historical_data: pd.DataFrame, # Full historical data for universe assets
        benchmark_historical_data: pd.DataFrame, # Full historical data for benchmark
        current_date: pd.Timestamp, # The current date for signal generation
        start_date: Optional[pd.Timestamp] = None, # Optional start date for WFO window
        end_date: Optional[pd.Timestamp] = None, # Optional end date for WFO window
    ) -> pd.DataFrame: # Returns a DataFrame of weights
        """
        Generates trading signals based on historical data and current date.
        Subclasses must implement this method.

        Args:
            all_historical_data: DataFrame with historical OHLCV data for all assets
                                 in the strategy's universe, up to and including current_date.
            benchmark_historical_data: DataFrame with historical OHLCV data for the benchmark,
                                       up to and including current_date.
            current_date: The specific date for which signals are to be generated.
                          Calculations should not use data beyond this date.
            start_date: If provided, signals should only be generated on or after this date.
            end_date: If provided, signals should only be generated on or before this date.

        Returns:
            A DataFrame indexed by date, with columns for each asset, containing
            the target weights. Should typically contain a single row for current_date
            if generating signals for one date at a time, or multiple rows if the
            strategy generates signals for a range and then filters.
            The weights should adhere to the start_date and end_date if provided.
        """
        pass

    # --- Helper methods that might be used by subclasses ---

    # _calculate_candidate_weights and _apply_leverage_and_smoothing remain as they are useful general helpers.
    # The SMA-based risk filter and RoRo signal logic will be moved into concrete strategies
    # or handled by updated RoRo/StopLoss handlers that take full historical data.

    # _calculate_derisk_flags might still be useful if strategies reimplement SMA logic.
    # It will need access to benchmark_historical_data passed to generate_signals.
    def _calculate_benchmark_sma(self, benchmark_historical_data: pd.DataFrame, window: int, price_column: str = 'Close') -> pd.Series:
        """Calculates SMA for the benchmark."""
        if benchmark_historical_data.empty or price_column not in benchmark_historical_data.columns:
            # Ensure benchmark_historical_data.index is valid even if empty for pd.Series constructor
            index = benchmark_historical_data.index if benchmark_historical_data.index is not None else pd.Index([])
            return pd.Series(dtype=float, index=index)

        # Ensure we only use data up to current_date if current_date is within the index
        # This is more of a safeguard, as input data should already be sliced.
        # However, rolling calculations might inadvertently see future data if not careful with slicing *before* this call.
        # For now, assume benchmark_historical_data is correctly pre-sliced.
        return benchmark_historical_data[price_column].rolling(window=window, min_periods=max(1, window // 2)).mean()


    def _calculate_derisk_flags(self, benchmark_prices_at_current_date: pd.Series, benchmark_sma_at_current_date: pd.Series, derisk_periods: int, previous_derisk_flag: bool, consecutive_periods_under_sma: int) -> tuple[bool, int]:
        """
        Calculates a derisk flag for the current period based on benchmark price vs. SMA.
        This is a stateful calculation for a single point in time.

        Args:
            benchmark_prices_at_current_date: Series of benchmark prices for the current_date (should be one value).
            benchmark_sma_at_current_date: Series of benchmark SMA for the current_date (should be one value).
            derisk_periods: Number of consecutive periods benchmark must be under SMA to trigger derisking.
            previous_derisk_flag: Boolean indicating if derisking was active in the previous period.
            consecutive_periods_under_sma: Count of consecutive periods benchmark was under SMA leading up to current.

        Returns:
            A tuple: (current_derisk_flag: bool, updated_consecutive_periods_under_sma: int)
        """
        current_derisk_flag = previous_derisk_flag # Start with previous state

        if benchmark_prices_at_current_date.empty or benchmark_sma_at_current_date.empty or \
           benchmark_prices_at_current_date.iloc[0] is pd.NA or benchmark_sma_at_current_date.iloc[0] is pd.NA:
            # Not enough data, maintain previous state or default to not derisked if no previous state
            return previous_derisk_flag, consecutive_periods_under_sma

        price = benchmark_prices_at_current_date.iloc[0].item()
        sma = benchmark_sma_at_current_date.iloc[0].item()

        if price < sma:
            consecutive_periods_under_sma += 1
        else: # Price is >= SMA
            consecutive_periods_under_sma = 0
            current_derisk_flag = False # If above SMA, always turn off derisk flag

        if consecutive_periods_under_sma > derisk_periods:
            current_derisk_flag = True

        # If it was derisked and now price is above SMA, it's handled by consecutive_periods_under_sma = 0 and current_derisk_flag = False

        return current_derisk_flag, consecutive_periods_under_sma

    def get_minimum_required_periods(self) -> int:
        """
        Calculate the minimum number of periods (months) of historical data required
        for this strategy to function properly. This should be overridden by subclasses
        to provide strategy-specific requirements.
        
        Returns:
            int: Minimum number of months of historical data required
        """
        # Base implementation returns a conservative default
        return 12

    def validate_data_sufficiency(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        current_date: pd.Timestamp
    ) -> tuple[bool, str]:
        """
        Validates that there is sufficient historical data available for the strategy
        to perform reliable calculations as of the current_date.
        
        Args:
            all_historical_data: DataFrame with historical data for universe assets
            benchmark_historical_data: DataFrame with historical data for benchmark
            current_date: The date for which we're checking data sufficiency
            
        Returns:
            tuple[bool, str]: (is_sufficient, reason_if_not)
        """
        min_periods_required = self.get_minimum_required_periods()
        
        # Check universe data
        if all_historical_data.empty:
            return False, "No historical data available for universe assets"
            
        # Check if current_date is beyond available data
        latest_available_date = all_historical_data.index.max()
        if current_date > latest_available_date:
            return False, f"Current date {current_date} is beyond available data (latest: {latest_available_date})"
            
        # Filter data up to current_date
        available_data = all_historical_data[all_historical_data.index <= current_date]
        if available_data.empty:
            return False, f"No historical data available up to {current_date}"
            
        # Calculate available periods (assuming monthly frequency)
        earliest_date = available_data.index.min()
        available_months = (current_date.year - earliest_date.year) * 12 + (current_date.month - earliest_date.month)
        
        if available_months < min_periods_required:
            return False, f"Insufficient historical data: {available_months} months available, {min_periods_required} months required"
            
        # Check benchmark data if strategy uses SMA filtering
        sma_filter_window = self.strategy_config.get("strategy_params", {}).get("sma_filter_window")
        if sma_filter_window and sma_filter_window > 0:
            if benchmark_historical_data.empty:
                return False, "No benchmark data available but SMA filtering is enabled"
                
            # Check if current_date is beyond benchmark data
            benchmark_latest_date = benchmark_historical_data.index.max()
            if current_date > benchmark_latest_date:
                return False, f"Current date {current_date} is beyond available benchmark data (latest: {benchmark_latest_date})"
                
            benchmark_available = benchmark_historical_data[benchmark_historical_data.index <= current_date]
            if benchmark_available.empty:
                return False, f"No benchmark data available up to {current_date}"
                
            benchmark_earliest = benchmark_available.index.min()
            benchmark_months = (current_date.year - benchmark_earliest.year) * 12 + (current_date.month - benchmark_earliest.month)
            
            if benchmark_months < sma_filter_window:
                return False, f"Insufficient benchmark data for SMA filter: {benchmark_months} months available, {sma_filter_window} months required"
        
        return True, ""

    def filter_universe_by_data_availability(
        self,
        all_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
        min_periods_override: int = None
    ) -> list:
        """
        Filter the universe to only include assets that have sufficient historical data
        as of the current date. This handles cases where stocks were not yet listed
        or have been delisted.
        
        Args:
            all_historical_data: DataFrame with historical data for universe assets
            current_date: The date for which we're checking data availability
            min_periods_override: Override minimum periods requirement (default: use strategy requirement)
            
        Returns:
            list: List of assets that have sufficient data
        """
        min_periods_required = min_periods_override or self.get_minimum_required_periods()
        
        if all_historical_data.empty:
            return []
        
        # Filter data up to current_date
        available_data = all_historical_data[all_historical_data.index <= current_date]
        if available_data.empty:
            return []
        
        valid_assets = []
        
        # Get asset list based on column structure
        if isinstance(all_historical_data.columns, pd.MultiIndex):
            # MultiIndex columns - get unique tickers
            asset_list = all_historical_data.columns.get_level_values('Ticker').unique()
        else:
            # Simple columns - column names are tickers
            asset_list = all_historical_data.columns
        
        for asset in asset_list:
            try:
                # Extract asset data
                if isinstance(all_historical_data.columns, pd.MultiIndex):
                    # For MultiIndex, get all fields for this ticker
                    asset_data = available_data.xs(asset, level='Ticker', axis=1, drop_level=False)
                    # Check if we have Close prices (most important)
                    if (asset, 'Close') in asset_data.columns:
                        asset_prices = asset_data[(asset, 'Close')].dropna()
                    else:
                        continue  # Skip if no Close prices
                else:
                    # Simple column structure
                    if asset not in available_data.columns:
                        continue
                    asset_prices = available_data[asset].dropna()
                
                # Check if asset has sufficient data
                if len(asset_prices) == 0:
                    continue  # No data for this asset
                
                # Check data availability period
                asset_earliest = asset_prices.index.min()
                asset_latest = asset_prices.index.max()
                
                # Skip if asset data doesn't reach current date (delisted or data gap)
                if asset_latest < current_date - pd.DateOffset(days=30):  # Allow 30-day lag
                    continue
                
                # Calculate available months for this asset
                available_months = (current_date.year - asset_earliest.year) * 12 + (current_date.month - asset_earliest.month)
                
                # Check if asset has minimum required data
                if available_months >= min_periods_required:
                    valid_assets.append(asset)
                    
            except Exception as e:
                # Skip assets that cause errors in data processing
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Skipping asset {asset} due to data processing error: {e}")
                continue
        
        if len(valid_assets) < len(asset_list):
            excluded_count = len(asset_list) - len(valid_assets)
            exclusion_rate = excluded_count / len(asset_list)
            
            # Only log if there are significant issues
            if len(valid_assets) == 0:
                if logger.isEnabledFor(logging.ERROR):
                    logger.error(f"No assets have sufficient data for {current_date.strftime('%Y-%m-%d')} - all {len(asset_list)} assets excluded")
            elif exclusion_rate > 0.5:  # More than 50% excluded
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(f"High asset exclusion rate: {len(valid_assets)}/{len(asset_list)} assets have sufficient data for {current_date.strftime('%Y-%m-%d')} ({exclusion_rate:.1%} excluded)")
                # Also log at debug level when filtered universe is less than 50% of original
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Filtered universe: {len(valid_assets)}/{len(asset_list)} assets have sufficient data for {current_date.strftime('%Y-%m-%d')} (excluded {excluded_count} assets)")
            # Remove the else clause - no debug logging for normal filtering (exclusion_rate <= 50%)
        
        return valid_assets
