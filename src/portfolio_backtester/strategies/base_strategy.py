from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional # Added Optional

import numpy as np
import pandas as pd

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

    # The old _calculate_derisk_flags was designed to work on a whole series.
    # The new one above is for point-in-time calculation within a loop.
    # If a series-based calculation is still needed by a strategy, it can implement it locally.
    # For now, I'll comment out the old one to avoid confusion.

    # def _calculate_derisk_flags(self, sma_risk_on_series: pd.Series, derisk_days: int) -> pd.Series:
    #     """
    #     Calculates flags indicating when to derisk based on consecutive days under SMA.

    #     Parameters:
    #     - sma_risk_on_series (pd.Series): Boolean series indicating if risk is on (True, price >= SMA)
    #                                       or off (False, price < SMA). Index must match prices.index.
    #     - derisk_days (int): Number of consecutive days asset must be under SMA to trigger derisking.

    #     Returns:
    #     - pd.Series: Boolean series with True where derisking should occur.
    #     """
    #     if not isinstance(sma_risk_on_series, pd.Series) or not isinstance(derisk_days, int) or derisk_days <= 0:
    #         # This case should ideally be handled by the caller (use_sma_derisk check)
    #         # but as a safeguard, return an all-false series.
    #         return pd.Series(False, index=sma_risk_on_series.index)

    #     derisk_flags = pd.Series(False, index=sma_risk_on_series.index)
    #     consecutive_days_risk_off = 0

    #     for date_val in sma_risk_on_series.index: # Renamed 'date' to 'date_val' to avoid conflict
    #         if sma_risk_on_series.loc[date_val]:  # Price is >= SMA (Risk is ON based on SMA)
    #             consecutive_days_risk_off = 0
    #         else:  # Price is < SMA (Risk is OFF based on SMA)
    #             consecutive_days_risk_off += 1

    #         if consecutive_days_risk_off > derisk_days: # Note: strictly greater
    #             derisk_flags.loc[date_val] = True
    #         # else: # If it's not > derisk_days, the flag remains False (or becomes False if it was True and now sma_risk_on_series is True)
    #         #    if derisk_flags.loc[date_val]: # This logic is not needed as we reset consecutive_days_risk_off
    #         #        pass
    #     return derisk_flags
