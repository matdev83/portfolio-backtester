from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional, TYPE_CHECKING, Dict, Any, List, Tuple, Union

import numpy as np
import pandas as pd

from ..universe_resolver import resolve_universe_config

# Removed BaseSignalGenerator as it's being phased out
# from ..signal_generators import BaseSignalGenerator
from ..roro_signals import BaseRoRoSignal
from .stop_loss_strategy import AtrBasedStopLoss, BaseStopLoss, NoStopLoss
from ..portfolio.position_sizer import get_position_sizer
from ..api_stability import api_stable

if TYPE_CHECKING:
    from ..timing.timing_controller import TimingController

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Base class for trading strategies.
    
    TESTING NOTE: When testing strategy classes, be aware that strategy instances
    are created with a configuration dictionary that contains 'strategy_params'.
    The strategy parameters are stored in self.strategy_config['strategy_params'],
    not directly in self.strategy_config. This is important for test assertions
    that check parameter values.
    """

    # Removed signal_generator_class as signal generation is now internal
    # signal_generator_class: type[BaseSignalGenerator] | None = None

    #: class attribute specifying which RoRo signal generator to use
    roro_signal_class: type[BaseRoRoSignal] | None = None
    #: class attribute specifying which Stop Loss handler to use by default
    stop_loss_handler_class: type[BaseStopLoss] = NoStopLoss


    def __init__(self, strategy_config: Dict[str, Any]) -> None:
        self.strategy_config = strategy_config
        self._roro_signal_instance: BaseRoRoSignal | None = None
        self._stop_loss_handler_instance: BaseStopLoss | None = None
        self.entry_prices: pd.Series | None = None
        self._timing_controller: Optional[TimingController] = None
        self._initialize_timing_controller()


    # ------------------------------------------------------------------ #
    # Timing Controller Integration
    # ------------------------------------------------------------------ #
    
    def _initialize_timing_controller(self) -> None:
        """Initialize the appropriate timing controller based on configuration."""
        # Import here to avoid circular imports
        from ..timing.time_based_timing import TimeBasedTiming
        from ..timing.signal_based_timing import SignalBasedTiming
        from ..timing.backward_compatibility import ensure_backward_compatibility_with_strategy
        
        try:
            # Ensure backward compatibility and migrate legacy configuration
            # Pass strategy instance for method override detection
            migrated_config = ensure_backward_compatibility_with_strategy(self.strategy_config, self)
            
            # Update strategy config with migrated version
            self.strategy_config = migrated_config
            
            # Get timing configuration (guaranteed to exist after migration)
            timing_config = self.strategy_config['timing_config']
            timing_mode = timing_config.get('mode', 'time_based')
            
            # Initialize appropriate timing controller
            if timing_mode == 'time_based':
                self._timing_controller = TimeBasedTiming(timing_config)
            elif timing_mode == 'signal_based':
                self._timing_controller = SignalBasedTiming(timing_config)
            else:
                # Support custom timing controllers
                custom_class = timing_config.get('custom_class')
                if custom_class:
                    self._timing_controller = custom_class(timing_config)
                else:
                    # Default to time-based for backward compatibility
                    logger.warning(f"Unknown timing mode '{timing_mode}', defaulting to time_based")
                    default_config = {'mode': 'time_based', 'rebalance_frequency': 'M'}
                    self._timing_controller = TimeBasedTiming(default_config)
                    
        except Exception as e:
            # Fallback to time-based timing if initialization fails
            logger.error(f"Failed to initialize timing controller: {e}")
            logger.info("Falling back to time-based timing with monthly frequency")
            fallback_config = {'mode': 'time_based', 'rebalance_frequency': 'M'}
            self._timing_controller = TimeBasedTiming(fallback_config)
            
            # Update strategy config with fallback timing config
            self.strategy_config['timing_config'] = fallback_config
    
    def get_timing_controller(self) -> Optional['TimingController']:
        """Get the timing controller for this strategy."""
        if self._timing_controller is None:
            self._initialize_timing_controller()
        return self._timing_controller
    
    def supports_daily_signals(self) -> bool:
        """
        Determine if strategy supports daily signals based on timing controller.
        This method maintains backward compatibility while using the new timing system.
        """
        # Import here to avoid circular imports
        from ..timing.signal_based_timing import SignalBasedTiming
        
        timing_controller = self.get_timing_controller()
        return isinstance(timing_controller, SignalBasedTiming)

    # ------------------------------------------------------------------ #
    # Hooks to override in subclasses
    # ------------------------------------------------------------------ #

    # Removed get_signal_generator as it's no longer needed
    # def get_signal_generator(self) -> BaseSignalGenerator:
    #     if self.signal_generator_class is None:
    #         raise NotImplementedError("signal_generator_class must be set")
    #     return self.signal_generator_class(self.strategy_config)

    @api_stable(version="1.0", strict_params=True, strict_return=False)
    def get_roro_signal(self) -> Optional[BaseRoRoSignal]:
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


    # Removed get_required_features as features are now internal to strategies
    # @classmethod
    # def get_required_features(cls, strategy_config: dict) -> Set[Feature]:
    #     features: Set[Feature] = set()
    #     # ... (old logic removed) ...
    #     return features

    # ------------------------------------------------------------------ #
    # Universe helper
    # ------------------------------------------------------------------ #
    def get_universe(self, global_config: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Get the universe of assets for this strategy.
        
        Resolution priority:
        1. Strategy-specific universe_config
        2. Legacy strategy get_universe override (if overridden in subclass)
        3. Global config universe
        
        Args:
            global_config: Global configuration dictionary
            
        Returns:
            List of (ticker, weight) tuples. Weight is typically 1.0 for equal consideration.
            
        Raises:
            ValueError: If the universe is empty (contains no symbols).
        """
        # Check if strategy has universe_config
        universe_config = self.strategy_config.get("universe_config")
        
        if universe_config:
            try:
                tickers = resolve_universe_config(universe_config)
                universe = [(ticker, 1.0) for ticker in tickers]
                if not universe:
                    raise ValueError("Strategy universe_config resolved to an empty universe")
                return universe
            except Exception as e:
                logger.error(f"Failed to resolve universe_config: {e}")
                logger.info("Falling back to global universe")
        
        # Check if this method is overridden in a subclass (legacy behavior)
        if type(self).get_universe is not BaseStrategy.get_universe:
            # This method is overridden, so we should not interfere with custom logic
            # However, we need to avoid infinite recursion, so we call the parent implementation
            # This is a bit tricky - we'll use the default behavior as fallback
            pass
        
        # Fallback to global config universe
        default_universe = global_config.get("universe", [])
        universe = [(ticker, 1.0) for ticker in default_universe]
        if not universe:
            raise ValueError("Global config universe is empty")
        return universe
    
    
    
    def get_universe_method_with_date(self, global_config: Dict[str, Any], current_date: pd.Timestamp) -> List[Tuple[str, float]]:
        """
        Get the universe of assets for this strategy with date context.
        
        This method is similar to get_universe but provides date context for
        dynamic universe methods that need to know the current date.
        
        Args:
            global_config: Global configuration dictionary
            current_date: Current date for universe resolution
            
        Returns:
            List of (ticker, weight) tuples
            
        Raises:
            ValueError: If the universe is empty (contains no symbols).
        """
        # Check if strategy has universe_config
        universe_config = self.strategy_config.get("universe_config")
        
        if universe_config:
            try:
                tickers = resolve_universe_config(universe_config, current_date)
                universe = [(ticker, 1.0) for ticker in tickers]
                if not universe:
                    raise ValueError("Strategy universe_config resolved to an empty universe")
                return universe
            except Exception as e:
                logger.error(f"Failed to resolve universe_config with date {current_date}: {e}")
                logger.info("Falling back to global universe")
        
        # Fallback to global config universe
        default_universe = global_config.get("universe", [])
        universe = [(ticker, 1.0) for ticker in default_universe]
        if not universe:
            raise ValueError("Global config universe is empty")
        return universe
    
    

    def get_non_universe_data_requirements(self) -> List[str]:
        """
        Returns a list of tickers that are not part of the trading universe
        but are required for the strategy's calculations.
        """
        return []

    def get_synthetic_data_requirements(self) -> bool:
        """
        Returns a boolean indicating whether the strategy requires synthetic data generation.
        """
        return True


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
    

    # ------------------------------------------------------------------ #
    # Default signal generation pipeline (Abstract method to be implemented by subclasses)
    # ------------------------------------------------------------------ #
    from numba import njit

    @api_stable(version="1.0", strict_params=True, strict_return=True)
    def generate_signals(
        self,
        all_historical_data: pd.DataFrame, # Full historical data for universe assets
        benchmark_historical_data: pd.DataFrame, # Full historical data for benchmark
        non_universe_historical_data: pd.DataFrame, # Full historical data for non-universe assets
        current_date: pd.Timestamp, # The current date for signal generation
        start_date: Optional[pd.Timestamp] = None, # Optional start date for WFO window
        end_date: Optional[pd.Timestamp] = None, # Optional end date for WFO window
    ) -> pd.DataFrame: # Returns a DataFrame of weights
        """
        IMPORTANT: Method signature has evolved over time!
        
        LEGACY INTERFACE WARNING: Some old tests may call this method with a different signature:
        - generate_signals(prices, features, benchmark) - OLD 3-argument format
        - generate_signals(all_historical_data, benchmark_historical_data, current_date) - CURRENT format
        
        If you encounter TypeError about numpy array vs Timestamp comparisons, it's likely
        because legacy code is passing arguments in the wrong order, causing current_date
        to receive a pandas Series instead of a Timestamp.
        
        The validate_data_sufficiency() method includes defensive type checking to handle
        these cases gracefully, but it's better to update calling code to use the correct signature.
        """
        """
        Generates trading signals based on historical data and current date.
        Subclasses must implement this method.

        Args:
            all_historical_data: DataFrame with historical OHLCV data for all assets
                                 in the strategy's universe, up to and including current_date.
            benchmark_historical_data: DataFrame with historical OHLCV data for the benchmark,
                                       up to and including current_date.
            non_universe_historical_data: DataFrame with historical OHLCV data for non-universe assets.
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
            
        TESTING PITFALLS FOR FUTURE DEVELOPERS:
        ======================================
        1. Always use the current 6-parameter signature when writing new tests
        2. If you see tests calling generate_signals(prices, features, benchmark), 
           they need to be updated to the current interface
        3. Mock data should use proper MultiIndex OHLCV format, not simple price DataFrames
        4. Always pass current_date as pd.Timestamp, never as Series or numpy array
        5. The validate_data_sufficiency() method will catch type mismatches, but 
           it's better to fix the test interface than rely on defensive coding
        """
        return pd.DataFrame()

    @staticmethod
    @njit
    def run_logic(signals, w_prev, num_holdings, top_decile_fraction, long_only, leverage, smoothing_lambda):
        if num_holdings is not None and num_holdings > 0:
            nh = int(num_holdings)
        else:
            nh = max(
                int(
                    np.ceil(
                        top_decile_fraction
                        * signals.shape[0]
                    )
                ),
                1,
            )

        winners = np.argsort(signals)[-nh:]
        losers = np.argsort(signals)[:nh]

        cand = np.zeros_like(signals)
        if winners.shape[0] > 0:
            cand[winners] = 1 / winners.shape[0]
        if not long_only and losers.shape[0] > 0:
            cand[losers] = -1 / losers.shape[0]

        w_new = smoothing_lambda * w_prev + (1 - smoothing_lambda) * cand

        if np.abs(cand).sum() > 1e-9:
            long_lev = np.sum(w_new[w_new > 0])
            short_lev = -np.sum(w_new[w_new < 0])

            if long_lev > leverage:
                w_new[w_new > 0] *= leverage / long_lev
            if short_lev > leverage:
                w_new[w_new < 0] *= leverage / short_lev

        return w_new

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

    @api_stable(version="1.0", strict_params=True, strict_return=False)
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
            
        # CRITICAL: Defensive type checking for current_date parameter
        # 
        # PROBLEM ENCOUNTERED: Tests and legacy code may pass current_date as various types:
        # - pandas Series (from old test interfaces)
        # - numpy arrays (from parameter mismatches)
        # - strings, datetime objects, etc.
        # 
        # This causes "TypeError: '>' not supported between instances of 'numpy.ndarray' and 'Timestamp'"
        # when comparing current_date > latest_available_date below.
        #
        # SOLUTION: Always ensure current_date is a proper pd.Timestamp before any comparisons
        if not isinstance(current_date, pd.Timestamp):
            # Handle case where current_date might be a numpy array or other iterable type
            # This commonly happens when tests call generate_signals() with wrong signature
            if hasattr(current_date, '__iter__') and not isinstance(current_date, str):
                # Extract first element from array-like objects
                try:
                    if hasattr(current_date, 'iloc'):
                        # For pandas Series/DataFrame - use .iloc to avoid deprecation warnings
                        # about positional indexing with []
                        current_date = pd.Timestamp(current_date.iloc[0])
                    else:
                        # For numpy arrays, lists, tuples, etc.
                        current_date = pd.Timestamp(current_date[0])
                except (IndexError, TypeError):
                    return False, f"Invalid current_date format: {type(current_date)}"
            else:
                # Handle scalar values that aren't Timestamps (strings, datetime, etc.)
                try:
                    current_date = pd.Timestamp(current_date)
                except (ValueError, TypeError):
                    return False, f"Cannot convert current_date to Timestamp: {current_date}"
            
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
        min_periods_override: Optional[int] = None
    ) -> List[str]:
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
