import logging
import math
import numpy as np
from typing import Any, Dict, Optional, Tuple

from ta.trend import EMAIndicator
import pandas as pd

from ..base.signal_strategy import SignalStrategy
from ...datetime_utils import get_bday_offset
from ...interfaces.signal_price_extractor_interface import (
    SignalPriceExtractorFactory,
    ISignalPriceExtractor,
)
from ...interfaces.data_type_converter_interface import (
    DataTypeConverterFactory,
)
from ...interfaces.column_handler_interface import (
    ColumnHandlerFactory,
    IColumnHandler,
)

# Import strategy base interface for composition instead of inheritance
from ...interfaces.strategy_base_interface import IStrategyBase, StrategyBaseFactory

logger = logging.getLogger(__name__)


class SeasonalSignalStrategy(SignalStrategy):
    # Cache for business day calendars and month ends (class-level, shared)
    _bday_range_cache: Dict[Tuple[int, int], pd.DatetimeIndex] = {}
    _month_end_cache: Dict[Tuple[int, int], pd.DatetimeIndex] = {}
    """
    Intramonth seasonal trading strategy.

    - Long-only or short-only (configurable).
    - Buy (or sell for short-only) on the n-th business day of the month.
      - n can be positive (from the start of the month) or negative (from the end of the month).
    - Sell (or buy to cover) m business days later.
    """

    def __init__(self, strategy_config: Dict[str, Any]):
        # Use composition instead of inheritance - create strategy base via factory
        self._strategy_base: IStrategyBase = StrategyBaseFactory.create_strategy_base(
            strategy_config
        )

        # Still call super() for SignalStrategy compatibility, but minimize dependency
        super().__init__(strategy_config)
        self._last_weights: Optional[pd.Series] = None
        # Use Numpy arrays for position tracking
        self._last_weights_np: Optional[np.ndarray] = None  # Numpy array for last weights
        self._exit_dates_np: Optional[np.ndarray] = (
            None  # Numpy array of exit dates (as int64 timestamps)
        )

        defaults = {
            "direction": "long",  # legacy field, retained for backward compatibility
            "entry_day": 5,
            "hold_days": 10,
            "price_column_asset": "Close",
            "trade_longs": True,
            "trade_shorts": True,
        }
        # Add trade_month_i defaults
        for i in range(1, 13):
            defaults[f"trade_month_{i}"] = True

        params_dict_to_update = self.strategy_config
        if "strategy_params" in self.strategy_config:
            params_dict_to_update = self.strategy_config["strategy_params"]
        else:
            params_dict_to_update = self.strategy_config

        for k, v in defaults.items():
            params_dict_to_update.setdefault(k, v)

        # EMA filter settings
        self.use_ema_filter = params_dict_to_update.get("use_ema_filter", False)
        self.fast_ema_len = max(3, params_dict_to_update.get("fast_ema_len", 21))
        self.slow_ema_multiplier = max(1.1, params_dict_to_update.get("slow_ema_multiplier", 2.0))
        self.slow_ema_len = int(math.ceil(self.fast_ema_len * self.slow_ema_multiplier))

        # Long/short trade toggles
        self.trade_longs = params_dict_to_update.get("trade_longs", True)
        self.trade_shorts = params_dict_to_update.get("trade_shorts", True)

        # PERFORMANCE: cache reusable objects -------------------------------
        # 1. Business-day offset is the same for the whole simulation, build it once.
        self._bday_offset = get_bday_offset()
        # 2. Cache entry date per (year, month, entry_day) to avoid recomputing
        #    bdate_range inside every generate_signals call.
        self._entry_date_cache: Dict[Tuple[int, int, int], pd.Timestamp] = {}

        # SOLID: Polymorphic components to eliminate isinstance violations
        self._data_type_processor = DataTypeConverterFactory.create_signal_processor()
        self._price_extractor_cache: Dict[int, ISignalPriceExtractor] = {}
        self._column_handler_cache: Dict[int, IColumnHandler] = {}

    def _get_price_extractor(self, data: pd.DataFrame) -> ISignalPriceExtractor:
        """Get appropriate price extractor for the DataFrame structure."""
        data_key = id(data.columns)
        if data_key not in self._price_extractor_cache:
            self._price_extractor_cache[data_key] = SignalPriceExtractorFactory.create(data)
        return self._price_extractor_cache[data_key]

    def _get_column_handler(self, data: pd.DataFrame) -> IColumnHandler:
        """Get appropriate column handler for the DataFrame structure."""
        data_key = id(data.columns)
        if data_key not in self._column_handler_cache:
            self._column_handler_cache[data_key] = ColumnHandlerFactory.create(data)
        return self._column_handler_cache[data_key]

    @classmethod
    def tunable_parameters(cls) -> dict[str, dict[str, Any]]:
        return {
            param: {"type": "float", "min": 0, "max": 1}
            for param in [
                "direction",
                "entry_day",
                "hold_days",
                "use_ema_filter",
                "fast_ema_len",
                "slow_ema_multiplier",
                "trade_longs",
                "trade_shorts",
            ]
            + [f"trade_month_{i}" for i in range(1, 13)]
        }

    def get_minimum_required_periods(self) -> int:
        """
        This strategy does not depend on a long history, but a few months
        of data is good for handling edge cases around month ends.
        """
        # Adjust based on EMA lengths
        if self.use_ema_filter:
            return self.slow_ema_len + 5  # Add a small buffer
        return 1  # Default if EMA is not used

    def get_synthetic_data_requirements(self) -> bool:
        """
        This strategy does not require synthetic data generation.
        """
        return False

    # Delegate base strategy methods to interface instead of using super()
    def get_timing_controller(self):
        """Get the timing controller via interface delegation."""
        return self._strategy_base.get_timing_controller()

    def supports_daily_signals(self) -> bool:
        """Check if strategy supports daily signals via interface delegation."""
        return self._strategy_base.supports_daily_signals()

    def get_roro_signal(self):
        """Get RoRo signal - returns None by default for this signal strategy."""
        return None

    def get_stop_loss_handler(self):
        """Get stop loss handler via interface delegation."""
        return self._strategy_base.get_stop_loss_handler()

    def get_universe(self, global_config: Dict[str, Any]):
        """Get universe via interface delegation."""
        return self._strategy_base.get_universe(global_config)

    def get_universe_method_with_date(
        self, global_config: Dict[str, Any], current_date: pd.Timestamp
    ):
        """Get universe with date context via interface delegation."""
        return self._strategy_base.get_universe_method_with_date(global_config, current_date)

    def filter_universe_by_data_availability(
        self,
        all_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
        min_periods_override: Optional[int] = None,
    ) -> list:
        """
        Overrides the base method to simplify data availability check for intramonth strategy.
        Only includes assets that have data for the current_date.

        SOLID: Uses polymorphic price extractor instead of isinstance checks
        """
        if all_historical_data.empty or current_date not in all_historical_data.index:
            return []

        # Use polymorphic extractor - eliminates isinstance violations
        extractor = self._get_price_extractor(all_historical_data)
        asset_list = extractor.get_available_tickers(all_historical_data)

        valid_assets = []
        for asset in asset_list:
            # Use polymorphic validation instead of direct isinstance checks
            if extractor.validate_ticker_data_availability(
                all_historical_data, asset, current_date
            ):
                valid_assets.append(asset)

        return valid_assets

    def validate_data_sufficiency(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> tuple[bool, str]:
        # For this strategy, we only need data for the current day.
        # The base method is too restrictive.
        if current_date in all_historical_data.index:
            return True, ""
        else:
            return False, f"No data for current_date: {current_date}"

    def _calculate_ema_values(
        self, all_historical_data: pd.DataFrame, valid_assets: list, current_date: pd.Timestamp
    ) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
        """
        Helper to calculate EMA values.

        SOLID: Uses polymorphic data type processor instead of isinstance checks
        """
        if self.use_ema_filter and len(all_historical_data) > self.slow_ema_len:
            # Filter for Close prices using polymorphic approach
            close_prices_data = all_historical_data.loc[:current_date]

            # Use polymorphic extractor - eliminates isinstance violations
            extractor = self._get_price_extractor(close_prices_data)
            close_prices_df = extractor.extract_all_close_prices(close_prices_data, "Close")

            # Use polymorphic data type processor to ensure DataFrame
            close_prices = self._data_type_processor.process_close_prices_extraction(
                close_prices_df, "Close"
            )
            close_prices = close_prices[valid_assets]

            fast_ema_series = {
                col: EMAIndicator(
                    close_prices[col].dropna(), window=self.fast_ema_len
                ).ema_indicator()
                for col in valid_assets
            }
            fast_ema_df = pd.DataFrame(fast_ema_series)

            slow_ema_series = {
                col: EMAIndicator(
                    close_prices[col].dropna(), window=self.slow_ema_len
                ).ema_indicator()
                for col in valid_assets
            }
            slow_ema_df = pd.DataFrame(slow_ema_series)

            if not fast_ema_df.empty and not slow_ema_df.empty:
                fast_ema_values = fast_ema_df.iloc[-1]
                slow_ema_values = slow_ema_df.iloc[-1]

                # Print for demonstration
                asset_to_print = valid_assets[0]
                logger.debug(
                    "Date: %s | Asset: %s | Fast EMA: %.2f | Slow EMA: %.2f",
                    current_date.date(),
                    asset_to_print,
                    fast_ema_values.get(asset_to_print, float("nan")),
                    slow_ema_values.get(asset_to_print, float("nan")),
                )
                return fast_ema_values, slow_ema_values

        return None, None

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: Optional[pd.DataFrame] = None,
        current_date: Optional[pd.Timestamp] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        **kwargs,
    ) -> pd.DataFrame:
        # Handle None current_date gracefully - use the last date in the data
        if current_date is None:
            current_date = pd.Timestamp(all_historical_data.index[-1])

        # Use polymorphic extractor - eliminates isinstance violations
        extractor = self._get_price_extractor(all_historical_data)
        all_assets_list = extractor.get_available_tickers(all_historical_data)
        all_assets = np.array(all_assets_list)

        is_sufficient, _ = self.validate_data_sufficiency(
            all_historical_data, benchmark_historical_data, current_date
        )
        if not is_sufficient:
            return pd.DataFrame(0.0, index=[current_date], columns=all_assets)

        valid_assets = self.filter_universe_by_data_availability(all_historical_data, current_date)
        if not valid_assets:
            return pd.DataFrame(0.0, index=[current_date], columns=all_assets)

        # Debugging output via logger
        logger.debug(
            "Date: %s | Strategy Params: fast_ema_len=%d, slow_ema_multiplier=%.2f, use_ema_filter=%s",
            current_date.date(),
            self.fast_ema_len,
            self.slow_ema_multiplier,
            self.use_ema_filter,
        )

        fast_ema_values, slow_ema_values = self._calculate_ema_values(
            all_historical_data, valid_assets, current_date
        )

        params = self.strategy_config.get("strategy_params", self.strategy_config)
        direction = params["direction"]
        entry_day = params["entry_day"]
        hold_days = params["hold_days"]
        allowed_months = [i for i in range(1, 13) if params.get(f"trade_month_{i}", True)]

        asset_idx_map = {asset: i for i, asset in enumerate(all_assets)}
        valid_indices = np.array(
            [asset_idx_map[a] for a in valid_assets if a in asset_idx_map], dtype=int
        )

        if (
            not hasattr(self, "_last_weights_np")
            or self._last_weights_np is None
            or self._last_weights_np.shape[0] != len(all_assets)
        ):
            self._last_weights_np = np.zeros(len(all_assets), dtype=np.float64)
        target_weights_np = self._last_weights_np.copy()

        if self._exit_dates_np is None or self._exit_dates_np.shape[0] != len(all_assets):
            self._exit_dates_np = np.full(
                len(all_assets), np.datetime64("NaT"), dtype="datetime64[ns]"
            )

        # Exit positions that have reached their exit date
        exit_mask = (~pd.isna(self._exit_dates_np)) & (
            self._exit_dates_np <= np.datetime64(current_date)
        )
        if exit_mask.any():
            target_weights_np[exit_mask] = 0
            self._exit_dates_np[exit_mask] = np.datetime64("NaT")
            if logger.isEnabledFor(logging.DEBUG):
                num_exits = np.count_nonzero(exit_mask)
                logger.debug(f"Exiting {num_exits} positions on {current_date.date()}")

        entry_date = (
            self.get_entry_date_for_month(current_date, entry_day)
            if current_date.month in allowed_months
            else None
        )

        if entry_date is not None and current_date == entry_date:
            year, month = current_date.year, current_date.month
            cache_key = (year, month)
            if cache_key not in self._bday_range_cache:
                start_of_month = current_date.replace(day=1)
                end_of_month = start_of_month + pd.offsets.MonthEnd(1)
                self._bday_range_cache[cache_key] = pd.bdate_range(
                    start=start_of_month, end=end_of_month
                )
            b_days = self._bday_range_cache[cache_key]

            if (entry_day > 0 and entry_day > len(b_days)) or (
                entry_day < 0 and abs(entry_day) > len(b_days)
            ):
                return pd.DataFrame(0.0, index=[current_date], columns=all_assets)

            not_in_position = (
                pd.isna(self._exit_dates_np[valid_indices])
                if self._exit_dates_np is not None
                else np.ones(len(valid_indices), dtype=bool)
            )
            if not_in_position.any():
                ema_condition_met_values = np.array([True])  # Default to True
                if (
                    self.use_ema_filter
                    and fast_ema_values is not None
                    and slow_ema_values is not None
                ):
                    if direction == "long":
                        ema_condition_met_values = np.array(
                            (fast_ema_values > slow_ema_values)
                            .reindex(valid_assets)
                            .fillna(False)
                            .tolist()
                        )
                    else:  # short
                        ema_condition_met_values = np.array(
                            (fast_ema_values < slow_ema_values)
                            .reindex(valid_assets)
                            .fillna(False)
                            .tolist()
                        )

                    # Debugging
                    logger.debug("EMA Condition Met: %s", ema_condition_met_values)

                # Convert to numpy arrays using polymorphic approach - eliminates isinstance violations
                not_in_position_np = self._data_type_processor.process_boolean_conditions(
                    not_in_position
                )
                ema_condition_met_np = self._data_type_processor.process_boolean_conditions(
                    ema_condition_met_values
                )
                trade_mask = not_in_position_np & ema_condition_met_np.astype(bool)

                # Debugging
                logger.debug("Trade Mask: %s", trade_mask)

                if direction == "long" and self.trade_longs:
                    target_weights_np[valid_indices[trade_mask]] = 1.0
                elif direction == "short" and self.trade_shorts:
                    target_weights_np[valid_indices[trade_mask]] = -1.0

                exit_date = np.datetime64(current_date + self._bday_offset * hold_days)
                self._exit_dates_np[valid_indices[trade_mask]] = exit_date

        num_long = np.count_nonzero(target_weights_np > 0)
        if num_long > 0:
            target_weights_np[target_weights_np > 0] = 1.0 / num_long

        num_short = np.count_nonzero(target_weights_np < 0)
        if num_short > 0:
            target_weights_np[target_weights_np < 0] = -1.0 / num_short

        self._last_weights_np = target_weights_np.copy()
        result_df = pd.DataFrame(
            [target_weights_np], index=[current_date], columns=all_assets
        ).rename_axis(None)

        # Enforce trade direction constraints - this will raise an exception if violated
        result_df = self._enforce_trade_direction_constraints(result_df)

        return result_df

    def get_entry_date_for_month(self, date: pd.Timestamp, entry_day: int) -> pd.Timestamp:
        """Calculates the target entry date for a given month, using an internal cache for speed."""
        cache_key = (date.year, date.month, entry_day)
        if cache_key in self._entry_date_cache:
            return self._entry_date_cache[cache_key]

        start_of_month = date.replace(day=1)
        end_of_month = start_of_month + pd.offsets.MonthEnd(1)
        b_days = pd.bdate_range(start=start_of_month, end=end_of_month)

        if entry_day > 0:
            if entry_day > len(b_days):
                entry_date = b_days[-1]
            else:
                entry_date = b_days[entry_day - 1]
        else:
            if abs(entry_day) > len(b_days):
                entry_date = b_days[0]
            else:
                entry_date = b_days[entry_day]

        # Store in cache and return
        self._entry_date_cache[cache_key] = entry_date
        return entry_date

    def reset_state(self) -> None:
        """Resets the internal state of the strategy."""
        self._last_weights = None
        self._exit_dates_np = None
