import logging
import math
import numpy as np
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from pandas.tseries.offsets import BDay
from ta.trend import EMAIndicator

from ..base.signal_strategy import SignalStrategy
from ...datetime_utils import get_bday_offset

logger = logging.getLogger(__name__)


class IntramonthSeasonalStrategy(SignalStrategy):
    # Cache for business day calendars and month ends (class-level, shared)
    _bday_range_cache = {}
    _month_end_cache = {}
    """
    Intramonth seasonal trading strategy.

    - Long-only or short-only (configurable).
    - Buy (or sell for short-only) on the n-th business day of the month.
      - n can be positive (from the start of the month) or negative (from the end of the month).
    - Sell (or buy to cover) m business days later.
    """

    def __init__(self, strategy_config: Dict[str, Any]):
        super().__init__(strategy_config)
        self._last_weights: Optional[pd.Series] = None
        # Use Numpy arrays for position tracking
        self._exit_dates_np = None  # Numpy array of exit dates (as int64 timestamps)

        defaults = {
            "direction": "long",
            "entry_day": 5,
            "hold_days": 10,
            "price_column_asset": "Close",
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

        # PERFORMANCE: cache reusable objects -------------------------------
        # 1. Business-day offset is the same for the whole simulation, build it once.
        self._bday_offset = get_bday_offset()
        # 2. Cache entry date per (year, month, entry_day) to avoid recomputing
        #    bdate_range inside every generate_signals call.
        self._entry_date_cache: Dict[Tuple[int, int, int], pd.Timestamp] = {}

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        return {
            "direction",
            "entry_day",
            "hold_days",
            "use_ema_filter",
            "fast_ema_len",
            "slow_ema_multiplier",
        } | {f"trade_month_{i}" for i in range(1, 13)}

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

    def filter_universe_by_data_availability(
        self,
        all_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
        min_periods_override: Optional[int] = None,
    ) -> list:
        """
        Overrides the base method to simplify data availability check for intramonth strategy.
        Only includes assets that have data for the current_date.
        """
        if all_historical_data.empty or current_date not in all_historical_data.index:
            return []

        valid_assets = []
        current_day_data = all_historical_data.loc[current_date]

        if isinstance(all_historical_data.columns, pd.MultiIndex):
            asset_list = all_historical_data.columns.get_level_values("Ticker").unique()
            for asset in asset_list:
                if (asset, "Close") in all_historical_data.columns and np.all(
                    pd.notna(current_day_data[(asset, "Close")])
                ):
                    valid_assets.append(asset)
        else:
            asset_list = all_historical_data.columns
            for asset in asset_list:
                if asset in current_day_data.index and np.all(pd.notna(current_day_data[asset])):
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
        """Helper to calculate EMA values."""
        if self.use_ema_filter and len(all_historical_data) > self.slow_ema_len:
            close_prices = all_historical_data.loc[:current_date, (slice(None), "Close")].droplevel(1, axis=1)[
                valid_assets
            ]

            fast_ema_series = {
                col: EMAIndicator(close_prices[col].dropna(), window=self.fast_ema_len).ema_indicator()
                for col in valid_assets
            }
            fast_ema_df = pd.DataFrame(fast_ema_series)

            slow_ema_series = {
                col: EMAIndicator(close_prices[col].dropna(), window=self.slow_ema_len).ema_indicator()
                for col in valid_assets
            }
            slow_ema_df = pd.DataFrame(slow_ema_series)

            if not fast_ema_df.empty and not slow_ema_df.empty:
                fast_ema_values = fast_ema_df.iloc[-1]
                slow_ema_values = slow_ema_df.iloc[-1]

                # Print for demonstration
                asset_to_print = valid_assets[0]
                print(
                    f"Date: {current_date.date()} | Asset: {asset_to_print} | "
                    f"Fast EMA: {fast_ema_values.get(asset_to_print, 'N/A'):.2f} | "
                    f"Slow EMA: {slow_ema_values.get(asset_to_print, 'N/A'):.2f}"
                )
                return fast_ema_values, slow_ema_values

        return None, None

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        if isinstance(all_historical_data.columns, pd.MultiIndex):
            all_assets = all_historical_data.columns.get_level_values(0).unique().to_numpy()
        else:
            all_assets = all_historical_data.columns.to_numpy()

        is_sufficient, _ = self.validate_data_sufficiency(
            all_historical_data, benchmark_historical_data, current_date
        )
        if not is_sufficient:
            return pd.DataFrame(0.0, index=[current_date], columns=all_assets)

        valid_assets = self.filter_universe_by_data_availability(all_historical_data, current_date)
        if not valid_assets:
            return pd.DataFrame(0.0, index=[current_date], columns=all_assets)

        # --- Debugging --- #
        print(f"[DEBUG] Date: {current_date.date()} | Strategy Params: fast_ema_len={self.fast_ema_len}, slow_ema_multiplier={self.slow_ema_multiplier}, use_ema_filter={self.use_ema_filter}")

        fast_ema_values, slow_ema_values = self._calculate_ema_values(
            all_historical_data, valid_assets, current_date
        )

        params = self.strategy_config.get("strategy_params", self.strategy_config)
        direction = params["direction"]
        entry_day = params["entry_day"]
        hold_days = params["hold_days"]
        allowed_months = [i for i in range(1, 13) if params.get(f"trade_month_{i}", True)]

        asset_idx_map = {asset: i for i, asset in enumerate(all_assets)}
        valid_indices = np.array([asset_idx_map[a] for a in valid_assets if a in asset_idx_map], dtype=int)

        if not hasattr(self, "_last_weights_np") or self._last_weights_np.shape[0] != len(all_assets):
            self._last_weights_np = np.zeros(len(all_assets), dtype=np.float64)
        target_weights_np = self._last_weights_np.copy()

        if self._exit_dates_np is None or self._exit_dates_np.shape[0] != len(all_assets):
            self._exit_dates_np = np.full(len(all_assets), np.datetime64("NaT"), dtype="datetime64[ns]")

        exit_mask = (~pd.isna(self._exit_dates_np)) & (self._exit_dates_np <= np.datetime64(current_date))
        target_weights_np[exit_mask] = 0
        self._exit_dates_np[exit_mask] = np.datetime64("NaT")

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

            not_in_position = pd.isna(self._exit_dates_np[valid_indices])
            if not_in_position.any():
                ema_condition_met = True  # Default to True
                if self.use_ema_filter and fast_ema_values is not None and slow_ema_values is not None:
                    if direction == "long":
                        ema_condition_met = fast_ema_values > slow_ema_values
                    else:  # short
                        ema_condition_met = fast_ema_values < slow_ema_values
                    
                    # Align ema_condition_met with valid_indices
                    ema_condition_met = ema_condition_met.reindex(valid_assets).fillna(False).values

                    # --- Debugging ---
                    print(f"[DEBUG] EMA Condition Met: {ema_condition_met}")
                
                trade_mask = not_in_position & ema_condition_met

                # --- Debugging ---
                print(f"[DEBUG] Trade Mask: {trade_mask}")

                if direction == "long":
                    target_weights_np[valid_indices[trade_mask]] = 1.0
                elif direction == "short":
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
        return pd.DataFrame([target_weights_np], index=[current_date], columns=all_assets).rename_axis(
            None
        )

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