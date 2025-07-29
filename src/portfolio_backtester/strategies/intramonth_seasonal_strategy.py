import logging
import numpy as np
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from pandas.tseries.offsets import BDay

from .base_strategy import BaseStrategy
from ..datetime_utils import get_bday_offset

logger = logging.getLogger(__name__)


class IntramonthSeasonalStrategy(BaseStrategy):
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
            "direction": "long",  # "long" or "short"
            "entry_day": 5,  # n-th business day
            "hold_days": 10,  # m business days
            "price_column_asset": "Close",
            # Months in which the strategy is allowed to open new trades (1=Jan, …, 12=Dec)
            # By default all months are enabled.
            "allowed_months": list(range(1, 13)),
        }

        params_dict_to_update = self.strategy_config
        if "strategy_params" in self.strategy_config:
            if self.strategy_config["strategy_params"] is None:
                self.strategy_config["strategy_params"] = {}
            params_dict_to_update = self.strategy_config["strategy_params"]

        for k, v in defaults.items():
            params_dict_to_update.setdefault(k, v)

        # PERFORMANCE: cache reusable objects -------------------------------
        # 1. Business-day offset is the same for the whole simulation, build it once.
        self._bday_offset = get_bday_offset()
        # 2. Cache entry date per (year, month, entry_day) to avoid recomputing
        #    bdate_range inside every generate_signals call.
        self._entry_date_cache: Dict[Tuple[int, int, int], pd.Timestamp] = {}

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        # allowed_months can be optimised by passing categorical choices in the scenario config
        return {"direction", "entry_day", "hold_days", "allowed_months"}

    def get_minimum_required_periods(self) -> int:
        """
        This strategy does not depend on a long history, but a few months
        of data is good for handling edge cases around month ends.
        """
        return 1 # Only need data for the current day

    def get_synthetic_data_requirements(self) -> bool:
        """
        This strategy does not require synthetic data generation.
        """
        return False

    def filter_universe_by_data_availability(
        self,
        all_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
        min_periods_override: Optional[int] = None # This parameter is ignored in this override
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
            asset_list = all_historical_data.columns.get_level_values('Ticker').unique()
            for asset in asset_list:
                if (asset, "Close") in all_historical_data.columns and np.all(
                    pd.notna(current_day_data[(asset, "Close")])
                ):
                    valid_assets.append(asset)
        else:
            asset_list = all_historical_data.columns
            for asset in asset_list:
                if asset in current_day_data.index and np.all(
                    pd.notna(current_day_data[asset])
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

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        # --- Step 1: Extract all relevant arrays and info up front ---
        # Asset names
        if isinstance(all_historical_data.columns, pd.MultiIndex):
            all_assets = all_historical_data.columns.get_level_values(0).unique().to_numpy()
        else:
            all_assets = all_historical_data.columns.to_numpy()

        # Current day data as numpy
        if not all_historical_data.empty and current_date in all_historical_data.index:
            current_day_data = all_historical_data.loc[current_date].to_numpy()
        else:
            current_day_data = np.array([])

        # Validate data sufficiency
        is_sufficient, reason = self.validate_data_sufficiency(
            all_historical_data, benchmark_historical_data, current_date
        )
        if not is_sufficient:
            return pd.DataFrame(0.0, index=[current_date], columns=all_assets)

        # Valid assets (still as list for now)
        valid_assets = self.filter_universe_by_data_availability(
            all_historical_data, current_date, min_periods_override=1
        )

        if not valid_assets:
            return pd.DataFrame(0.0, index=[current_date], columns=all_assets)

        params = self.strategy_config.get("strategy_params", self.strategy_config)
        direction = params["direction"]
        entry_day = params["entry_day"]
        hold_days = params["hold_days"]
        allowed_months_param = params.get("allowed_months", list(range(1, 13)))

        # Normalise allowed_months to a list of unique ints 1..12
        if isinstance(allowed_months_param, (list, tuple, set)):
            allowed_months = sorted({int(m) for m in allowed_months_param})
        elif isinstance(allowed_months_param, int):
            allowed_months = [allowed_months_param]
        elif isinstance(allowed_months_param, str):
            # Accept comma-separated string like "1,3,5"
            allowed_months = sorted({int(part.strip()) for part in allowed_months_param.split(',') if part.strip()})
        else:
            raise ValueError("allowed_months must be an int, list/tuple/set of ints, or comma-separated str")

        # Validation – ensure 1 ≤ month ≤ 12 and at least one month enabled
        if not allowed_months:
            raise ValueError("IntramonthSeasonalStrategy: allowed_months must contain at least one month (1-12)")
        if any(m < 1 or m > 12 for m in allowed_months):
            raise ValueError("IntramonthSeasonalStrategy: allowed_months values must be between 1 and 12")

        # --- Step 2: Use Numpy arrays for weights ---

        # Map valid_assets to indices in all_assets
        asset_idx_map = {asset: i for i, asset in enumerate(all_assets)}
        valid_indices = np.array([asset_idx_map[a] for a in valid_assets if a in asset_idx_map], dtype=int)

        # Initialize last_weights and target_weights as Numpy arrays
        if not hasattr(self, '_last_weights_np') or self._last_weights_np.shape[0] != len(all_assets):
            self._last_weights_np = np.zeros(len(all_assets), dtype=np.float64)
        target_weights_np = self._last_weights_np.copy()

        # Initialize exit_dates array (int64 timestamps, NaT for no position)
        if self._exit_dates_np is None or self._exit_dates_np.shape[0] != len(all_assets):
            self._exit_dates_np = np.full(len(all_assets), np.datetime64('NaT'), dtype='datetime64[ns]')

        # Vectorized exit: set weights to 0 where exit_date <= current_date
        exit_mask = (~pd.isna(self._exit_dates_np)) & (self._exit_dates_np <= np.datetime64(current_date))
        target_weights_np[exit_mask] = 0
        self._exit_dates_np[exit_mask] = np.datetime64('NaT')

        # Vectorized entry: only if trading allowed this month
        if current_date.month in allowed_months:
            entry_date = self.get_entry_date_for_month(current_date, entry_day)
        else:
            entry_date = None

        if entry_date is not None and current_date == entry_date:
            # Validate that the entry_day is valid for the given month
            year, month = current_date.year, current_date.month
            cache_key = (year, month)
            if cache_key in self._month_end_cache:
                end_of_month = self._month_end_cache[cache_key]
            else:
                start_of_month = current_date.replace(day=1)
                end_of_month = start_of_month + pd.offsets.MonthEnd(1)
                self._month_end_cache[cache_key] = end_of_month
            if cache_key in self._bday_range_cache:
                b_days = self._bday_range_cache[cache_key]
            else:
                start_of_month = current_date.replace(day=1)
                b_days = pd.bdate_range(start=start_of_month, end=end_of_month)
                self._bday_range_cache[cache_key] = b_days

            if entry_day > 0 and entry_day > len(b_days):
                return pd.DataFrame(0.0, index=[current_date], columns=all_assets)
            elif entry_day < 0 and abs(entry_day) > len(b_days):
                return pd.DataFrame(0.0, index=[current_date], columns=all_assets)

            # Only enter for valid_indices not already in a position
            not_in_position = pd.isna(self._exit_dates_np[valid_indices])
            if not_in_position.any():
                if direction == "long":
                    target_weights_np[valid_indices[not_in_position]] = 1.0
                else:
                    target_weights_np[valid_indices[not_in_position]] = -1.0
                # Set exit dates for new positions
                exit_date = np.datetime64(current_date + self._bday_offset * hold_days)
                self._exit_dates_np[valid_indices[not_in_position]] = exit_date

        if hasattr(self, '_last_weights_np') and np.any(self._last_weights_np != 0):
            logger.debug(f"Date: {current_date}, Previous weights: {self._last_weights_np[self._last_weights_np != 0]}")
        if np.any(target_weights_np != 0):
            logger.debug(f"Date: {current_date}, New weights: {target_weights_np[target_weights_np != 0]}")

        # Normalize weights to sum to 1 (or -1 for short)
        num_positions = np.count_nonzero(target_weights_np)
        if num_positions > 0:
            if direction == "long":
                target_weights_np[target_weights_np > 0] = 1.0 / num_positions
            else:
                target_weights_np[target_weights_np < 0] = -1.0 / num_positions
                target_weights_np[target_weights_np > 0] = -1.0 / num_positions

        self._last_weights_np = target_weights_np.copy()

        # Output as DataFrame (single row)
        return pd.DataFrame([target_weights_np], index=[current_date], columns=all_assets).rename_axis(None)

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