import logging
import numpy as np
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from pandas.tseries.offsets import BDay

from .base_strategy import BaseStrategy
from ..datetime_utils import get_bday_offset

logger = logging.getLogger(__name__)


class IntramonthSeasonalStrategy(BaseStrategy):
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
        self.positions: Dict[str, Dict[str, Any]] = {}

        defaults = {
            "direction": "long",  # "long" or "short"
            "entry_day": 5,  # n-th business day
            "hold_days": 10,  # m business days
            "price_column_asset": "Close",
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
        return {"direction", "entry_day", "hold_days"}

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
        is_sufficient, reason = self.validate_data_sufficiency(
            all_historical_data, benchmark_historical_data, current_date
        )
        if not is_sufficient:
            columns = (
                all_historical_data.columns.get_level_values(0).unique()
                if isinstance(all_historical_data.columns, pd.MultiIndex)
                else all_historical_data.columns
            )
            return pd.DataFrame(0.0, index=[current_date], columns=columns)

        valid_assets = self.filter_universe_by_data_availability(
            all_historical_data, current_date, min_periods_override=1
        )

        # Strategy can work with multiple assets

        if not valid_assets:
            columns = (
                all_historical_data.columns.get_level_values(0).unique()
                if isinstance(all_historical_data.columns, pd.MultiIndex)
                else all_historical_data.columns
            )
            return pd.DataFrame(0.0, index=[current_date], columns=columns)

        params = self.strategy_config.get("strategy_params", self.strategy_config)
        direction = params["direction"]
        entry_day = params["entry_day"]
        hold_days = params["hold_days"]

        if self._last_weights is None:
            self._last_weights = pd.Series(0.0, index=valid_assets)
        else:
            self._last_weights = self._last_weights.reindex(valid_assets, fill_value=0.0)

        target_weights = self._last_weights.copy()

        # Check for exits
        for asset, pos_info in list(self.positions.items()):
            if current_date >= pos_info["exit_date"]:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Exiting position for {asset} on {current_date}")
                target_weights[asset] = 0
                del self.positions[asset]

        # Check for entries
        entry_date = self.get_entry_date_for_month(current_date, entry_day)

        if current_date == entry_date:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Entry condition met for {current_date}. Assets: {valid_assets}")
            for asset in valid_assets:
                if asset not in self.positions:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Entering position for {asset} on {current_date}")
                    target_weights[asset] = 1.0 if direction == "long" else -1.0
                    exit_date = current_date + self._bday_offset * hold_days
                    self.positions[asset] = {"exit_date": exit_date}

        if (self._last_weights is not None) and (self._last_weights != 0).any():
            logger.debug(f"Date: {current_date}, Previous weights: {self._last_weights[self._last_weights != 0]}")
        if (target_weights != 0).any():
            logger.debug(f"Date: {current_date}, New weights: {target_weights[target_weights != 0]}")

        # Normalize weights to sum to 1 (or -1 for short)
        num_positions = (target_weights != 0).sum()
        if num_positions > 0:
            if direction == "long":
                target_weights[target_weights > 0] = 1.0 / num_positions
            else:
                target_weights[target_weights < 0] = -1.0 / num_positions

        self._last_weights = target_weights
        
        # Return a transposed DataFrame
        return pd.DataFrame(target_weights.to_dict(), index=[current_date]).reindex(columns=valid_assets).fillna(0.0).rename_axis(None)

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