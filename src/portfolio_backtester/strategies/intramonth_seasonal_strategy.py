import logging
from typing import Any, Dict, Optional

import pandas as pd
from pandas.tseries.offsets import BDay

from .base_strategy import BaseStrategy

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
        self.w_prev: Optional[pd.Series] = None
        self.positions: Dict[str, Dict[str, Any]] = {}  # To track open positions and their exit dates

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

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        return {"direction", "entry_day", "hold_days"}

    def get_minimum_required_periods(self) -> int:
        """
        This strategy does not depend on a long history, but a few months
        of data is good for handling edge cases around month ends.
        """
        return 1 # Only need data for the current day

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
                if (asset, "Close") in all_historical_data.columns and not pd.isna(
                    current_day_data[(asset, "Close")]
                ).all():
                    valid_assets.append(asset)
        else:
            asset_list = all_historical_data.columns
            for asset in asset_list:
                if asset in current_day_data.index and not pd.isna(
                    current_day_data[asset]
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

        if len(valid_assets) > 1:
            raise ValueError(
                "IntramonthSeasonalStrategy can only operate on a universe with a single symbol."
            )

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

        if self.w_prev is None:
            self.w_prev = pd.Series(0.0, index=valid_assets)
        else:
            self.w_prev = self.w_prev.reindex(valid_assets).fillna(0.0)

        target_weights = self.w_prev.copy()

        # Check for exits
        for asset, pos_info in list(self.positions.items()):
            if current_date >= pos_info["exit_date"]:
                target_weights[asset] = 0
                del self.positions[asset]

        # Check for entries
        entry_date = self.get_entry_date_for_month(current_date, entry_day)

        if current_date == entry_date:
            for asset in valid_assets:
                if asset not in self.positions:
                    target_weights[asset] = 1.0 if direction == "long" else -1.0
                    exit_date = current_date + BDay(hold_days)
                    self.positions[asset] = {"exit_date": exit_date}

        # Normalize weights to sum to 1 (or -1 for short)
        num_positions = (target_weights != 0).sum()
        if num_positions > 0:
            if direction == "long":
                target_weights[target_weights > 0] = 1.0 / num_positions
            else:
                target_weights[target_weights < 0] = -1.0 / num_positions

        self.w_prev = target_weights
        
        # Return a transposed DataFrame
        return pd.DataFrame(target_weights.to_dict(), index=[current_date]).reindex(columns=valid_assets).fillna(0.0).rename_axis(None)

    def get_entry_date_for_month(self, date: pd.Timestamp, entry_day: int) -> pd.Timestamp:
        """Calculates the target entry date for a given month."""
        start_of_month = date.replace(day=1)
        end_of_month = start_of_month + pd.offsets.MonthEnd(1)
        b_days = pd.bdate_range(start=start_of_month, end=end_of_month)
        
        if entry_day > 0:
            if entry_day > len(b_days):
                return b_days[-1]
            return b_days[entry_day - 1]
        else:
            if abs(entry_day) > len(b_days):
                return b_days[0]
            return b_days[entry_day]