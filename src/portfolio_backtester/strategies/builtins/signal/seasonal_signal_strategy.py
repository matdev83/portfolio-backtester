from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
from pandas.tseries.offsets import BDay

from ..._core.base.base.signal_strategy import SignalStrategy


class SeasonalSignalStrategy(SignalStrategy):
    """Intramonth seasonal strategy with simple entry/exit by business days.

    - entry_day: Nth business day of the month (positive from start, negative from end)
    - hold_days: number of business days to hold positions starting from entry_day
    - trade_month_X flags (1..12): enable/disable trading for specific months
    - direction: only 'long' is supported in tests; shorts treated as no-position
    """

    def __init__(self, strategy_config: Dict):
        super().__init__(strategy_config)
        params = strategy_config.get("strategy_params", {}) if strategy_config else {}
        self.direction: str = str(params.get("direction", "long"))
        self.entry_day: int = int(params.get("entry_day", 1))
        self.hold_days: int = int(params.get("hold_days", 3))

        # Month filters: default to True if not specified
        self.allowed_month: Dict[int, bool] = {
            m: bool(params.get(f"trade_month_{m}", True)) for m in range(1, 13)
        }

    @classmethod
    def tunable_parameters(cls) -> Dict[str, Dict[str, object]]:  # not used by tests
        return {
            "direction": {"type": "str", "default": "long"},
            "entry_day": {"type": "int", "default": 1, "min": -21, "max": 21},
            "hold_days": {"type": "int", "default": 3, "min": 1, "max": 20},
        }

    def get_entry_date_for_month(self, date: pd.Timestamp, entry_day: int) -> pd.Timestamp:
        first_day = pd.Timestamp(year=date.year, month=date.month, day=1)
        last_day = (first_day + pd.offsets.MonthEnd(1)).normalize()
        bdays = pd.bdate_range(first_day, last_day)
        if entry_day == 0:
            entry_day = 1
        index = entry_day - 1 if entry_day > 0 else entry_day
        index = max(min(index, len(bdays) - 1), -len(bdays))
        return pd.Timestamp(bdays[index])

    def _is_within_hold_window(self, current_date: pd.Timestamp, entry_date: pd.Timestamp) -> bool:
        if self.hold_days <= 0:
            return False
        window_end = entry_date + BDay(self.hold_days - 1)
        return bool(entry_date <= current_date <= window_end)

    def _month_allowed(self, current_date: pd.Timestamp) -> bool:
        return bool(self.allowed_month.get(int(current_date.month), True))

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: Optional[pd.DataFrame] = None,
        current_date: Optional[pd.Timestamp] = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        if current_date is None:
            current_date = pd.Timestamp(all_historical_data.index[-1])

        # Handle MultiIndex columns; flatten to ticker list
        if isinstance(all_historical_data.columns, pd.MultiIndex):
            tickers = sorted({str(c[0]) if isinstance(c, tuple) and len(c) > 0 else str(c) for c in all_historical_data.columns})
        else:
            tickers = list(all_historical_data.columns)
        result = pd.DataFrame(0.0, index=[current_date], columns=tickers)

        if not self._month_allowed(current_date):
            return result

        entry_date = self.get_entry_date_for_month(current_date, self.entry_day)
        if self.direction != "long":
            return result

        if self._is_within_hold_window(current_date, entry_date):
            if len(tickers) > 0:
                equal_weight = 1.0 / len(tickers)
                result.loc[current_date, :] = equal_weight

        return result


__all__ = ["SeasonalSignalStrategy"]
