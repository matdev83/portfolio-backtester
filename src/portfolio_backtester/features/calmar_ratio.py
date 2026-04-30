from ..feature import Feature
import numpy as np
import pandas as pd

# Direct import of optimized function - no fallback needed
from ..numba_optimized import calmar_batch_fixed


_TRADING_DAYS_PER_MONTH = 21
_DAILY_ANNUALIZATION = 252.0
_MONTHLY_ANNUALIZATION = 12.0
_MONTH_STYLE_WINDOW_MAX = 24


class CalmarRatio(Feature):
    """Computes the Calmar ratio."""

    def __init__(self, rolling_window: int):
        super().__init__(rolling_window=rolling_window)
        self.rolling_window = rolling_window

    @property
    def name(self) -> str:
        return f"calmar_{self.rolling_window}m"

    def compute(self, data: pd.DataFrame, benchmark_data: pd.Series | None = None) -> pd.DataFrame:
        # Handle empty data edge case
        if data.empty:
            return pd.DataFrame()

        rets = data.pct_change(fill_method=None).fillna(0)

        # Handle edge case where all returns are zero or NaN
        if rets.empty:
            return pd.DataFrame(index=data.index, columns=data.columns)

        effective_window = int(self.rolling_window)
        annualization_factor = _MONTHLY_ANNUALIZATION

        # Interpret small strategy-style windows as months on daily-like data.
        # Keep larger windows as direct periods for backward compatibility.
        if (
            len(rets.index) >= 2
            and effective_window > 0
            and effective_window <= _MONTH_STYLE_WINDOW_MAX
            and rets.index.to_series().diff().dropna().dt.days.median() <= 3
        ):
            effective_window *= _TRADING_DAYS_PER_MONTH
            annualization_factor = _DAILY_ANNUALIZATION

        # Use optimized batch calculation with proper maximum drawdown
        returns_np = rets.to_numpy(dtype=np.float64)
        calmar_matrix = calmar_batch_fixed(
            returns_np, effective_window, cal_factor=annualization_factor
        )
        calmar_ratio = pd.DataFrame(calmar_matrix, index=rets.index, columns=rets.columns)

        # Clip extreme values
        calmar_ratio = calmar_ratio.clip(-10.0, 10.0)

        return calmar_ratio
