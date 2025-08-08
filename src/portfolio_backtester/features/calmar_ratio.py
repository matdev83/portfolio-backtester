from ..feature import Feature
import numpy as np
import pandas as pd

# Direct import of optimized function - no fallback needed
from ..numba_optimized import calmar_batch_fixed


class CalmarRatio(Feature):
    """Computes the Calmar ratio."""

    def __init__(self, rolling_window: int):
        super().__init__(rolling_window=rolling_window)
        self.rolling_window = rolling_window
        self.needs_close_prices_only = True

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

        # Use optimized batch calculation with proper maximum drawdown
        returns_np = rets.to_numpy(dtype=np.float64)
        calmar_matrix = calmar_batch_fixed(returns_np, self.rolling_window, cal_factor=12.0)
        calmar_ratio = pd.DataFrame(calmar_matrix, index=rets.index, columns=rets.columns)

        # Clip extreme values
        calmar_ratio = calmar_ratio.clip(-10.0, 10.0)

        return calmar_ratio
