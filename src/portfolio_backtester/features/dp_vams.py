from ..feature import Feature
import numpy as np
import pandas as pd

# Direct import of optimized function - no fallback needed
from ..numba_optimized import dp_vams_batch_fixed


class DPVAMS(Feature):
    """Computes Downside Penalized Volatility Adjusted Momentum Scores (dp-VAMS)."""

    def __init__(self, lookback_months: int, alpha: float):
        super().__init__(lookback_months=lookback_months, alpha=alpha)
        self.lookback_months = lookback_months
        self.alpha = alpha
        self.needs_close_prices_only = True

    @property
    def name(self) -> str:
        return f"dp_vams_{self.lookback_months}m_{self.alpha:.2f}a"

    def compute(self, data: pd.DataFrame, benchmark_data: pd.Series | None = None) -> pd.DataFrame:
        # Handle empty data edge case
        if data.empty:
            return pd.DataFrame()

        rets = data.pct_change(fill_method=None)
        rets = rets.infer_objects().fillna(0.0)

        # Handle edge case where all returns are zero or NaN
        if rets.empty:
            return pd.DataFrame(index=data.index, columns=data.columns)

        # Use optimized batch calculation with proper ddof=1 handling
        returns_np = rets.to_numpy(dtype=np.float64)
        dp_vams_matrix = dp_vams_batch_fixed(returns_np, self.lookback_months, self.alpha)
        dp_vams = pd.DataFrame(dp_vams_matrix, index=rets.index, columns=rets.columns)

        return dp_vams.fillna(0)
