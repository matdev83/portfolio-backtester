from ..feature import Feature
import numpy as np
import pandas as pd

# Direct import of optimized function - no fallback needed
from ..numba_optimized import sortino_fast_fixed


class SortinoRatio(Feature):
    """Computes the Sortino ratio."""

    def __init__(self, rolling_window: int, target_return: float = 0.0):
        super().__init__(rolling_window=rolling_window, target_return=target_return)
        self.rolling_window = rolling_window
        self.target_return = target_return
        self.needs_close_prices_only = True

    @property
    def name(self) -> str:
        return f"sortino_{self.rolling_window}m"

    def compute(self, data: pd.DataFrame, benchmark_data: pd.Series | None = None) -> pd.DataFrame:
        # Handle empty data edge case
        if data.empty:
            return pd.DataFrame()

        # Calculate returns (rows=time, columns=assets)
        rets = data.pct_change(fill_method=None)
        rets = rets.infer_objects().fillna(0.0)

        # Handle edge case where all returns are zero or NaN
        if rets.empty:
            return pd.DataFrame(index=data.index, columns=data.columns)

        # Use optimized batch calculation with proper ddof=1 handling
        returns_np = rets.to_numpy(dtype=np.float64)
        sortino_mat = sortino_fast_fixed(
            returns_np, self.rolling_window, self.target_return, annualization_factor=12.0
        )
        sortino_df = pd.DataFrame(sortino_mat, index=rets.index, columns=rets.columns)

        sortino_df = sortino_df.clip(-10.0, 10.0)

        # Only fill NaN values after the initial window period
        # Keep the first (rolling_window - 1) values as NaN
        mask = np.arange(len(sortino_df)) >= self.rolling_window - 1
        mask_indices = np.where(mask)[0]
        sortino_df.iloc[mask_indices] = sortino_df.iloc[mask_indices].fillna(0)
        return sortino_df
