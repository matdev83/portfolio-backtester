from ..feature import Feature # Corrected import
import numpy as np
import pandas as pd

# Optional Numba-accelerated kernel
try:
    from ..numba_optimized import sortino_fast
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

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
        # Calculate returns (rows=time, columns=assets)
        rets = data.pct_change(fill_method=None)

        if NUMBA_AVAILABLE and not rets.empty:
            returns_np = rets.to_numpy(dtype=np.float64)
            sortino_mat = sortino_fast(returns_np, self.rolling_window, self.target_return, annualization_factor=12.0)
            sortino_df = pd.DataFrame(sortino_mat, index=rets.index, columns=rets.columns)
        else:
            cal_factor = np.sqrt(12)

            rolling_mean = rets.rolling(self.rolling_window).mean()

            def downside_deviation(series):
                downside_returns = series[series < self.target_return]
                if len(downside_returns) == 0:
                    return 1e-9
                return np.sqrt(np.mean((downside_returns - self.target_return) ** 2))

            rolling_downside_dev = rets.rolling(self.rolling_window).apply(downside_deviation, raw=False)
            excess_return = rolling_mean - self.target_return
            stable_downside_dev = np.maximum(rolling_downside_dev, 1e-6)
            sortino_df = (excess_return * cal_factor) / (stable_downside_dev * cal_factor)

        sortino_df = sortino_df.clip(-10.0, 10.0)
        return sortino_df.fillna(0)