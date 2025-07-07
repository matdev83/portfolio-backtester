from .base import Feature
import numpy as np
import pandas as pd

class SharpeRatio(Feature):
    """Computes the Sharpe ratio."""

    def __init__(self, rolling_window: int):
        super().__init__(rolling_window=rolling_window)
        self.rolling_window = rolling_window
        self.needs_close_prices_only = True

    @property
    def name(self) -> str:
        return f"sharpe_{self.rolling_window}m"

    def compute(self, data: pd.DataFrame, benchmark_data: pd.Series | None = None) -> pd.DataFrame:
        rets = data.pct_change().fillna(0)
        cal_factor = np.sqrt(12)
        rolling_mean = rets.rolling(self.rolling_window).mean()
        rolling_std = rets.rolling(self.rolling_window).std()
        sharpe_ratio = (rolling_mean * cal_factor) / (rolling_std * cal_factor).replace(0, np.nan)
        return sharpe_ratio.fillna(0)