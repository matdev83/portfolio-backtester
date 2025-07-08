from ..feature import Feature
import numpy as np
import pandas as pd

class VAMS(Feature):
    """Computes Volatility Adjusted Momentum Scores (VAMS)."""

    def __init__(self, lookback_months: int):
        super().__init__(lookback_months=lookback_months)
        self.lookback_months = lookback_months
        self.needs_close_prices_only = True

    @property
    def name(self) -> str:
        return f"vams_{self.lookback_months}m"

    def compute(self, data: pd.DataFrame, benchmark_data: pd.Series | None = None) -> pd.DataFrame:
        rets = data.pct_change().fillna(0)
        momentum = (1 + rets).rolling(self.lookback_months).apply(np.prod, raw=True) - 1
        total_vol = rets.rolling(self.lookback_months).std()
        denominator = total_vol.replace(0, np.nan)
        vams = momentum / denominator
        return vams