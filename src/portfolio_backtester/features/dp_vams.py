from ..feature import Feature
import numpy as np
import pandas as pd

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
        rets = data.pct_change().fillna(0)
        momentum = (1 + rets).rolling(self.lookback_months).apply(np.prod, raw=True) - 1
        
        negative_rets = rets[rets < 0].fillna(0)
        downside_dev = negative_rets.rolling(self.lookback_months).std().fillna(0)
        total_vol = rets.rolling(self.lookback_months).std().fillna(0)
        
        denominator = self.alpha * downside_dev + (1 - self.alpha) * total_vol
        denominator = denominator.replace(0, np.nan)
        
        dp_vams = momentum / denominator
        return dp_vams.fillna(0)