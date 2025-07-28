from ..feature import Feature
import numpy as np
import pandas as pd

# Import Numba optimization with fallback
try:
    from ..numba_optimized import vams_batch_fast
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

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
        rets = data.pct_change(fill_method=None).fillna(0)
        
        if NUMBA_AVAILABLE and not rets.empty:
            # Use Numba-optimized batch calculation
            returns_np = rets.to_numpy(dtype=np.float64)
            vams_matrix = vams_batch_fast(returns_np, self.lookback_months)
            vams = pd.DataFrame(vams_matrix, index=rets.index, columns=rets.columns)
        else:
            # Fallback to pandas implementation
            momentum = (1 + rets).rolling(self.lookback_months).apply(np.prod, raw=True) - 1
            total_vol = rets.rolling(self.lookback_months).std()
            denominator = total_vol.replace(0, np.nan)
            vams = momentum / denominator
            
        return vams