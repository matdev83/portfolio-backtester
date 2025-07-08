from ..feature import Feature
import numpy as np
import pandas as pd

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
        rets = data.pct_change(fill_method=None).fillna(0)
        cal_factor = 12
        rolling_mean = rets.rolling(self.rolling_window).mean() * cal_factor

        def max_drawdown(series):
            series = series.dropna()
            if series.empty:
                return 0.0
            cumulative_returns = (1 + series).cumprod()
            peak = cumulative_returns.expanding(min_periods=1).max()
            peak = peak.replace(0, 1e-9)
            drawdown = (cumulative_returns / peak) - 1
            drawdown = drawdown.replace([np.inf, -np.inf], [0, 0]).fillna(0)
            min_drawdown = abs(drawdown.min())
            return min_drawdown

        rolling_max_dd = rets.rolling(self.rolling_window).apply(max_drawdown, raw=False)
        with np.errstate(divide='ignore', invalid='ignore'):
            calmar_ratio = rolling_mean / rolling_max_dd
        calmar_ratio.replace([np.inf, -np.inf], [10.0, -10.0], inplace=True)
        calmar_ratio = calmar_ratio.clip(-10.0, 10.0)
        return calmar_ratio