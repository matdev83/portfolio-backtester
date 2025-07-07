from .base import Feature
import pandas as pd

class Momentum(Feature):
    """Computes momentum for each asset: P(t-skip_months)/P(t-skip_months-lookback_months) - 1."""

    def __init__(self, lookback_months: int, skip_months: int = 0, name_suffix: str = ""):
        super().__init__(lookback_months=lookback_months, skip_months=skip_months, name_suffix=name_suffix)
        self.lookback_months = lookback_months
        self.skip_months = skip_months
        self.name_suffix = name_suffix
        self.needs_close_prices_only = True

    @property
    def name(self) -> str:
        if self.skip_months == 0 and not self.name_suffix:
            return f"momentum_{self.lookback_months}m"

        base_name = f"momentum_{self.lookback_months}m_skip{self.skip_months}m"
        if self.name_suffix:
            return f"{base_name}_{self.name_suffix}"
        return base_name

    def compute(self, data: pd.DataFrame, benchmark_data: pd.Series | None = None) -> pd.DataFrame:
        shifted_prices = data.shift(self.skip_months)
        momentum = (shifted_prices / shifted_prices.shift(self.lookback_months)) - 1
        return momentum