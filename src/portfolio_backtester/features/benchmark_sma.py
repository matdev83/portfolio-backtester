from .base import Feature
import pandas as pd

class BenchmarkSMA(Feature):
    """Computes the Simple Moving Average (SMA) for the benchmark."""

    def __init__(self, sma_filter_window: int):
        super().__init__(sma_filter_window=sma_filter_window)
        self.sma_filter_window = sma_filter_window

    @property
    def name(self) -> str:
        return f"benchmark_sma_{self.sma_filter_window}m"

    def compute(self, data: pd.DataFrame, benchmark_data: pd.Series | None = None) -> pd.Series:
        if benchmark_data is None:
            raise ValueError("Benchmark data is required for BenchmarkSMA feature.")
        return (benchmark_data > benchmark_data.rolling(self.sma_filter_window).mean()).astype(int)