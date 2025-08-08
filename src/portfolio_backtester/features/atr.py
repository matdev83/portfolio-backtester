import pandas as pd
import numpy as np

# Direct import of optimized function - no fallback needed
from ..numba_optimized import atr_fast_fixed
from ..feature import Feature


class ATRFeature(Feature):
    """Average True Range (ATR) feature implementation."""

    def __init__(self, atr_period: int = 14):
        super().__init__(atr_period=atr_period)
        self.atr_period = atr_period

    @property
    def name(self) -> str:
        return f"atr_{self.atr_period}"

    def compute(self, data: pd.DataFrame, benchmark_data: pd.Series | None = None) -> pd.DataFrame:
        """
        Compute ATR for OHLC data.

        Args:
            data: DataFrame with MultiIndex columns (Ticker, Field) containing OHLC data
            benchmark_data: Not used for ATR calculation

        Returns:
            DataFrame with ATR values indexed by date, columns by asset tickers
        """
        if data.empty:
            return pd.DataFrame()

        # Handle MultiIndex columns (Ticker, Field)
        if isinstance(data.columns, pd.MultiIndex) and "Ticker" in data.columns.names:
            assets = data.columns.get_level_values("Ticker").unique()
            atr_df = pd.DataFrame(index=data.index)

            for asset in assets:
                try:
                    # Extract OHLC data for this asset
                    high = data[(asset, "High")]
                    low = data[(asset, "Low")]
                    close = data[(asset, "Close")]

                    # Use optimized calculation with proper edge case handling
                    if (
                        len(high) > 1
                        and not high.isna().all()
                        and not low.isna().all()
                        and not close.isna().all()
                    ):

                        # Use optimized Numba calculation with proper type conversion
                        atr_values = atr_fast_fixed(
                            high.values.astype(np.float64),
                            low.values.astype(np.float64),
                            close.values.astype(np.float64),
                            self.atr_period,
                        )
                        atr = pd.Series(atr_values, index=high.index, name=f"ATR_{asset}")

                    else:
                        # Handle edge case with insufficient data
                        atr = pd.Series(np.nan, index=high.index, name=f"ATR_{asset}")

                    atr_df[asset] = atr

                except KeyError:
                    # If OHLC data is not available for this asset, fill with NaN
                    atr_df[asset] = np.nan

            return atr_df

        else:
            # Handle case where data doesn't have the expected MultiIndex structure
            # This might happen if data is just close prices - cannot calculate ATR
            raise ValueError(
                "ATR calculation requires OHLC data with MultiIndex columns (Ticker, Field)"
            )

    def __eq__(self, other):
        if not isinstance(other, ATRFeature):
            return NotImplemented
        return self.atr_period == other.atr_period

    def __hash__(self):
        return hash((self.__class__, self.atr_period))
