from .base import Feature
import numpy as np
import pandas as pd
import pandas_ta as ta

class ATRFeature(Feature):
    """Computes Average True Range (ATR) for each asset."""

    def __init__(self, atr_period: int):
        super().__init__(atr_period=atr_period)
        self.atr_period = atr_period

    @property
    def name(self) -> str:
        return f"atr_{self.atr_period}"

    def compute(self, data: pd.DataFrame, benchmark_data: pd.Series | None = None) -> pd.DataFrame:
        if not isinstance(data.columns, pd.MultiIndex):
            raise ValueError(
                "ATRFeature expects data with MultiIndex columns (ticker, field). "
                "E.g., ('AAPL', 'High'). Please ensure the input data is formatted correctly. "
                "This typically means daily OHLC data aggregated appropriately if using monthly features."
            )

        all_atr_series = {}
        asset_tickers = data.columns.get_level_values(0).unique()

        for ticker in asset_tickers:
            try:
                high_prices = data[(ticker, "High")]
                low_prices = data[(ticker, "Low")]
                close_prices = data[(ticker, "Close")]

                atr_series = ta.atr(
                    high=high_prices,
                    low=low_prices,
                    close=close_prices,
                    length=self.atr_period,
                )
                if atr_series is not None:
                    all_atr_series[ticker] = atr_series.rename(ticker)
                else:
                    all_atr_series[ticker] = pd.Series(np.nan, index=data.index, name=ticker)

            except KeyError as e:
                all_atr_series[ticker] = pd.Series(np.nan, index=data.index, name=ticker)
            except Exception as e:
                all_atr_series[ticker] = pd.Series(np.nan, index=data.index, name=ticker)


        if not all_atr_series:
            return pd.DataFrame(index=data.index)

        atr_df = pd.concat(all_atr_series, axis=1)
        return atr_df

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        return super().__eq__(other)