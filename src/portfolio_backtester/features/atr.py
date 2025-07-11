import pandas as pd
import numpy as np

# Import Numba optimizations with fallback
try:
    from ..numba_optimized import atr_fast, atr_exponential_fast, true_range_fast
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
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
        if isinstance(data.columns, pd.MultiIndex) and 'Ticker' in data.columns.names:
            assets = data.columns.get_level_values('Ticker').unique()
            atr_df = pd.DataFrame(index=data.index)
            
            for asset in assets:
                try:
                    # Extract OHLC data for this asset
                    high = data[(asset, 'High')]
                    low = data[(asset, 'Low')]
                    close = data[(asset, 'Close')]
                    
                    # Use Numba optimization if available and data is suitable
                    if (NUMBA_AVAILABLE and 
                        len(high) > 1 and 
                        not high.isna().all() and 
                        not low.isna().all() and 
                        not close.isna().all()):
                        
                        # Fast path: Use Numba-optimized calculation
                        atr_values = atr_fast(high.values, low.values, close.values, self.atr_period)
                        atr = pd.Series(atr_values, index=high.index, name=f'ATR_{asset}')
                        
                    else:
                        # Fallback path: Use pandas calculation
                        # Calculate True Range
                        tr1 = high - low
                        tr2 = abs(high - close.shift(1))
                        tr3 = abs(low - close.shift(1))
                        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                        
                        # Calculate ATR as rolling mean of True Range
                        atr = true_range.rolling(window=self.atr_period, min_periods=self.atr_period).mean()
                    
                    atr_df[asset] = atr
                    
                except KeyError:
                    # If OHLC data is not available for this asset, fill with NaN
                    atr_df[asset] = np.nan
                    
            return atr_df
        
        else:
            # Handle case where data doesn't have the expected MultiIndex structure
            # This might happen if data is just close prices - cannot calculate ATR
            raise ValueError("ATR calculation requires OHLC data with MultiIndex columns (Ticker, Field)")
    
    def __eq__(self, other):
        if not isinstance(other, ATRFeature):
            return NotImplemented
        return self.atr_period == other.atr_period
    
    def __hash__(self):
        return hash((self.__class__, self.atr_period)) 