"""
In-memory data source for testing purposes.
"""

import pandas as pd
from typing import Dict, Any, List, Optional

class MemoryDataSource:
    """A data source that uses in-memory pandas DataFrames."""

    def __init__(self, data_source_config: Dict[str, Any]):
        """
        Initializes the MemoryDataSource.

        Args:
            data_source_config: Configuration dictionary containing the dataframes.
                                Expected to have a 'data_frames' key with a dict
                                containing 'daily_data' and 'benchmark_data'.
        """
        self.data_frames = data_source_config.get("data_frames", {})
        self.daily_data = self.data_frames.get("daily_data")
        self.benchmark_data = self.data_frames.get("benchmark_data")

    def get_data(self, tickers: List[str], start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Returns the pre-loaded daily data.
        Ignores the arguments as data is already loaded.
        """
        if self.daily_data is None:
            return pd.DataFrame()
        
        # Filter by date range
        mask = (self.daily_data.index >= start_date) & (self.daily_data.index <= end_date)
        
        # Filter by tickers
        if isinstance(self.daily_data.columns, pd.MultiIndex):
            available_tickers = self.daily_data.columns.get_level_values('Ticker').unique()
            selected_tickers = [t for t in tickers if t in available_tickers]
            return self.daily_data.loc[mask, (selected_tickers, slice(None))]
        else:
            selected_tickers = [t for t in tickers if t in self.daily_data.columns]
            return self.daily_data.loc[mask, selected_tickers]


    def get_benchmark_data(self, benchmark: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Returns the pre-loaded benchmark data.
        Ignores the arguments as data is already loaded.
        """
        if self.benchmark_data is None:
            return pd.DataFrame()
            
        mask = (self.benchmark_data.index >= start_date) & (self.benchmark_data.index <= end_date)
        return self.benchmark_data.loc[mask]
