from abc import ABC, abstractmethod
import pandas as pd


class BaseDataSource(ABC):
    """Abstract base class for data sources."""

    @abstractmethod
    def get_data(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetches price data for the given tickers and date range."""
        pass
