
from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, strategy_config):
        self.strategy_config = strategy_config

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, benchmark_data: pd.Series) -> pd.DataFrame:
        """Generates trading signals for the given data."""
        pass
