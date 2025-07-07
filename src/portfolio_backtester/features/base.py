from abc import ABC, abstractmethod
import pandas as pd
from typing import Any, Optional

class Feature(ABC):
    """Abstract base class for a feature required by a strategy."""
    needs_close_prices_only: Optional[bool] = None

    def __init__(self, **kwargs):
        self.params = kwargs

    @abstractmethod
    def compute(self, data: pd.DataFrame, benchmark_data: pd.Series | None = None) -> pd.DataFrame | pd.Series:
        """Computes the feature."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of the feature."""
        pass

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.params == other.params

    def __hash__(self):
        return hash((self.__class__, frozenset(self.params.items())))