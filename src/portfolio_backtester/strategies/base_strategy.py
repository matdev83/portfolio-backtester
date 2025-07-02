from abc import ABC, abstractmethod
from typing import Set
import pandas as pd
import numpy as np

from ..feature import Feature


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, strategy_config):
        self.strategy_config = strategy_config

    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame, features: dict, benchmark_data: pd.Series) -> pd.DataFrame:
        """Generates trading signals for the given data."""
        pass

    @classmethod
    def get_required_features(cls, strategy_config: dict) -> Set[Feature]:
        """
        Returns a set of Feature instances required by the strategy for a given configuration.
        This method should be overridden by concrete strategy implementations.
        """
        return set()

    def get_universe(self, global_config: dict) -> list[tuple[str, float]]:
        """
        Returns the list of (ticker, weight_in_index) tuples for the strategy.
        By default, it returns the universe from the global config with a default weight of 1.0.
        Concrete strategies can override this to provide a custom universe and weights.
        """
        default_universe = global_config.get("universe", [])
        return [(ticker, 1.0) for ticker in default_universe]

    # ------------------------------------------------------------------ #
    # Optimiser-introspection hook                                       #
    # ------------------------------------------------------------------ #
    @classmethod
    def tunable_parameters(cls) -> set[str]:
        """Names of hyper-parameters this strategy understands."""
        return set()
