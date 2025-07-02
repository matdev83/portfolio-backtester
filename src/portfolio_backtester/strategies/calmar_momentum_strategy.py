from typing import Set

from .base_strategy import BaseStrategy
from ..signal_generators import CalmarSignalGenerator


class CalmarMomentumStrategy(BaseStrategy):
    """Momentum strategy implementation using Calmar ratio for ranking."""

    signal_generator_class = CalmarSignalGenerator

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        return {"num_holdings", "rolling_window", "sma_filter_window"}

    

