from typing import Set

from .base_strategy import BaseStrategy
from ..signal_generators import CalmarSignalGenerator
from ..portfolio.volatility_targeting import NoVolatilityTargeting


class CalmarMomentumStrategy(BaseStrategy):
    """Momentum strategy implementation using Calmar ratio for ranking."""

    signal_generator_class = CalmarSignalGenerator
    volatility_targeting_class = NoVolatilityTargeting

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        return {"num_holdings", "rolling_window", "sma_filter_window"}

    

