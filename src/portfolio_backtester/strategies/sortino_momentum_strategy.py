from typing import Set

from .base_strategy import BaseStrategy
from ..signal_generators import SortinoSignalGenerator
from ..portfolio.volatility_targeting import NoVolatilityTargeting


class SortinoMomentumStrategy(BaseStrategy):
    """Momentum strategy implementation using Sortino ratio for ranking."""

    signal_generator_class = SortinoSignalGenerator
    volatility_targeting_class = NoVolatilityTargeting

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        return {"num_holdings", "rolling_window", "sma_filter_window", "target_return"}


