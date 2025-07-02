from typing import Set

from .base_strategy import BaseStrategy
from ..signal_generators import SharpeSignalGenerator
from ..portfolio.volatility_targeting import NoVolatilityTargeting


class SharpeMomentumStrategy(BaseStrategy):
    """Momentum strategy implementation using Sharpe ratio for ranking."""

    signal_generator_class = SharpeSignalGenerator
    volatility_targeting_class = NoVolatilityTargeting

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        return {"num_holdings", "rolling_window", "sma_filter_window"}


