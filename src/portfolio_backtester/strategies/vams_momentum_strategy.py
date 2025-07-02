from typing import Set

from .base_strategy import BaseStrategy
from ..signal_generators import DPVAMSSignalGenerator
from ..portfolio.volatility_targeting import NoVolatilityTargeting


class VAMSMomentumStrategy(BaseStrategy):
    """Momentum strategy implementation using Volatility Adjusted Momentum Scores (VAMS)."""

    signal_generator_class = DPVAMSSignalGenerator
    volatility_targeting_class = NoVolatilityTargeting

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        return {"num_holdings", "lookback_months", "alpha", "sma_filter_window"}


