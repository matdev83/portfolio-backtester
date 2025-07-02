from typing import Set

from .base_strategy import BaseStrategy
from ..signal_generators import DPVAMSSignalGenerator


class VAMSMomentumStrategy(BaseStrategy):
    """Momentum strategy implementation using Volatility Adjusted Momentum Scores (VAMS)."""

    signal_generator_class = DPVAMSSignalGenerator

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        return {"num_holdings", "lookback_months", "alpha", "sma_filter_window"}


