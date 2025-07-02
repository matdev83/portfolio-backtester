from typing import Set

from .base_strategy import BaseStrategy
from ..signal_generators import VAMSSignalGenerator


class VAMSNoDownsideStrategy(BaseStrategy):
    """Momentum strategy implementation using Volatility Adjusted Momentum Scores (VAMS), without downside volatility penalization."""

    signal_generator_class = VAMSSignalGenerator

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        return {"num_holdings", "lookback_months", "top_decile_fraction", "smoothing_lambda", "leverage", "sma_filter_window"}

    

