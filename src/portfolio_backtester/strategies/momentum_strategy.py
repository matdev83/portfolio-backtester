from typing import Set

from .base_strategy import BaseStrategy
from ..signal_generators import MomentumSignalGenerator
from ..portfolio.volatility_targeting import NoVolatilityTargeting


class MomentumStrategy(BaseStrategy):
    """Momentum strategy implementation."""

    signal_generator_class = MomentumSignalGenerator
    volatility_targeting_class = NoVolatilityTargeting

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        return {
            "lookback_months", "num_holdings", "top_decile_fraction",
            "smoothing_lambda", "leverage", "long_only", "sma_filter_window",
            "derisk_days_under_sma",
        }


    def __init__(self, strategy_config):
        # Ensure all tunable parameters are present in the config with sensible defaults
        defaults = {
            "lookback_months": 6,
            "num_holdings": None,
            "top_decile_fraction": 0.1,
            "smoothing_lambda": 0.5,
            "leverage": 1.0,
            "long_only": True,
            "sma_filter_window": None,
            "derisk_days_under_sma": 10,
        }
        for k, v in defaults.items():
            strategy_config.setdefault(k, v)
        self.strategy_config = strategy_config
        super().__init__(strategy_config)

