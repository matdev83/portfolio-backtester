from typing import Set

from .base_strategy import BaseStrategy
from ..signal_generators import MomentumSignalGenerator
from ..feature import Momentum, BenchmarkSMA, Feature


class MomentumStrategy(BaseStrategy):
    """Momentum strategy implementation."""

    signal_generator_class = MomentumSignalGenerator

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        return {
            "lookback_months", "num_holdings", "top_decile_fraction",
            "smoothing_lambda", "leverage", "long_only", "sma_filter_window",
            "derisk_days_under_sma",
        }

    @classmethod
    def get_required_features(cls, strategy_config: dict) -> Set[Feature]:
        params = strategy_config.get("strategy_params", {})
        features: Set[Feature] = {Momentum(lookback_months=params["lookback_months"])}
        if "sma_filter_window" in params and params["sma_filter_window"] is not None:
            features.add(BenchmarkSMA(sma_filter_window=params["sma_filter_window"]))
        return features

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
