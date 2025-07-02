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
        # Get the actual strategy parameters, falling back to strategy_config if 'strategy_params' is not present
        params = strategy_config.get("strategy_params", strategy_config)

        # Ensure 'lookback_months' is present in params, otherwise use a default from __init__ or handle error
        # For now, we assume it's present as per typical scenario structure.
        # If this method is called with a raw strategy_config (not a full scenario dict),
        # 'lookback_months' should be directly in it.
        lookback = params.get("lookback_months")
        if lookback is None:
            # This might happen if called with a config that's not a full scenario
            # and doesn't directly contain 'lookback_months'.
            # Consider accessing self.strategy_config if this were an instance method,
            # or ensure callers always provide it.
            # For a classmethod, the input dict is all we have.
            # Fallback to a default or raise a more specific error.
            # The __init__ sets a default, but that's not accessible here directly.
            # Let's assume 'lookback_months' must be resolvable from 'params'.
            raise KeyError("'lookback_months' not found in strategy parameters. Ensure it is defined in 'strategy_params' or directly in the passed config.")

        features: Set[Feature] = {Momentum(lookback_months=lookback)}

        sma_filter = params.get("sma_filter_window")
        if sma_filter is not None:
            features.add(BenchmarkSMA(sma_filter_window=sma_filter))
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
