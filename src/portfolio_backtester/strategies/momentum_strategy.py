from typing import Set
import pandas as pd # Keep for general pd use, though not strictly needed by this version

from .base_strategy import BaseStrategy
from ..signal_generators import MomentumSignalGenerator # Original generator
from ..feature import Momentum, BenchmarkSMA, Feature


class MomentumStrategy(BaseStrategy):
    """Momentum strategy implementation (Original Version)."""

    signal_generator_class = MomentumSignalGenerator

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        return {
            "lookback_months", "num_holdings", "top_decile_fraction",
            "smoothing_lambda", "leverage", "long_only", "sma_filter_window",
            "derisk_days_under_sma", "apply_trading_lag", # Added
        }

    @classmethod
    def get_required_features(cls, strategy_config: dict) -> Set[Feature]:
        # Get the actual strategy parameters, falling back to strategy_config if 'strategy_params' is not present
        params = strategy_config.get("strategy_params", strategy_config)

        features: Set[Feature] = set()
        lookback = params.get("lookback_months")
        if lookback is not None:
            # Uses default skip_months=0, name_suffix="".
            # Name will be "momentum_{lookback}m" due to backward compatibility in Momentum.name
            features.add(Momentum(lookback_months=lookback))

        sma_filter = params.get("sma_filter_window")
        if sma_filter is not None:
            features.add(BenchmarkSMA(sma_filter_window=sma_filter))
        return features

    def __init__(self, strategy_config: dict):
        # Original simple __init__ default handling
        defaults = {
            "lookback_months": 6,
            "num_holdings": None,
            "top_decile_fraction": 0.1,
            "smoothing_lambda": 0.5,
            "leverage": 1.0,
            "long_only": True,
            "sma_filter_window": None,
            "derisk_days_under_sma": 10,
            "apply_trading_lag": False, # Added
        }

        # Determine the correct dictionary to apply defaults to
        # This logic ensures that if strategy_config contains a "strategy_params" sub-dict,
        # defaults are applied there. Otherwise, they are applied to strategy_config itself.
        # This is important for how BaseSignalGenerator and BaseStrategy access params.
        params_dict_to_update = strategy_config
        if "strategy_params" in strategy_config:
            if strategy_config["strategy_params"] is None: # Handle case where it's explicitly None
                 strategy_config["strategy_params"] = {}
            params_dict_to_update = strategy_config["strategy_params"]

        for k, v in defaults.items():
            params_dict_to_update.setdefault(k, v)

        self.strategy_config = strategy_config # This is the full config passed to Base class
        super().__init__(strategy_config)

    def generate_signals(
        self,
        prices: pd.DataFrame,
        features: dict,
        benchmark_data: pd.Series,
    ) -> pd.DataFrame:
        """
        Generates trading signals. Applies a one-month trading lag if configured.
        """
        weights = super().generate_signals(prices, features, benchmark_data)

        if self.strategy_config.get("apply_trading_lag", False):
            return weights.shift(1)
        return weights
