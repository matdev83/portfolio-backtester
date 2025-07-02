from typing import Set
import numpy as np

from .base_strategy import BaseStrategy
from ..feature import Momentum, BenchmarkSMA, Feature
from ..signal_generators import ranking_signal_generator


class MomentumStrategy(BaseStrategy):
    """Momentum strategy implementation."""

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        return {
            "lookback_months", "num_holdings", "top_decile_fraction",
            "smoothing_lambda", "leverage", "long_only", "sma_filter_window",
            "derisk_days_under_sma",
        }

    @classmethod
    def get_required_features(cls, strategy_config: dict) -> Set[Feature]:
        """Specifies the features required by the momentum strategy."""
        features = set()
        params = strategy_config.get("strategy_params", {})
        
        if "lookback_months" in params:
            features.add(Momentum(lookback_months=params["lookback_months"]))
        
        if "sma_filter_window" in params and params["sma_filter_window"] is not None:
            features.add(BenchmarkSMA(sma_filter_window=params["sma_filter_window"]))
            
        if "optimize" in strategy_config:
            for opt_spec in strategy_config["optimize"]:
                param_name = opt_spec["parameter"]
                min_val, max_val, step = opt_spec["min_value"], opt_spec["max_value"], opt_spec.get("step", 1)
                
                if param_name == "lookback_months":
                    for val in np.arange(min_val, max_val + step, step):
                        features.add(Momentum(lookback_months=int(val)))
                elif param_name == "sma_filter_window":
                    for val in np.arange(min_val, max_val + step, step):
                        features.add(BenchmarkSMA(sma_filter_window=int(val)))

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

    def get_signal_generator(self):
        lookback_months = self.strategy_config.get("lookback_months", 6)
        feature_name = f"momentum_{lookback_months}m"
        derisk_days = self.strategy_config.get("derisk_days_under_sma", 10)
        # Only enable immediate derisk logic if a SMA window is specified
        immediate = derisk_days if self.strategy_config.get("sma_filter_window") else None
        return ranking_signal_generator(
            feature_name,
            dropna=True,
            zero_if_any_nan=False,
            immediate_derisk_days=immediate,
        )
