from typing import Set
import numpy as np

from .base_strategy import BaseStrategy
from ..feature import SortinoRatio, BenchmarkSMA, Feature
from ..signal_generators import ranking_signal_generator


class SortinoMomentumStrategy(BaseStrategy):
    """Momentum strategy implementation using Sortino ratio for ranking."""

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        return {"num_holdings", "rolling_window", "sma_filter_window", "target_return"}

    @classmethod
    def get_required_features(cls, strategy_config: dict) -> Set[Feature]:
        """Specifies the features required by the Sortino momentum strategy."""
        features = set()
        params = strategy_config.get("strategy_params", {})
        
        if "rolling_window" in params:
            features.add(SortinoRatio(rolling_window=params["rolling_window"], target_return=params.get("target_return", 0.0)))
        
        if "sma_filter_window" in params and params["sma_filter_window"] is not None:
            features.add(BenchmarkSMA(sma_filter_window=params["sma_filter_window"]))
            
        if "optimize" in strategy_config:
            for opt_spec in strategy_config["optimize"]:
                param_name = opt_spec["parameter"]
                min_val, max_val, step = opt_spec["min_value"], opt_spec["max_value"], opt_spec.get("step", 1)
                
                if param_name == "rolling_window":
                    for val in np.arange(min_val, max_val + step, step):
                        features.add(SortinoRatio(rolling_window=int(val), target_return=params.get("target_return", 0.0)))
                elif param_name == "sma_filter_window":
                    for val in np.arange(min_val, max_val + step, step):
                        features.add(BenchmarkSMA(sma_filter_window=int(val)))

        return features

    def get_signal_generator(self):
        rolling_window = self.strategy_config.get("rolling_window", 6)
        feature_name = f"sortino_{rolling_window}m"
        return ranking_signal_generator(feature_name, dropna=False, zero_if_any_nan=False)
