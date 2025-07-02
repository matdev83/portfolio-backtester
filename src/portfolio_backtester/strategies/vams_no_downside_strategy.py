from typing import Set
import numpy as np

from .base_strategy import BaseStrategy
from ..feature import VAMS, BenchmarkSMA, Feature
from ..signal_generators import ranking_signal_generator


class VAMSNoDownsideStrategy(BaseStrategy):
    """Momentum strategy implementation using Volatility Adjusted Momentum Scores (VAMS), without downside volatility penalization."""

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        return {"num_holdings", "lookback_months", "top_decile_fraction", "smoothing_lambda", "leverage", "sma_filter_window"}

    @classmethod
    def get_required_features(cls, strategy_config: dict) -> Set[Feature]:
        """Specifies the features required by the VAMS no downside strategy."""
        features = set()
        params = strategy_config.get("strategy_params", {})
        
        if "lookback_months" in params:
            features.add(VAMS(lookback_months=params["lookback_months"]))
        
        if "sma_filter_window" in params and params["sma_filter_window"] is not None:
            features.add(BenchmarkSMA(sma_filter_window=params["sma_filter_window"]))
            
        if "optimize" in strategy_config:
            for opt_spec in strategy_config["optimize"]:
                param_name = opt_spec["parameter"]
                min_val, max_val, step = opt_spec["min_value"], opt_spec["max_value"], opt_spec.get("step", 1)
                
                if param_name == "lookback_months":
                    for val in np.arange(min_val, max_val + step, step):
                        features.add(VAMS(lookback_months=int(val)))
                elif param_name == "sma_filter_window":
                    for val in np.arange(min_val, max_val + step, step):
                        features.add(BenchmarkSMA(sma_filter_window=int(val)))

        return features

    def get_signal_generator(self):
        lookback_months = self.strategy_config.get("lookback_months", 6)
        feature_name = f"vams_{lookback_months}m"
        return ranking_signal_generator(feature_name, dropna=False, zero_if_any_nan=True)
