from typing import Set
import numpy as np

from .base_strategy import BaseStrategy
from ..feature import DPVAMS, BenchmarkSMA, Feature
from ..signal_generators import ranking_signal_generator


class VAMSMomentumStrategy(BaseStrategy):
    """Momentum strategy implementation using Volatility Adjusted Momentum Scores (VAMS)."""

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        return {"num_holdings", "lookback_months", "alpha", "sma_filter_window"}

    @classmethod
    def get_required_features(cls, strategy_config: dict) -> Set[Feature]:
        """Specifies the features required by the VAMS momentum strategy."""
        features = set()
        params = strategy_config.get("strategy_params", {})
        
        if "lookback_months" in params and "alpha" in params:
            alpha_val = params["alpha"]
            features.add(DPVAMS(lookback_months=params["lookback_months"], alpha=f"{alpha_val:.2f}"))
        
        if "sma_filter_window" in params and params["sma_filter_window"] is not None:
            features.add(BenchmarkSMA(sma_filter_window=params["sma_filter_window"]))
            
        if "optimize" in strategy_config:
            for opt_spec in strategy_config["optimize"]:
                param_name = opt_spec["parameter"]
                min_val = opt_spec["min_value"]
                max_val = opt_spec["max_value"]
                step = opt_spec.get("step", 1)
                
                if param_name == "lookback_months":
                    for val in np.arange(min_val, max_val + step, step):
                        alpha_default_val = params.get("alpha", 0.5)
                        features.add(DPVAMS(lookback_months=int(val), alpha=f"{alpha_default_val:.2f}"))
                elif param_name == "alpha":
                    for val in np.arange(min_val, max_val + step, step):
                        features.add(DPVAMS(lookback_months=params.get("lookback_months", 6), alpha=f"{val:.2f}"))
                elif param_name == "sma_filter_window":
                    for val in np.arange(min_val, max_val + step, step):
                        features.add(BenchmarkSMA(sma_filter_window=int(val)))

        return features

    def get_signal_generator(self):
        lookback_months = self.strategy_config.get("lookback_months", 6)
        alpha_val = self.strategy_config.get("alpha", 0.5)
        feature_name = f"dp_vams_{lookback_months}m_{alpha_val:.2f}a"
        return ranking_signal_generator(feature_name, dropna=False, zero_if_any_nan=False)
