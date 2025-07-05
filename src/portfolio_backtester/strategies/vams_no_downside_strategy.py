from typing import Set
import pandas as pd # Added

from .base_strategy import BaseStrategy
from ..signal_generators import VAMSSignalGenerator


class VAMSNoDownsideStrategy(BaseStrategy):
    """Momentum strategy implementation using Volatility Adjusted Momentum Scores (VAMS), without downside volatility penalization."""

    signal_generator_class = VAMSSignalGenerator

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        return {"num_holdings", "lookback_months", "top_decile_fraction", "smoothing_lambda", "leverage", "sma_filter_window", "apply_trading_lag"} # Added

    def __init__(self, strategy_config: dict):
        params_dict_to_update = strategy_config
        if "strategy_params" in strategy_config:
            if strategy_config["strategy_params"] is None:
                 strategy_config["strategy_params"] = {}
            params_dict_to_update = strategy_config["strategy_params"]

        params_dict_to_update.setdefault("apply_trading_lag", False)
        params_dict_to_update.setdefault("lookback_months", 6) # Default for VAMS

        self.strategy_config = strategy_config
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

    

