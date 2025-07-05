from typing import Set
import pandas as pd # Added

from .base_strategy import BaseStrategy
from ..signal_generators import CalmarSignalGenerator


class CalmarMomentumStrategy(BaseStrategy):
    """Momentum strategy implementation using Calmar ratio for ranking."""

    signal_generator_class = CalmarSignalGenerator

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        return {"num_holdings", "rolling_window", "sma_filter_window", "apply_trading_lag"} # Added

    def __init__(self, strategy_config: dict):
        # Ensure 'apply_trading_lag' and other relevant params have defaults
        # Assuming other defaults like 'rolling_window' are handled by BaseStrategy
        # or expected to be in strategy_config.
        # For CalmarSignalGenerator, 'rolling_window' is primary.
        # BaseStrategy handles 'num_holdings', 'top_decile_fraction', 'smoothing_lambda', etc.

        params_dict_to_update = strategy_config
        if "strategy_params" in strategy_config:
            if strategy_config["strategy_params"] is None:
                 strategy_config["strategy_params"] = {}
            params_dict_to_update = strategy_config["strategy_params"]

        params_dict_to_update.setdefault("apply_trading_lag", False)
        # Other CalmarMomentumStrategy specific defaults could be added here if any.
        # For example, if 'rolling_window' for Calmar needs a default here:
        params_dict_to_update.setdefault("rolling_window", 6) # Default if not provided

        self.strategy_config = strategy_config
        super().__init__(strategy_config)

    def generate_signals(
        self,
        prices: pd.DataFrame, # Added import pandas as pd for this
        features: dict,
        benchmark_data: pd.Series, # Added import pandas as pd for this
    ) -> pd.DataFrame:
        """
        Generates trading signals. Applies a one-month trading lag if configured.
        """
        weights = super().generate_signals(prices, features, benchmark_data)

        if self.strategy_config.get("apply_trading_lag", False):
            return weights.shift(1)
        return weights

