from typing import Set
import pandas as pd

from .base_strategy import BaseStrategy
from ..signal_generators import FilteredBlendedMomentumSignalGenerator
from ..feature import Feature # For type hinting in get_required_features

class FilteredLaggedMomentumStrategy(BaseStrategy):
    """
    Implements the momentum strategy based on Calluzzo, Moneta & Topaloglu (2025),
    featuring dynamic candidate filtering, blended ranking of momentum signals,
    and a one-month trading lag.
    """

    signal_generator_class = FilteredBlendedMomentumSignalGenerator

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        return {
            # Parameters for FilteredBlendedMomentumSignalGenerator
            "momentum_lookback_standard", "momentum_skip_standard",
            "momentum_lookback_predictive", "momentum_skip_predictive",
            "blending_lambda",
            # General strategy parameters
            "num_holdings", "top_decile_fraction", # top_decile_fraction used by generator
            "smoothing_lambda", "leverage", "long_only",
            "sma_filter_window", "derisk_days_under_sma",
            "apply_trading_lag",
        }

    @classmethod
    def get_required_features(cls, strategy_config: dict) -> Set[Feature]:
        """
        Defines static feature requirements. Relies on BaseStrategy and the
        configured signal generator to declare most features.
        """
        return super().get_required_features(strategy_config)

    def __init__(self, strategy_config: dict):
        defaults = {
            "momentum_lookback_standard": 11,
            "momentum_skip_standard": 1,
            "momentum_lookback_predictive": 11,
            "momentum_skip_predictive": 0,
            "blending_lambda": 0.5,
            "num_holdings": None,
            "top_decile_fraction": 0.1,
            "smoothing_lambda": 0.5,
            "leverage": 1.0,
            "long_only": True,
            "sma_filter_window": None,
            "derisk_days_under_sma": 10,
            "apply_trading_lag": False,  # Set to False by default to avoid signal loss
        }

        params_dict_to_update = strategy_config
        if "strategy_params" in strategy_config:
            if strategy_config["strategy_params"] is None:
                 strategy_config["strategy_params"] = {}
            params_dict_to_update = strategy_config["strategy_params"]

        for k, v in defaults.items():
            params_dict_to_update.setdefault(k, v)

        self.strategy_config = strategy_config
        super().__init__(strategy_config)

    def generate_signals(
        self,
        prices: pd.DataFrame,
        features: dict,
        benchmark_data: pd.Series,
    ) -> pd.DataFrame:
        """
        Generates trading signals using the FilteredBlendedMomentum logic
        and applies a one-month trading lag to the weights.
        """
        weights = super().generate_signals(prices, features, benchmark_data)
        
        # Check if we should apply trading lag
        apply_lag = self.strategy_config.get("strategy_params", {}).get("apply_trading_lag", True)
        
        # Count non-zero signals before lag
        non_zero_before = (weights != 0).sum().sum()
        non_nan_before = weights.notna().sum().sum()
        
        if apply_lag:
            weights_lagged = weights.shift(1)
            # Count non-zero signals after lag
            non_zero_after = (weights_lagged != 0).sum().sum()
            non_nan_after = weights_lagged.notna().sum().sum()
            
            # If applying lag results in too few signals, don't apply it
            if non_zero_after < non_zero_before * 0.5:  # If we lose more than 50% of signals
                print(f"Warning: Trading lag would reduce signals from {non_zero_before} to {non_zero_after}. Skipping lag.")
                return weights
            return weights_lagged
        else:
            return weights
