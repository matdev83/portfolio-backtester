from abc import ABC, abstractmethod
from typing import Set

import numpy as np
import pandas as pd

from ..feature import Feature
from ..portfolio.position_sizer import equal_weight_sizer


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, strategy_config):
        self.strategy_config = strategy_config

    # ------------------------------------------------------------------ #
    # Abstract API
    # ------------------------------------------------------------------ #

    @abstractmethod
    def get_signal_generator(self):
        """Return a callable used to generate raw position signals."""

    def get_position_sizer(self):
        """Return the position sizing function (default: equal weight)."""
        return equal_weight_sizer

    def get_volatility_target(self) -> float | None:
        """Optional volatility target for the strategy."""
        return self.strategy_config.get("volatility_target")

    def generate_signals(self, prices: pd.DataFrame, features: dict, benchmark_data: pd.Series) -> pd.DataFrame:
        """Generate trading signals using the configured signal generator."""
        generator = self.get_signal_generator()
        return generator(self, prices, features, benchmark_data)

    @classmethod
    def get_required_features(cls, strategy_config: dict) -> Set[Feature]:
        """
        Returns a set of Feature instances required by the strategy for a given configuration.
        This method should be overridden by concrete strategy implementations.
        """
        return set()

    def get_universe(self, global_config: dict) -> list[tuple[str, float]]:
        """
        Returns the list of (ticker, weight_in_index) tuples for the strategy.
        By default, it returns the universe from the global config with a default weight of 1.0.
        Concrete strategies can override this to provide a custom universe and weights.
        """
        default_universe = global_config.get("universe", [])
        return [(ticker, 1.0) for ticker in default_universe]

    # ------------------------------------------------------------------ #
    # Optimiser-introspection hook                                       #
    # ------------------------------------------------------------------ #
    @classmethod
    def tunable_parameters(cls) -> set[str]:
        """Names of hyper-parameters this strategy understands."""
        return set()

    # ------------------------------------------------------------------ #
    # Shared helpers                                                     #
    # ------------------------------------------------------------------ #

    def _calculate_candidate_weights(self, look: pd.Series) -> pd.Series:
        """Default implementation for ranking-based strategies."""
        num_holdings = self.strategy_config.get("num_holdings")
        if num_holdings is not None and num_holdings > 0:
            nh = int(num_holdings)
        else:
            frac = self.strategy_config.get("top_decile_fraction", 0.1)
            nh = max(int(np.ceil(frac * look.count())), 1)

        winners = look.nlargest(nh).index
        losers = look.nsmallest(nh).index

        cand = pd.Series(index=look.index, dtype=float).fillna(0.0)
        if len(winners) > 0:
            cand[winners] = 1 / len(winners)
        if not self.strategy_config.get("long_only", True) and len(losers) > 0:
            cand[losers] = -1 / len(losers)
        return cand

    def _apply_leverage_and_smoothing(self, cand: pd.Series, w_prev: pd.Series) -> pd.Series:
        """Apply leverage scaling and path-dependent smoothing."""
        leverage = self.strategy_config.get("leverage", 1.0)
        smoothing_lambda = self.strategy_config.get("smoothing_lambda", 0.5)

        w_new = smoothing_lambda * w_prev + (1 - smoothing_lambda) * cand

        if cand.abs().sum() > 1e-9:
            long_leverage = w_new[w_new > 0].sum()
            short_leverage = -w_new[w_new < 0].sum()

            if long_leverage > leverage:
                w_new[w_new > 0] *= leverage / long_leverage

            if short_leverage > leverage:
                w_new[w_new < 0] *= leverage / short_leverage

        return w_new
