from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Set, Callable

import numpy as np
import pandas as pd

from ..feature import Feature, BenchmarkSMA
from ..portfolio.position_sizer import get_position_sizer
from ..portfolio.volatility_targeting import (
    BaseVolatilityTargeting,
    NoVolatilityTargeting,
)
from ..signal_generators import BaseSignalGenerator


class BaseStrategy(ABC):
    """Base class for trading strategies using pluggable signal generators."""

    #: class attribute specifying which signal generator to use
    signal_generator_class: type[BaseSignalGenerator] | None = None
    #: class attribute specifying which volatility targeting implementation to use
    volatility_targeting_class: type[BaseVolatilityTargeting] = NoVolatilityTargeting

    def __init__(self, strategy_config: dict):
        self.strategy_config = strategy_config

    # ------------------------------------------------------------------ #
    # Hooks to override in subclasses
    # ------------------------------------------------------------------ #

    def get_signal_generator(self) -> BaseSignalGenerator:
        if self.signal_generator_class is None:
            raise NotImplementedError("signal_generator_class must be set")
        return self.signal_generator_class(self.strategy_config)

    def get_position_sizer(self) -> Callable[[pd.DataFrame], pd.DataFrame]:
        name = self.strategy_config.get("position_sizer", "equal_weight")
        return get_position_sizer(name)

    def get_volatility_target(self) -> float | None:
        return self.strategy_config.get("volatility_target")

    def get_volatility_targeting(self) -> BaseVolatilityTargeting:
        target = self.get_volatility_target()
        return self.volatility_targeting_class(target)

    # ------------------------------------------------------------------ #
    # Required features
    # ------------------------------------------------------------------ #
    @classmethod
    def get_required_features(cls, strategy_config: dict) -> Set[Feature]:
        generator_cls = getattr(cls, "signal_generator_class", None)
        features: Set[Feature] = set()
        if generator_cls is not None:
            generator = generator_cls(strategy_config)
            features.update(generator.required_features())

        params = strategy_config.get("strategy_params", strategy_config)
        sma_window = params.get("sma_filter_window")
        if sma_window is not None:
            features.add(BenchmarkSMA(sma_filter_window=sma_window))

        return features

    # ------------------------------------------------------------------ #
    # Universe helper
    # ------------------------------------------------------------------ #
    def get_universe(self, global_config: dict) -> list[tuple[str, float]]:
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
    # Shared helpers
    # ------------------------------------------------------------------ #
    def _calculate_candidate_weights(self, look: pd.Series) -> pd.Series:
        num_holdings = self.strategy_config.get("num_holdings")
        if num_holdings is not None and num_holdings > 0:
            nh = int(num_holdings)
        else:
            nh = max(
                int(
                    np.ceil(
                        self.strategy_config.get("top_decile_fraction", 0.1)
                        * look.count()
                    )
                ),
                1,
            )

        winners = look.nlargest(nh).index
        losers = look.nsmallest(nh).index

        cand = pd.Series(index=look.index, dtype=float).fillna(0.0)
        if len(winners) > 0:
            cand[winners] = 1 / len(winners)
        if not self.strategy_config.get("long_only", True) and len(losers) > 0:
            cand[losers] = -1 / len(losers)
        return cand

    def _apply_leverage_and_smoothing(
        self, cand: pd.Series, w_prev: pd.Series
    ) -> pd.Series:
        leverage = self.strategy_config.get("leverage", 1.0)
        smoothing_lambda = self.strategy_config.get("smoothing_lambda", 0.5)

        w_new = smoothing_lambda * w_prev + (1 - smoothing_lambda) * cand

        if cand.abs().sum() > 1e-9:
            long_lev = w_new[w_new > 0].sum()
            short_lev = -w_new[w_new < 0].sum()

            if long_lev > leverage:
                w_new[w_new > 0] *= leverage / long_lev
            if short_lev > leverage:
                w_new[w_new < 0] *= leverage / short_lev

        return w_new

    # ------------------------------------------------------------------ #
    # Default signal generation pipeline
    # ------------------------------------------------------------------ #
    def generate_signals(
        self,
        prices: pd.DataFrame,
        features: dict,
        benchmark_data: pd.Series,
    ) -> pd.DataFrame:
        generator = self.get_signal_generator()
        scores = generator.scores(features)

        weights = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
        w_prev = pd.Series(index=prices.columns, dtype=float).fillna(0.0)

        sma_window = self.strategy_config.get("sma_filter_window")
        derisk_days = self.strategy_config.get("derisk_days_under_sma", 10)
        use_derisk = sma_window and derisk_days > 0

        if sma_window is not None:
            sma_name = f"benchmark_sma_{sma_window}m"
            risk_on_series = features[sma_name].reindex(prices.index, fill_value=1)
        else:
            risk_on_series = pd.Series(1, index=prices.index)

        if use_derisk:
            under_sma = 0
            derisk_flags = pd.Series(False, index=prices.index)
            for date in prices.index:
                if risk_on_series.loc[date]:
                    under_sma = 0
                else:
                    under_sma += 1
                    if under_sma > derisk_days:
                        derisk_flags.loc[date] = True
        else:
            derisk_flags = pd.Series(False, index=prices.index)

        for date in prices.index:
            look = scores.loc[date]

            if generator.zero_if_nan and look.isna().any():
                weights.loc[date] = 0.0
                w_prev = weights.loc[date]
                continue

            if look.count() == 0:
                weights.loc[date] = w_prev
                continue

            look = look.dropna()

            cand = self._calculate_candidate_weights(look)
            w_new = self._apply_leverage_and_smoothing(cand, w_prev)

            if use_derisk and derisk_flags.loc[date]:
                w_new[:] = 0.0

            weights.loc[date] = w_new
            w_prev = w_new

        if sma_window is not None:
            weights.loc[~risk_on_series.astype(bool)] = 0.0

        return weights
