from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Set, Callable

import numpy as np
import pandas as pd

from ..feature import Feature, BenchmarkSMA
from ..portfolio.position_sizer import get_position_sizer
from ..signal_generators import BaseSignalGenerator
from ..roro_signals import BaseRoRoSignal # Import BaseRoRoSignal


class BaseStrategy(ABC):
    """Base class for trading strategies using pluggable signal generators."""

    #: class attribute specifying which signal generator to use
    signal_generator_class: type[BaseSignalGenerator] | None = None
    #: class attribute specifying which RoRo signal generator to use
    roro_signal_class: type[BaseRoRoSignal] | None = None

    def __init__(self, strategy_config: dict):
        self.strategy_config = strategy_config
        self._roro_signal_instance: BaseRoRoSignal | None = None

    # ------------------------------------------------------------------ #
    # Hooks to override in subclasses
    # ------------------------------------------------------------------ #

    def get_signal_generator(self) -> BaseSignalGenerator:
        if self.signal_generator_class is None:
            raise NotImplementedError("signal_generator_class must be set")
        return self.signal_generator_class(self.strategy_config)

    def get_roro_signal(self) -> BaseRoRoSignal | None:
        if self.roro_signal_class is None:
            return None
        if self._roro_signal_instance is None:
            # Assuming RoRo signal might have its own config under "roro_signal_params"
            roro_config = self.strategy_config.get("roro_signal_params", self.strategy_config)
            self._roro_signal_instance = self.roro_signal_class(roro_config)
        return self._roro_signal_instance

    def get_position_sizer(self) -> Callable[[pd.DataFrame], pd.DataFrame]:
        name = self.strategy_config.get("position_sizer", "equal_weight")
        return get_position_sizer(name)

    # ------------------------------------------------------------------ #
    # Required features
    # ------------------------------------------------------------------ #
    @classmethod
    def get_required_features(cls, strategy_config: dict) -> Set[Feature]:
        features: Set[Feature] = set()

        # Features from signal generator
        signal_gen_cls = getattr(cls, "signal_generator_class", None)
        if signal_gen_cls is not None:
            generator = signal_gen_cls(strategy_config)
            features.update(generator.required_features())

        # Features from RoRo signal generator
        roro_gen_cls = getattr(cls, "roro_signal_class", None)
        if roro_gen_cls is not None:
            # Assuming RoRo signal might have its own config under "roro_signal_params"
            roro_config = strategy_config.get("roro_signal_params", strategy_config)
            roro_generator = roro_gen_cls(roro_config)
            features.update(roro_generator.get_required_features())

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

        # --- SMA-based risk filter ---
        sma_window = self.strategy_config.get("sma_filter_window")
        derisk_days = self.strategy_config.get("derisk_days_under_sma", 10)
        use_sma_derisk = sma_window and derisk_days > 0

        if sma_window is not None:
            sma_name = f"benchmark_sma_{sma_window}m"
            # Ensure sma_risk_on_series has a boolean dtype after reindexing and filling
            sma_risk_on_series = features[sma_name].reindex(prices.index, fill_value=True).astype(bool)
        else:
            sma_risk_on_series = pd.Series(True, index=prices.index, name="sma_risk_on")

        derisk_flags = self._calculate_derisk_flags(sma_risk_on_series, derisk_days) if use_sma_derisk else pd.Series(False, index=prices.index)

        # --- RoRo signal based risk filter ---
        roro_signal_instance = self.get_roro_signal()
        if roro_signal_instance:
            # Generate RoRo signal for the prices.index (typically monthly dates)
            roro_signal_values = roro_signal_instance.generate_signal(prices.index)
            # Ensure it's boolean: 1 (risk-on) -> True, 0 (risk-off) -> False
            roro_risk_on_series = roro_signal_values.astype(bool)
        else:
            roro_risk_on_series = pd.Series(True, index=prices.index, name="roro_risk_on")


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

            # Apply SMA-based derisking flag
            if use_sma_derisk and derisk_flags.loc[date]:
                w_new[:] = 0.0

            # Apply general SMA filter (if price is below SMA, reduce risk)
            # This is applied *after* the derisk_flags logic which might be based on *consecutive* days.
            if sma_window is not None and not sma_risk_on_series.loc[date]:
                w_new[:] = 0.0

            # Apply RoRo signal: if RoRo is risk-off (False), set weights to 0
            if not roro_risk_on_series.loc[date]:
                w_new[:] = 0.0

            weights.loc[date] = w_new
            w_prev = w_new

        # The individual date logic should handle SMA and RoRo.
        # The blanket application of SMA filter after the loop might be redundant now or could be re-evaluated.
        # For now, let's rely on the per-date logic.
        # if sma_window is not None:
        #     weights.loc[~sma_risk_on_series.astype(bool)] = 0.0

        return weights

    def _calculate_derisk_flags(self, sma_risk_on_series: pd.Series, derisk_days: int) -> pd.Series:
        """
        Calculates flags indicating when to derisk based on consecutive days under SMA.

        Parameters:
        - sma_risk_on_series (pd.Series): Boolean series indicating if risk is on (True, price >= SMA)
                                          or off (False, price < SMA). Index must match prices.index.
        - derisk_days (int): Number of consecutive days asset must be under SMA to trigger derisking.

        Returns:
        - pd.Series: Boolean series with True where derisking should occur.
        """
        if not isinstance(sma_risk_on_series, pd.Series) or not isinstance(derisk_days, int) or derisk_days <= 0:
            # This case should ideally be handled by the caller (use_sma_derisk check)
            # but as a safeguard, return an all-false series.
            return pd.Series(False, index=sma_risk_on_series.index)

        derisk_flags = pd.Series(False, index=sma_risk_on_series.index)
        consecutive_days_risk_off = 0

        for date in sma_risk_on_series.index:
            if sma_risk_on_series.loc[date]:  # Price is >= SMA (Risk is ON based on SMA)
                consecutive_days_risk_off = 0
            else:  # Price is < SMA (Risk is OFF based on SMA)
                consecutive_days_risk_off += 1

            if consecutive_days_risk_off > derisk_days: # Note: strictly greater
                derisk_flags.loc[date] = True
            # else: # If it's not > derisk_days, the flag remains False (or becomes False if it was True and now sma_risk_on_series is True)
            #    if derisk_flags.loc[date]: # This logic is not needed as we reset consecutive_days_risk_off
            #        pass

        return derisk_flags
