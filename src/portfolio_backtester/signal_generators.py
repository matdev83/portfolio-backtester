from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Set

import numpy as np
import pandas as pd

from .feature import (
    Feature,
    Momentum,
    SharpeRatio,
    SortinoRatio,
    CalmarRatio,
    VAMS,
    DPVAMS,
)


class BaseSignalGenerator(ABC):
    """Abstract signal generator returning ranking scores."""

    zero_if_nan: bool = False

    def __init__(self, config: dict):
        self.config = config

    def _params(self) -> dict:
        return self.config.get("strategy_params", self.config)

    @abstractmethod
    def required_features(self) -> Set[Feature]:
        """Return features needed for this signal generator."""

    @abstractmethod
    def scores(self, features: dict) -> pd.DataFrame:
        """Return a DataFrame of ranking scores."""


class MomentumSignalGenerator(BaseSignalGenerator):
    def required_features(self) -> Set[Feature]:
        features: Set[Feature] = set()
        params = self._params()
        if "lookback_months" in params:
            features.add(Momentum(lookback_months=params["lookback_months"]))

        if "optimize" in self.config:
            for spec in self.config["optimize"]:
                if spec["parameter"] == "lookback_months":
                    min_v = spec["min_value"]
                    max_v = spec["max_value"]
                    step = spec.get("step", 1)
                    for val in np.arange(min_v, max_v + step, step):
                        features.add(Momentum(lookback_months=int(val)))
        return features

    def scores(self, features: dict) -> pd.DataFrame:
        params = self._params()
        look = params.get("lookback_months", 6)
        return features[f"momentum_{look}m"]


class SharpeSignalGenerator(BaseSignalGenerator):
    def required_features(self) -> Set[Feature]:
        features: Set[Feature] = set()
        params = self._params()
        if "rolling_window" in params:
            features.add(SharpeRatio(rolling_window=params["rolling_window"]))

        if "optimize" in self.config:
            for spec in self.config["optimize"]:
                if spec["parameter"] == "rolling_window":
                    min_v = spec["min_value"]
                    max_v = spec["max_value"]
                    step = spec.get("step", 1)
                    for val in np.arange(min_v, max_v + step, step):
                        features.add(SharpeRatio(rolling_window=int(val)))
        return features

    def scores(self, features: dict) -> pd.DataFrame:
        params = self._params()
        window = params.get("rolling_window", 6)
        return features[f"sharpe_{window}m"]


class SortinoSignalGenerator(BaseSignalGenerator):
    def required_features(self) -> Set[Feature]:
        features: Set[Feature] = set()
        params = self._params()
        if "rolling_window" in params:
            features.add(
                SortinoRatio(
                    rolling_window=params["rolling_window"],
                    target_return=params.get("target_return", 0.0),
                )
            )
        if "optimize" in self.config:
            for spec in self.config["optimize"]:
                if spec["parameter"] == "rolling_window":
                    min_v = spec["min_value"]
                    max_v = spec["max_value"]
                    step = spec.get("step", 1)
                    for val in np.arange(min_v, max_v + step, step):
                        features.add(
                            SortinoRatio(
                                rolling_window=int(val),
                                target_return=params.get("target_return", 0.0),
                            )
                        )
        return features

    def scores(self, features: dict) -> pd.DataFrame:
        params = self._params()
        window = params.get("rolling_window", 6)
        return features[f"sortino_{window}m"]


class CalmarSignalGenerator(BaseSignalGenerator):
    zero_if_nan = True

    def required_features(self) -> Set[Feature]:
        features: Set[Feature] = set()
        params = self._params()
        if "rolling_window" in params:
            features.add(CalmarRatio(rolling_window=params["rolling_window"]))

        if "optimize" in self.config:
            for spec in self.config["optimize"]:
                if spec["parameter"] == "rolling_window":
                    min_v = spec["min_value"]
                    max_v = spec["max_value"]
                    step = spec.get("step", 1)
                    for val in np.arange(min_v, max_v + step, step):
                        features.add(CalmarRatio(rolling_window=int(val)))
        return features

    def scores(self, features: dict) -> pd.DataFrame:
        params = self._params()
        window = params.get("rolling_window", 6)
        return features[f"calmar_{window}m"]


class VAMSSignalGenerator(BaseSignalGenerator):
    zero_if_nan = True

    def required_features(self) -> Set[Feature]:
        features: Set[Feature] = set()
        params = self._params()
        if "lookback_months" in params:
            features.add(VAMS(lookback_months=params["lookback_months"]))

        if "optimize" in self.config:
            for spec in self.config["optimize"]:
                if spec["parameter"] == "lookback_months":
                    min_v = spec["min_value"]
                    max_v = spec["max_value"]
                    step = spec.get("step", 1)
                    for val in np.arange(min_v, max_v + step, step):
                        features.add(VAMS(lookback_months=int(val)))
        return features

    def scores(self, features: dict) -> pd.DataFrame:
        params = self._params()
        lb = params.get("lookback_months", 6)
        return features[f"vams_{lb}m"]


class DPVAMSSignalGenerator(BaseSignalGenerator):
    def required_features(self) -> Set[Feature]:
        features: Set[Feature] = set()
        params = self._params()
        if "lookback_months" in params and "alpha" in params:
            features.add(
                DPVAMS(
                    lookback_months=params["lookback_months"],
                    alpha=f"{params.get('alpha', 0.5):.2f}",
                )
            )

        if "optimize" in self.config:
            for spec in self.config["optimize"]:
                if spec["parameter"] == "lookback_months":
                    min_v = spec["min_value"]
                    max_v = spec["max_value"]
                    step = spec.get("step", 1)
                    for val in np.arange(min_v, max_v + step, step):
                        features.add(
                            DPVAMS(
                                lookback_months=int(val),
                                alpha=f"{params.get('alpha', 0.5):.2f}",
                            )
                        )
                elif spec["parameter"] == "alpha":
                    min_v = spec["min_value"]
                    max_v = spec["max_value"]
                    step = spec.get("step", 1)
                    for val in np.arange(min_v, max_v + step, step):
                        features.add(
                            DPVAMS(
                                lookback_months=params.get("lookback_months", 6),
                                alpha=f"{val:.2f}",
                            )
                        )
        return features

    def scores(self, features: dict) -> pd.DataFrame:
        params = self._params()
        lb = params.get("lookback_months", 6)
        alpha = params.get("alpha", 0.5)
        return features[f"dp_vams_{lb}m_{alpha:.2f}a"]


__all__ = [
    "BaseSignalGenerator",
    "MomentumSignalGenerator",
    "SharpeSignalGenerator",
    "SortinoSignalGenerator",
    "CalmarSignalGenerator",
    "VAMSSignalGenerator",
    "DPVAMSSignalGenerator",
]


class FilteredBlendedMomentumSignalGenerator(BaseSignalGenerator):
    """
    Signal generator for momentum strategy with dynamic candidate filtering
    and blended ranking based on Calluzzo, Moneta & Topaloglu (2025).
    """

    def __init__(self, config: dict):
        super().__init__(config)
        # Parameters are expected in config["strategy_params"]
        # e.g., momentum_lookback_standard, momentum_skip_standard
        #       momentum_lookback_predictive, momentum_skip_predictive
        #       blending_lambda, decile_fraction

        # Suffixes are stored for consistent feature retrieval
        self.std_mom_suffix = "std"
        self.pred_mom_suffix = "pred"


    def required_features(self) -> Set[Feature]:
        features: Set[Feature] = set()
        params = self._params()

        std_lookback = params.get("momentum_lookback_standard", 11)
        std_skip = params.get("momentum_skip_standard", 1)
        pred_lookback = params.get("momentum_lookback_predictive", 11)
        pred_skip = params.get("momentum_skip_predictive", 0)

        features.add(Momentum(lookback_months=std_lookback, skip_months=std_skip, name_suffix=self.std_mom_suffix))
        features.add(Momentum(lookback_months=pred_lookback, skip_months=pred_skip, name_suffix=self.pred_mom_suffix))

        return features

    def scores(self, features: dict) -> pd.DataFrame:
        params = self._params()

        std_lookback = params.get("momentum_lookback_standard", 11)
        std_skip = params.get("momentum_skip_standard", 1)
        pred_lookback = params.get("momentum_lookback_predictive", 11)
        pred_skip = params.get("momentum_skip_predictive", 0)

        blending_lambda = params.get("blending_lambda", 0.5)
        top_decile_fraction = params.get("top_decile_fraction", 0.1) # Changed from decile_fraction

        std_mom_name = f"momentum_{std_lookback}m_skip{std_skip}m_{self.std_mom_suffix}"
        pred_mom_name = f"momentum_{pred_lookback}m_skip{pred_skip}m_{self.pred_mom_suffix}"

        mom_std = features[std_mom_name]
        mom_pred = features[pred_mom_name]

        # Align dataframes by date index first
        aligned_idx = mom_std.index.intersection(mom_pred.index)
        mom_std = mom_std.loc[aligned_idx]
        mom_pred = mom_pred.loc[aligned_idx]

        final_scores = pd.DataFrame(np.nan, index=mom_std.index, columns=mom_std.columns)

        for date in mom_std.index:
            std_series_at_date = mom_std.loc[date].dropna()
            pred_series_at_date = mom_pred.loc[date].dropna()

            if std_series_at_date.empty or pred_series_at_date.empty:
                continue

            common_assets = std_series_at_date.index.intersection(pred_series_at_date.index)
            if common_assets.empty:
                continue

            std_mom_values = std_series_at_date[common_assets]
            pred_mom_values = pred_series_at_date[common_assets]

            num_assets = len(common_assets)
            n_decile = max(1, int(np.floor(num_assets * top_decile_fraction))) # Changed from decile_fraction

            current_winners_idx = std_mom_values.nlargest(n_decile).index
            current_losers_idx = std_mom_values.nsmallest(n_decile).index

            predicted_winners_next_month_idx = pred_mom_values.nlargest(n_decile).index
            predicted_losers_next_month_idx = pred_mom_values.nsmallest(n_decile).index

            surviving_winners = current_winners_idx.intersection(predicted_winners_next_month_idx)
            surviving_losers = current_losers_idx.intersection(predicted_losers_next_month_idx)

            survivors_idx = surviving_winners.union(surviving_losers)

            if survivors_idx.empty:
                continue

            std_mom_survivors = std_mom_values.loc[survivors_idx]
            pred_mom_survivors = pred_mom_values.loc[survivors_idx]

            # Percentile ranks (0 to 1), higher value = higher rank
            rank_std_survivors = std_mom_survivors.rank(pct=True)
            rank_pred_survivors = pred_mom_survivors.rank(pct=True)

            blended_rank_survivors = (blending_lambda * rank_std_survivors +
                                     (1 - blending_lambda) * rank_pred_survivors)

            final_scores.loc[date, blended_rank_survivors.index] = blended_rank_survivors

        return final_scores


__all__ = [
    "BaseSignalGenerator",
    "MomentumSignalGenerator",
    "SharpeSignalGenerator",
    "SortinoSignalGenerator",
    "CalmarSignalGenerator",
    "VAMSSignalGenerator",
    "DPVAMSSignalGenerator",
    "FilteredBlendedMomentumSignalGenerator", # Added here
]
