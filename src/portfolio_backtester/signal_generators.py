from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Set

import numpy as np
import pandas as pd

from .features.base import Feature
from .features.momentum import Momentum
from .features.sharpe_ratio import SharpeRatio
from .features.sortino_ratio import SortinoRatio
from .features.calmar_ratio import CalmarRatio
from .features.vams import VAMS
from .features.dp_vams import DPVAMS


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
    zero_if_nan = True  # Generate signals with available assets, zero out NaN assets
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
                    alpha=params.get("alpha", 0.5),
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
                                alpha=params.get("alpha", 0.5),
                            )
                        )
                elif spec["parameter"] == "alpha":
                    min_v = spec["min_value"]
                    max_v = spec["max_value"]
                    # Use appropriate default step for alpha (float parameter)
                    # Alpha typically ranges 0-1, so default step should be smaller
                    step = spec.get("step", 0.1 if isinstance(min_v, float) or isinstance(max_v, float) else 1)
                    for val in np.arange(min_v, max_v + step, step):
                        features.add(
                            DPVAMS(
                                lookback_months=params.get("lookback_months", 6),
                                alpha=float(val),
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
        
        # Default values
        default_std_lookback = params.get("momentum_lookback_standard", 11)
        default_std_skip = params.get("momentum_skip_standard", 1)
        default_pred_lookback = params.get("momentum_lookback_predictive", 11)
        default_pred_skip = params.get("momentum_skip_predictive", 0)

        # Add features for default parameters
        features.add(Momentum(lookback_months=default_std_lookback, skip_months=default_std_skip, name_suffix=self.std_mom_suffix))
        features.add(Momentum(lookback_months=default_pred_lookback, skip_months=default_pred_skip, name_suffix=self.pred_mom_suffix))

        # Add features for optimization parameters
        if "optimize" in self.config:
            # Extract all optimization ranges for momentum parameters
            std_lookbacks = [default_std_lookback]
            std_skips = [default_std_skip]
            pred_lookbacks = [default_pred_lookback]
            pred_skips = [default_pred_skip]

            for spec in self.config["optimize"]:
                param_name = spec["parameter"]
                if "min_value" in spec and "max_value" in spec:
                    min_v, max_v, step = spec["min_value"], spec["max_value"], spec.get("step", 1)
                    values = list(np.arange(min_v, max_v + step, step))
                else:
                    continue # Skip if not a range-based parameter

                if param_name == "momentum_lookback_standard":
                    std_lookbacks.extend(map(int, values))
                elif param_name == "momentum_skip_standard":
                    std_skips.extend(map(int, values))
                elif param_name == "momentum_lookback_predictive":
                    pred_lookbacks.extend(map(int, values))
                elif param_name == "momentum_skip_predictive":
                    pred_skips.extend(map(int, values))

            # Generate all combinations of features
            for std_lookback in set(std_lookbacks):
                for std_skip in set(std_skips):
                    features.add(Momentum(lookback_months=std_lookback, skip_months=std_skip, name_suffix=self.std_mom_suffix))
            
            for pred_lookback in set(pred_lookbacks):
                for pred_skip in set(pred_skips):
                    features.add(Momentum(lookback_months=pred_lookback, skip_months=pred_skip, name_suffix=self.pred_mom_suffix))

        return features

    def scores(self, features: dict) -> pd.DataFrame:
        params = self._params()

        std_lookback = params.get("momentum_lookback_standard", 11)
        std_skip = params.get("momentum_skip_standard", 1)
        pred_lookback = params.get("momentum_lookback_predictive", 11)
        pred_skip = params.get("momentum_skip_predictive", 0)

        blending_lambda = params.get("blending_lambda", 0.5)
        top_decile_fraction = params.get("top_decile_fraction", 0.1)

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
            
            # Make the filtering more flexible for smaller universes
            # Use a minimum of 2 assets per decile, and scale with universe size
            min_decile_size = 2
            n_decile = max(min_decile_size, int(np.floor(num_assets * top_decile_fraction)))
            
            # For very small universes, be even more flexible
            if num_assets <= 10:
                n_decile = max(1, int(np.ceil(num_assets * top_decile_fraction)))

            current_winners_idx = std_mom_values.nlargest(n_decile).index
            current_losers_idx = std_mom_values.nsmallest(n_decile).index

            predicted_winners_next_month_idx = pred_mom_values.nlargest(n_decile).index
            predicted_losers_next_month_idx = pred_mom_values.nsmallest(n_decile).index

            # Original strict filtering: only assets that are both current and predicted winners/losers
            surviving_winners = current_winners_idx.intersection(predicted_winners_next_month_idx)
            surviving_losers = current_losers_idx.intersection(predicted_losers_next_month_idx)
            strict_survivors = surviving_winners.union(surviving_losers)

            # If strict filtering produces too few survivors, use relaxed filtering
            min_survivors = max(2, int(num_assets * 0.1))  # At least 10% of universe or 2 assets
            
            if len(strict_survivors) < min_survivors:
                # Relaxed filtering: include assets that are winners/losers in either standard OR predictive
                all_winners = current_winners_idx.union(predicted_winners_next_month_idx)
                all_losers = current_losers_idx.union(predicted_losers_next_month_idx)
                
                # Take top performers from the combined set
                combined_candidates = all_winners.union(all_losers)
                
                # If still not enough, just take the top and bottom performers from standard momentum
                if len(combined_candidates) < min_survivors:
                    # Take top half and bottom half of assets based on standard momentum
                    top_half_size = max(1, num_assets // 2)
                    bottom_half_size = max(1, num_assets - top_half_size)
                    
                    top_performers = std_mom_values.nlargest(top_half_size).index
                    bottom_performers = std_mom_values.nsmallest(bottom_half_size).index
                    survivors_idx = top_performers.union(bottom_performers)
                else:
                    survivors_idx = combined_candidates
            else:
                survivors_idx = strict_survivors

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
