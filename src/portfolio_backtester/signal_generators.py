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
