from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd
import numpy as np


class BaseVolatilityTargeting(ABC):
    """Interface for portfolio-level volatility targeting."""

    def __init__(self, target_vol: float | None = None):
        self.target_vol = target_vol

    @abstractmethod
    def adjust_returns(self, returns: pd.Series) -> pd.Series:
        """Adjust a return series according to the implemented rule."""
        raise NotImplementedError


class NoVolatilityTargeting(BaseVolatilityTargeting):
    """Dummy implementation that leaves returns unchanged."""

    def adjust_returns(self, returns: pd.Series) -> pd.Series:
        return returns


@dataclass
class AnnualizedVolatilityTargeting(BaseVolatilityTargeting):
    """Scale returns to achieve a constant annualized volatility."""

    target_vol: float | None = None
    window: int = 20
    ann_factor: int = 252
    max_leverage: float = 3.0

    def adjust_returns(self, returns: pd.Series) -> pd.Series:
        if self.target_vol is None:
            return returns
        rolling_vol = returns.rolling(self.window).std() * np.sqrt(self.ann_factor)
        scaling = (self.target_vol / rolling_vol).shift(1)
        scaling = scaling.clip(upper=self.max_leverage).fillna(1.0)
        return returns * scaling


__all__ = [
    "BaseVolatilityTargeting",
    "NoVolatilityTargeting",
    "AnnualizedVolatilityTargeting",
]
