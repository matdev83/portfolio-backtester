from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class VolatilityTargetingMethod(ABC):
    """Interface for portfolio-level volatility targeting."""

    @abstractmethod
    def adjust_returns(self, returns: pd.Series) -> pd.Series:
        """Scale portfolio returns according to the targeting rule."""
        raise NotImplementedError


class NoVolatilityTargeting(VolatilityTargetingMethod):
    """Dummy implementation that performs no volatility targeting."""

    def adjust_returns(self, returns: pd.Series) -> pd.Series:  # noqa: D401
        """Return the input unchanged."""
        return returns


class AnnualizedVolatilityTargeting(VolatilityTargetingMethod):
    """Target a constant annualized volatility by scaling returns."""

    def __init__(self, target_vol: float, window: int = 63, max_leverage: float = 3.0) -> None:
        self.target_vol = float(target_vol)
        self.window = int(window)
        self.max_leverage = float(max_leverage)

    def adjust_returns(self, returns: pd.Series) -> pd.Series:
        if returns.empty:
            return returns
        # Realized volatility based on trailing window, shifted to avoid look-ahead
        realized_vol = returns.rolling(self.window).std().shift(1) * np.sqrt(252)
        scaling = self.target_vol / realized_vol
        scaling = scaling.clip(upper=self.max_leverage).fillna(1.0)
        return returns * scaling

