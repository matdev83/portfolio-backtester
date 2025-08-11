from __future__ import annotations

from typing import Dict, Any, cast

import pandas as pd

from .base_momentum_portfolio_strategy import BaseMomentumPortfolioStrategy


class LowVolatilityFactorPortfolioStrategy(BaseMomentumPortfolioStrategy):
    """Low-volatility factor portfolio.

    Ranks assets by inverse volatility (lower volatility preferred) over a window
    and selects the top cohort per standard momentum template.
    """

    def __init__(self, strategy_config: Dict[str, Any]):
        super().__init__(strategy_config)
        params = self.strategy_config.get("strategy_params", {})
        params.setdefault("vol_lookback_days", 63)  # ~ 3 months of trading days

    @classmethod
    def tunable_parameters(_cls) -> Dict[str, Dict[str, Any]]:
        return {
            "vol_lookback_days": {"type": "int", "min": 21, "max": 252, "default": 63},
        }

    def _calculate_scores(
        self, asset_prices: pd.DataFrame, current_date: pd.Timestamp
    ) -> pd.Series:
        params = self.strategy_config.get("strategy_params", self.strategy_config)
        window = int(params.get("vol_lookback_days", 63))

        # Use data up to current date
        px = asset_prices[asset_prices.index <= current_date]
        if px.empty:
            return pd.Series(dtype=float, index=asset_prices.columns)

        rets = px.pct_change(fill_method=None)
        vol = rets.rolling(window).std().iloc[-1]
        if isinstance(vol, pd.DataFrame):
            vol = cast(pd.Series, vol.squeeze())

        # Score = inverse volatility (higher score is better)
        scores = (1.0 / vol.replace(0.0, pd.NA)).fillna(0.0)
        scores = scores.reindex(asset_prices.columns)
        return scores.fillna(0.0)


__all__ = ["LowVolatilityFactorPortfolioStrategy"]
