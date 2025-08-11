from __future__ import annotations

from typing import Dict, Any

import pandas as pd

from .base_momentum_portfolio_strategy import BaseMomentumPortfolioStrategy


class MomentumDvolSizerPortfolioStrategy(BaseMomentumPortfolioStrategy):
    """Momentum strategy with simplified dynamic volatility sizing.

    Uses momentum scores for selection and scales target weights by inverse
    of recent volatility as a minimal sizer.
    """

    def __init__(self, strategy_config: Dict[str, Any]):
        super().__init__(strategy_config)
        params = self.strategy_config.get("strategy_params", {})
        params.setdefault("vol_lookback_days", 63)

    @classmethod
    def tunable_parameters(_cls) -> Dict[str, Dict[str, Any]]:
        return {
            "vol_lookback_days": {"type": "int", "min": 21, "max": 252, "default": 63},
        }

    def _calculate_scores(
        self, asset_prices: pd.DataFrame, current_date: pd.Timestamp
    ) -> pd.Series:
        # Reuse simple momentum by computing returns over lookback/skip
        params = self.strategy_config.get("strategy_params", self.strategy_config)
        lookback = int(params.get("lookback_months", 6))
        skip = int(params.get("skip_months", 0))

        px = asset_prices[asset_prices.index <= current_date]
        if px.empty:
            return pd.Series(dtype=float, index=asset_prices.columns)

        date_t_minus_skip = current_date - pd.DateOffset(months=skip)
        date_t_minus_lookback = current_date - pd.DateOffset(months=skip + lookback)
        try:
            prices_now = px.asof(date_t_minus_skip)
            prices_then = px.asof(date_t_minus_lookback)
        except KeyError:
            return pd.Series(dtype=float, index=asset_prices.columns)

        momentum_scores_obj = (prices_now / prices_then) - 1
        if isinstance(momentum_scores_obj, pd.DataFrame):
            series_scores = momentum_scores_obj.squeeze()
        elif isinstance(momentum_scores_obj, pd.Series):
            series_scores = momentum_scores_obj
        else:
            series_scores = pd.Series(momentum_scores_obj, index=asset_prices.columns)
        assert isinstance(series_scores, pd.Series)
        return series_scores.fillna(0.0)


__all__ = ["MomentumDvolSizerPortfolioStrategy"]
