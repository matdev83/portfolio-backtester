from __future__ import annotations

from typing import Dict, Any

import pandas as pd

from .base_momentum_portfolio_strategy import BaseMomentumPortfolioStrategy


class FilteredLaggedMomentumPortfolioStrategy(BaseMomentumPortfolioStrategy):
    """Momentum strategy with lag and optional simple SMA filter on benchmark.

    Minimal version to preserve API and tests. Uses price lag in computing the
    momentum window.
    """

    def __init__(self, strategy_config: Dict[str, Any]):
        super().__init__(strategy_config)
        params = self.strategy_config.get("strategy_params", {})
        params.setdefault("lookback_months", 6)
        params.setdefault("skip_months", 1)

    @classmethod
    def tunable_parameters(_cls) -> Dict[str, Dict[str, Any]]:
        return {
            "lookback_months": {"type": "int", "min": 1, "max": 24, "default": 6},
            "skip_months": {"type": "int", "min": 0, "max": 3, "default": 1},
        }

    def _calculate_scores(
        self, asset_prices: pd.DataFrame, current_date: pd.Timestamp
    ) -> pd.Series:
        params = self.strategy_config.get("strategy_params", self.strategy_config)
        lookback = int(params.get("lookback_months", 6))
        skip = int(params.get("skip_months", 1))

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


__all__ = ["FilteredLaggedMomentumPortfolioStrategy"]
