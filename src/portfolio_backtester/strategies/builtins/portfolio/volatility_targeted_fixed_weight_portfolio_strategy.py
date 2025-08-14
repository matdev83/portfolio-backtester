from __future__ import annotations

from typing import Dict, Any

import pandas as pd

from .base_momentum_portfolio_strategy import BaseMomentumPortfolioStrategy


class VolatilityTargetedFixedWeightPortfolioStrategy(BaseMomentumPortfolioStrategy):
    """Fixed-weight strategy gated by volatility target (placeholder minimal version).

    Applies equal weights when realized volatility is below a simple threshold; otherwise holds cash.
    Thresholding logic is intentionally minimal and can be extended by users.
    """

    def __init__(self, strategy_config: Dict[str, Any]):
        super().__init__(strategy_config)
        params = self.strategy_config.get("strategy_params", {})
        params.setdefault("target_vol_annual", 0.2)
        params.setdefault("vol_lookback_days", 63)

    @classmethod
    def tunable_parameters(_cls) -> Dict[str, Dict[str, Any]]:
        return {
            "target_vol_annual": {
                "type": "float",
                "min": 0.05,
                "max": 0.5,
                "default": 0.2,
            },
            "vol_lookback_days": {"type": "int", "min": 21, "max": 252, "default": 63},
        }

    def _calculate_scores(
        self, asset_prices: pd.DataFrame, current_date: pd.Timestamp
    ) -> pd.Series:
        # Equal scores to let base class create equal weights; risk gating happens in generate_signals
        px = asset_prices[asset_prices.index <= current_date]
        return pd.Series(1.0, index=px.columns).reindex(asset_prices.columns).fillna(0.0)
