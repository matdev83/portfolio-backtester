from __future__ import annotations

from typing import Dict, Any, cast
import pandas as pd

from .base_momentum_portfolio_strategy import BaseMomentumPortfolioStrategy


class CalmarMomentumPortfolioStrategy(BaseMomentumPortfolioStrategy):
    """Momentum strategy implementation using Calmar ratio for ranking."""

    def __init__(self, strategy_config: Dict[str, Any]):
        super().__init__(strategy_config)

        params_dict_to_update = self.strategy_config.get("strategy_params", {})
        calmar_defaults = {
            "rolling_window": 6,  # Default for Calmar
        }
        for k, v in calmar_defaults.items():
            params_dict_to_update.setdefault(k, v)

        # Import CalmarRatio at runtime so test patches on
        # portfolio_backtester.features.calmar_ratio.CalmarRatio are effective
        from portfolio_backtester.features.calmar_ratio import (
            CalmarRatio as _CalmarRatio,
        )

        self.calmar_feature = _CalmarRatio(rolling_window=params_dict_to_update["rolling_window"])

    @classmethod
    def tunable_parameters(_cls) -> Dict[str, Dict[str, Any]]:
        return {
            "num_holdings": {"type": "int", "min": 1, "max": 100, "default": 10},
            "rolling_window": {"type": "int", "min": 3, "max": 24, "default": 6},
            "sma_filter_window": {"type": "int", "min": 0, "max": 200, "default": 0},
            "apply_trading_lag": {"type": "bool", "default": False},
            "lookback_months": {"type": "int", "min": 1, "max": 36, "default": 12},
            "top_decile_fraction": {
                "type": "float",
                "min": 0.05,
                "max": 0.5,
                "default": 0.1,
            },
            "smoothing_lambda": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 1.0,
            },
            "leverage": {"type": "float", "min": 1.0, "max": 3.0, "default": 1.0},
            "trade_longs": {"type": "bool", "default": True},
            "trade_shorts": {"type": "bool", "default": False},
        }

    def get_minimum_required_periods(self) -> int:
        """
        Calculate minimum required periods for CalmarMomentumPortfolioStrategy.
        Requires: rolling_window for Calmar ratio calculation + SMA filter window
        """
        params = self.strategy_config.get("strategy_params", self.strategy_config)

        # Calmar ratio rolling window requirement
        rolling_window = params.get("rolling_window", 6)

        # SMA filter requirement (if enabled)
        sma_filter_window = params.get("sma_filter_window")
        sma_requirement = sma_filter_window if sma_filter_window and sma_filter_window > 0 else 0

        # Take the maximum of all requirements plus buffer
        total_requirement = max(rolling_window, sma_requirement)

        # Add 2-month buffer for reliable calculations
        return int(total_requirement + 2)

    def _calculate_scores(
        self,
        asset_prices: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> pd.Series:
        scores = self.calmar_feature.compute(asset_prices)
        if current_date in scores.index:
            return cast(pd.Series, scores.loc[current_date].squeeze())
        else:
            return pd.Series(dtype=float)


__all__ = ["CalmarMomentumPortfolioStrategy"]
