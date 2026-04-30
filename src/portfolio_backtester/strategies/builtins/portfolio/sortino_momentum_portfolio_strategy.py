from __future__ import annotations

from typing import Dict, Any, cast, Union, Mapping, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from portfolio_backtester.canonical_config import CanonicalScenarioConfig

# Do not import SortinoRatio at module import time. Import lazily in __init__
# so unit tests can monkeypatch
from .base_momentum_portfolio_strategy import BaseMomentumPortfolioStrategy


class SortinoMomentumPortfolioStrategy(BaseMomentumPortfolioStrategy):
    """Momentum strategy implementation using Sortino ratio for ranking."""

    def __init__(self, strategy_config: Union[Mapping[str, Any], "CanonicalScenarioConfig"]):
        super().__init__(strategy_config)

        params_dict_to_update = self.strategy_config.get("strategy_params", {})
        sortino_defaults = {
            "rolling_window": 3,  # Default for Sortino
            "target_return": 0.0,  # Default target return
        }
        for k, v in sortino_defaults.items():
            params_dict_to_update.setdefault(k, v)

        # Import SortinoRatio directly from features (alpha: legacy shim removed)
        from portfolio_backtester.features.sortino_ratio import (
            SortinoRatio as _SortinoRatio,
        )

        self.sortino_feature = _SortinoRatio(
            rolling_window=params_dict_to_update["rolling_window"],
            target_return=params_dict_to_update["target_return"],
        )

    @classmethod
    def tunable_parameters(_cls) -> Dict[str, Dict[str, Any]]:
        return {
            "num_holdings": {"type": "int", "min": 1, "max": 100, "default": 10},
            "rolling_window": {"type": "int", "min": 3, "max": 24, "default": 3},
            "target_return": {"type": "float", "min": -0.1, "max": 0.1, "default": 0.0},
            "sma_filter_window": {"type": "int", "min": 0, "max": 200, "default": 0},
            "apply_trading_lag": {"type": "bool", "default": False},
            "lookback_months": {"type": "int", "min": 1, "max": 36, "default": 12},
            "top_decile_fraction": {
                "type": "float",
                "min": 0.05,
                "max": 0.5,
                "default": 0.5,
            },
            "smoothing_lambda": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 0.5,
            },
            "leverage": {"type": "float", "min": 1.0, "max": 3.0, "default": 1.0},
            "trade_longs": {"type": "bool", "default": True},
            "trade_shorts": {"type": "bool", "default": False},
        }

    def get_minimum_required_periods(self) -> int:
        """
        Calculate minimum required periods for SortinoMomentumPortfolioStrategy.
        Requires: rolling_window for Sortino ratio calculation + SMA filter window
        """
        params = self.strategy_config.get("strategy_params", self.strategy_config)

        # Sortino ratio rolling window requirement
        rolling_window = params.get("rolling_window", 3)

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
        scores = self.sortino_feature.compute(asset_prices)
        if scores.empty:
            return pd.Series(dtype=float)

        idx_dates = scores.index
        if isinstance(idx_dates, pd.DatetimeIndex) and idx_dates.tz is not None:
            current_ts = pd.Timestamp(current_date)
            if current_ts.tzinfo is None:
                current_ts = current_ts.tz_localize(idx_dates.tz)
            else:
                current_ts = current_ts.tz_convert(idx_dates.tz)
        else:
            current_ts = pd.Timestamp(current_date)

        valid_dates = idx_dates[idx_dates <= current_ts]
        if len(valid_dates) == 0:
            return pd.Series(dtype=float)

        return cast(pd.Series, scores.loc[valid_dates[-1]].squeeze())


__all__ = ["SortinoMomentumPortfolioStrategy"]
