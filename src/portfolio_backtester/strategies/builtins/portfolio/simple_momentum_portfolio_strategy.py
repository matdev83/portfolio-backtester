from __future__ import annotations

import logging
from typing import Any, Dict, final

import pandas as pd
from .base_momentum_portfolio_strategy import BaseMomentumPortfolioStrategy

logger = logging.getLogger(__name__)


@final
class SimpleMomentumPortfolioStrategy(BaseMomentumPortfolioStrategy):
    """
    Simple momentum strategy implementation - a minimal example.

    This class is marked as @final to prevent inheritance.
    For custom momentum strategies, inherit from BaseMomentumPortfolioStrategy directly.

    Calculates momentum as simple price returns over a lookback period.
    Uses sensible defaults and minimal configuration for demonstration purposes.
    """

    def __init__(self, strategy_config: Dict[str, Any]):
        super().__init__(strategy_config)

        # Add simple momentum-specific defaults
        params_dict_to_update = self.strategy_config.get("strategy_params", {})
        momentum_defaults = {
            "lookback_months": 6,  # 6-month momentum lookback
            "skip_months": 0,  # No skip period (standard momentum)
        }
        for k, v in momentum_defaults.items():
            params_dict_to_update.setdefault(k, v)

    @classmethod
    def tunable_parameters(_cls) -> Dict[str, Dict[str, Any]]:
        """
        Define core tunable parameters for simple momentum strategy.

        Simplified parameter set focusing on the most important momentum parameters.
        """
        return {
            # Core momentum parameters
            "lookback_months": {"type": "int", "min": 1, "max": 24, "default": 6},
            "skip_months": {"type": "int", "min": 0, "max": 3, "default": 0},
            # Portfolio construction parameters
            "num_holdings": {"type": "int", "min": 1, "max": 20, "default": 10},
            "top_decile_fraction": {
                "type": "float",
                "min": 0.1,
                "max": 0.5,
                "default": 0.2,
            },
            # Risk management parameters
            "leverage": {"type": "float", "min": 0.5, "max": 2.0, "default": 1.0},
            "smoothing_lambda": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 0.5,
            },
        }

    def get_minimum_required_periods(self) -> int:
        """
        Calculate minimum required periods for simple momentum strategy.

        Simplified calculation focusing on core momentum requirements.
        """
        params = self.strategy_config.get("strategy_params", self.strategy_config)

        # Core momentum requirement: lookback + skip + buffer
        lookback_months = params.get("lookback_months", 6)
        skip_months = params.get("skip_months", 0)

        # Simple calculation with 3-month buffer for reliability
        return int(lookback_months + skip_months + 3)

    def _calculate_scores(
        self,
        asset_prices: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> pd.Series:
        """
        Calculate simple momentum scores as price returns.

        This is the core implementation that defines what "simple momentum" means:
        momentum = (price_now / price_then) - 1

        Args:
            asset_prices: DataFrame with historical prices, indexed by date, columns by asset
            current_date: The current date for score calculation

        Returns:
            Series of momentum scores (price returns) for each asset
        """
        params = self.strategy_config.get("strategy_params", self.strategy_config)
        lookback_months = params.get("lookback_months", 6)
        skip_months = params.get("skip_months", 0)

        # Filter to relevant price history
        relevant_prices = asset_prices[asset_prices.index <= current_date]
        if relevant_prices.empty:
            return pd.Series(dtype=float, index=asset_prices.columns)

        # Calculate momentum dates
        date_t_minus_skip = current_date - pd.DateOffset(months=skip_months)
        date_t_minus_lookback = current_date - pd.DateOffset(months=skip_months + lookback_months)

        try:
            # Get prices at the two time points
            prices_now = relevant_prices.asof(date_t_minus_skip)
            prices_then = relevant_prices.asof(date_t_minus_lookback)
        except KeyError:
            return pd.Series(dtype=float, index=asset_prices.columns)

        # Handle missing data
        if prices_now is None or prices_then is None:
            return pd.Series(dtype=float, index=asset_prices.columns)

        # Calculate simple momentum: (P_now / P_then) - 1
        momentum_scores = (prices_now / prices_then) - 1

        # Ensure we return a proper Series
        if isinstance(momentum_scores, pd.DataFrame):
            momentum_scores = momentum_scores.squeeze()  # type: ignore[assignment]

        if isinstance(momentum_scores, pd.Series):
            return momentum_scores.fillna(0.0)
        else:
            return pd.Series(momentum_scores, index=asset_prices.columns).fillna(0.0)


__all__ = ["SimpleMomentumPortfolioStrategy"]
