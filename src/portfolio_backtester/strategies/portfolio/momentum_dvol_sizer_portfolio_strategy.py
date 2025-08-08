from __future__ import annotations
from typing import Any

import pandas as pd
from .base_momentum_portfolio_strategy import BaseMomentumPortfolioStrategy

"""Momentum strategy using downside volatility for position sizing."""


class MomentumDvolSizerPortfolioStrategy(BaseMomentumPortfolioStrategy):
    """Momentum strategy that sizes positions by downside volatility."""

    def __init__(self, strategy_config: dict) -> None:
        cfg = dict(strategy_config)
        cfg.setdefault("position_sizer", "rolling_downside_volatility")
        cfg.setdefault("target_volatility", 1.0)  # Set a default for static use
        cfg.setdefault("max_leverage", 2.0)  # Set a default for max_leverage
        super().__init__(cfg)

    @classmethod
    def tunable_parameters(cls) -> dict[str, dict[str, Any]]:
        # Define base momentum parameters plus downside volatility sizer parameters
        return {
            "lookback_months": {"type": "int", "min": 1, "max": 60, "default": 6},
            "skip_months": {"type": "int", "min": 0, "max": 12, "default": 0},
            "num_holdings": {"type": "int", "min": 1, "max": 50, "default": None},
            "top_decile_fraction": {"type": "float", "min": 0.01, "max": 1.0, "default": 0.1},
            "smoothing_lambda": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.5},
            "leverage": {"type": "float", "min": 0.1, "max": 5.0, "default": 1.0},
            "trade_longs": {"type": "bool", "default": True},
            "trade_shorts": {"type": "bool", "default": False},
            "sma_filter_window": {"type": "int", "min": 0, "max": 24, "default": None},
            "derisk_days_under_sma": {"type": "int", "min": 1, "max": 60, "default": 10},
            "apply_trading_lag": {"type": "bool", "default": False},
            # Downside volatility sizer specific parameters
            "sizer_dvol_window": {"type": "int", "min": 3, "max": 36, "default": 12},
            "target_volatility": {"type": "float", "min": 0.1, "max": 3.0, "default": 1.0},
            "max_leverage": {"type": "float", "min": 0.5, "max": 5.0, "default": 2.0},
        }

    def get_minimum_required_periods(self) -> int:
        """
        Calculate minimum required periods for MomentumDvolSizerPortfolioStrategy.
        Requires: max(momentum requirements, sizer_dvol_window)
        """
        params = self.strategy_config.get("strategy_params", self.strategy_config)

        # Get base momentum requirements
        base_requirement = super().get_minimum_required_periods()

        # Downside volatility sizer requirement
        sizer_dvol_window = params.get("sizer_dvol_window", 12)

        # Take the maximum of all requirements
        total_requirement = max(base_requirement, sizer_dvol_window)

        # Add 2-month buffer for reliable calculations
        return int(total_requirement + 2)

    def _calculate_scores(self, asset_prices, current_date):
        """Calculate simple momentum scores for downside volatility sizer strategy."""
        params = self.strategy_config.get("strategy_params", self.strategy_config)
        lookback_months = params.get("lookback_months", 6)
        skip_months = params.get("skip_months", 0)

        relevant_prices = asset_prices[asset_prices.index <= current_date]
        if relevant_prices.empty:
            return pd.Series(dtype=float, index=asset_prices.columns)

        date_t_minus_skip = current_date - pd.DateOffset(months=skip_months)
        date_t_minus_lookback = current_date - pd.DateOffset(months=skip_months + lookback_months)

        try:
            prices_now = relevant_prices.asof(date_t_minus_skip)
            prices_then = relevant_prices.asof(date_t_minus_lookback)
        except KeyError:
            return pd.Series(dtype=float, index=asset_prices.columns)

        if prices_now is None or prices_then is None:
            return pd.Series(dtype=float, index=asset_prices.columns)

        # Calculate simple momentum as price returns
        momentum_scores = (prices_now / prices_then) - 1
        
        if isinstance(momentum_scores, pd.DataFrame):
            momentum_scores = momentum_scores.squeeze()
        
        if isinstance(momentum_scores, pd.Series):
            return momentum_scores.fillna(0.0)
        else:
            return pd.Series(momentum_scores, index=asset_prices.columns).fillna(0.0)
