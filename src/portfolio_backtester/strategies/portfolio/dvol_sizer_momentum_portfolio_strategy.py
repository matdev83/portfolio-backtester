from __future__ import annotations
from typing import Any

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
        return {
            **BaseMomentumPortfolioStrategy.tunable_parameters(),
            "sizer_dvol_window": {},
            "target_volatility": {},
            "max_leverage": {},
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
