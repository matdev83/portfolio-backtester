from __future__ import annotations

"""Momentum strategy using downside volatility for position sizing."""

from .momentum_strategy import MomentumStrategy
from ..portfolio.volatility_targeting import NoVolatilityTargeting


class MomentumDvolSizerStrategy(MomentumStrategy):
    """Momentum strategy that sizes positions by downside volatility."""

    volatility_targeting_class = NoVolatilityTargeting

    def __init__(self, strategy_config: dict) -> None:
        cfg = dict(strategy_config)
        cfg.setdefault("position_sizer", "rolling_downside_volatility")
        super().__init__(cfg)

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        return MomentumStrategy.tunable_parameters().union({"sizer_dvol_window"})



