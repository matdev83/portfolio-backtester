from __future__ import annotations

"""Momentum strategy using downside volatility for position sizing."""

from .momentum_strategy import MomentumStrategy


class MomentumDvolSizerStrategy(MomentumStrategy):
    """Momentum strategy that sizes positions by downside volatility."""

    def __init__(self, strategy_config: dict) -> None:
        cfg = dict(strategy_config)
        cfg.setdefault("position_sizer", "rolling_downside_volatility")
        cfg.setdefault("target_volatility", 1.0) # Set a default for static use
        cfg.setdefault("max_leverage", 2.0) # Set a default for max_leverage
        super().__init__(cfg)

    @classmethod
    def tunable_parameters(cls) -> set[str]:
        return MomentumStrategy.tunable_parameters().union({"sizer_dvol_window", "target_volatility", "max_leverage"})
