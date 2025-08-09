from __future__ import annotations

from typing import Dict, Any, cast

from ..._core.base.base.meta_strategy import BaseMetaStrategy


class SimpleMetaStrategy(BaseMetaStrategy):
    """Simple meta strategy that allocates capital using fixed percentage weights."""

    def __init__(
        self, strategy_params: Dict[str, Any], global_config: Dict[str, Any] | None = None
    ):
        super().__init__(strategy_params, global_config=global_config or {})
        defaults = {"min_allocation": 0.05, "rebalance_threshold": 0.05}
        for key, value in defaults.items():
            self.strategy_params.setdefault(key, value)

    def allocate_capital(self) -> Dict[str, float]:
        allocations: Dict[str, float] = {}
        for allocation in self.allocations:
            allocations[allocation.strategy_id] = allocation.weight
        return allocations

    @classmethod
    def tunable_parameters(cls) -> Dict[str, Dict[str, Any]]:
        base_params = super().tunable_parameters()
        return base_params


__all__ = ["SimpleMetaStrategy"]
