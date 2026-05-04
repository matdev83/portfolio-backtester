"""Runtime protocols for orchestration-layer strategy dependency injection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Protocol, Union

from portfolio_backtester.strategies._core.base.base.base_strategy import BaseStrategy

if TYPE_CHECKING:
    from portfolio_backtester.canonical_config import CanonicalScenarioConfig


class SupportsStrategyCreation(Protocol):
    """Minimal surface for creating strategies (production: :class:`StrategyManager`)."""

    def get_strategy(
        self,
        strategy_spec: Any,
        params: Union[Dict[str, Any], "CanonicalScenarioConfig"],
    ) -> BaseStrategy:
        """Instantiate a strategy from a name/dict spec and parameters or canonical config."""

    def get_strategy_class(self, strategy_name: str) -> type:
        """Resolve the strategy class by canonical name."""
