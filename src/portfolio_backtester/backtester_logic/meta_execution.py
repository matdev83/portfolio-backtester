"""Meta strategy execution labeling (non-canonical vs share/cash simulation)."""

from __future__ import annotations

from enum import Enum
from typing import Any


class MetaExecutionMode(Enum):
    """High-level portfolio return path.

    Ordinary strategies use the canonical share/cash Numba simulator. Meta strategies
    aggregate child-strategy trades and reconstruct returns from that ledger; they
    do not route through ``simulate_portfolio`` / ``canonical_portfolio_simulation_kernel``.

    ``TRADE_AGGREGATION`` is an intentional alternate execution model (see
    ``docs/simulation_execution_paths.md``): same product surface, different return
    and cost reconstruction until/unless unified with the canonical engine.
    """

    CANONICAL_SHARE_CASH_SIMULATION = "canonical_share_cash_simulation"
    TRADE_AGGREGATION = "trade_aggregation"


def portfolio_execution_mode_for_strategy(
    strategy: Any,
    *,
    strategy_resolver: Any = None,
) -> MetaExecutionMode:
    """Return the execution mode for portfolio returns given a strategy instance."""
    if strategy is None:
        return MetaExecutionMode.CANONICAL_SHARE_CASH_SIMULATION
    resolver = strategy_resolver
    if resolver is None:
        from ..interfaces.strategy_resolver import StrategyResolverFactory

        resolver = StrategyResolverFactory.create()
    if resolver.is_meta_strategy(type(strategy)):
        return MetaExecutionMode.TRADE_AGGREGATION
    return MetaExecutionMode.CANONICAL_SHARE_CASH_SIMULATION
