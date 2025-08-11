from __future__ import annotations

# Re-export base types from new core path to preserve import surface
from ._core.base import (
    BaseStrategy,
    PortfolioStrategy,
    SignalStrategy,
    BaseMetaStrategy,
)

from ._core import strategy_factory as strategy_factory  # Re-export without legacy alias
from ._core import registry as registry
from ._core.base import base as base

__all__ = [
    "BaseStrategy",
    "PortfolioStrategy",
    "SignalStrategy",
    "BaseMetaStrategy",
]
