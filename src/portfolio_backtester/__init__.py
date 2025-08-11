"""Package initializer for portfolio_backtester.

Exports select public APIs and avoids legacy alias side effects.
"""

from .universe_data.spy_holdings import get_top_weight_sp500_components
from .strategies._core.base.base.base_strategy import BaseStrategy  # re-export

__all__ = ["get_top_weight_sp500_components", "BaseStrategy"]
