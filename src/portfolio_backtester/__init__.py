# This file makes the directory a Python package.

from .universe_data.spy_holdings import get_top_weight_sp500_components
from .strategies._core.base.base.base_strategy import BaseStrategy  # re-export
import importlib
import sys

sys.modules["src.portfolio_backtester.base.strategy"] = importlib.import_module(
    "src.portfolio_backtester.strategies._core.base.base.base_strategy"
)

__all__ = ["get_top_weight_sp500_components", "BaseStrategy"]
