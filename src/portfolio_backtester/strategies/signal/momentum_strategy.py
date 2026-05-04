"""
Dummy momentum_strategy for testing purposes.
"""

from __future__ import annotations

import pandas as pd

from portfolio_backtester.strategies._core.base.base_strategy import BaseStrategy
from portfolio_backtester.strategies._core.target_generation import StrategyContext


class MomentumStrategy(BaseStrategy):
    """Non-production stub strategy used for testing.

    Full-period authoring API: :py:meth:`generate_target_weights`.
    """

    def __init__(self, params):
        super().__init__(params)

    def get_universe(self, global_config):
        return [("AAPL", 1.0), ("GOOGL", 1.0)]

    def generate_target_weights(self, context: StrategyContext) -> pd.DataFrame:
        """Return an inactive target grid matching legacy empty signal semantics."""
        cols = list(context.universe_tickers)
        idx = pd.DatetimeIndex(context.rebalance_dates)
        fill_value = float("nan") if context.use_sparse_nan_for_inactive_rows else 0.0
        return pd.DataFrame(fill_value, index=idx, columns=cols, dtype=float)
