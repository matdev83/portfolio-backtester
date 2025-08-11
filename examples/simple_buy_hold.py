"""
Simple Buy and Hold Example Strategy (moved out of import path).

Copy into `src/portfolio_backtester/strategies/user/portfolio/` to enable auto-discovery.
"""

from typing import Optional, Dict, Any
import pandas as pd

from portfolio_backtester.strategies._core.base.base.portfolio_strategy import (
    PortfolioStrategy,
)


class SimpleBuyHoldStrategy(PortfolioStrategy):
    """
    Example strategy that holds all assets with equal weights.

    This is the simplest possible portfolio strategy - it allocates
    equal weights to all assets in the universe and holds them.
    """

    def __init__(self, strategy_config: dict):
        super().__init__(strategy_config)
        self.rebalance_threshold = self.strategy_params.get(
            "simple_buy_hold.rebalance_threshold", 0.05
        )

    @classmethod
    def tunable_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "rebalance_threshold": {
                "type": "float",
                "min": 0.01,
                "max": 0.20,
                "default": 0.05,
                "description": "Threshold for rebalancing when weights drift",
            }
        }

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: Optional[pd.DataFrame] = None,
        current_date: Optional[pd.Timestamp] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        **kwargs,
    ) -> pd.DataFrame:
        if current_date is None:
            current_date = all_historical_data.index[-1]

        if isinstance(all_historical_data.columns, pd.MultiIndex):
            assets = all_historical_data.columns.get_level_values(0).unique()
        else:
            assets = all_historical_data.columns

        num_assets = len(assets)
        if num_assets == 0:
            return pd.DataFrame(index=[current_date])

        equal_weight = 1.0 / num_assets
        weights = pd.Series(equal_weight, index=assets)

        return pd.DataFrame([weights], index=[current_date])

    def __str__(self) -> str:
        return f"SimpleBuyHoldStrategy(threshold={self.rebalance_threshold})"
