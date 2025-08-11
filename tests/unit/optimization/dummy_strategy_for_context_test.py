import pandas as pd
from typing import Optional
from portfolio_backtester.strategies._core.base import BaseStrategy


class DummyStrategyForContextTest(BaseStrategy):
    """A simple, static strategy file for context management tests."""

    @staticmethod
    def tunable_parameters():
        return {"param1": {"type": "float", "low": 0.1, "high": 1.0}}

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: Optional[pd.DataFrame],
        current_date: pd.Timestamp,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        **kwargs,
    ) -> pd.DataFrame:
        # No-op for testing, return empty DataFrame to match signature
        return pd.DataFrame()
