from __future__ import annotations

from typing import Dict, Any, List

import pandas as pd

from portfolio_backtester.strategies._core.base.base.signal_strategy import (
    SignalStrategy,
)


class HelloWorldSignalStrategy(SignalStrategy):
    """
    Minimal user scaffold strategy (example).

    Generates equal-weight long exposure across the resolved universe on each signal date.
    This serves as a starting point for user-defined strategies. Copy this file into
    `src/portfolio_backtester/strategies/user/signal/` to enable auto-discovery.
    """

    @classmethod
    def tunable_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "leverage": {
                "type": "float",
                "min": 0.5,
                "max": 2.0,
                "default": 1.0,
                "description": "Scaling applied to equal-weight signals",
            }
        }

    def get_minimum_required_periods(self) -> int:
        return 1

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: pd.DataFrame,
        current_date: pd.Timestamp,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        # Determine asset list
        if isinstance(all_historical_data.columns, pd.MultiIndex):
            tickers: List[str] = list(
                all_historical_data.columns.get_level_values("Ticker").unique()
            )
        else:
            tickers = [str(c) for c in list(all_historical_data.columns)]

        if len(tickers) == 0:
            return pd.DataFrame(index=[current_date])

        leverage = float(self.strategy_params.get("leverage", 1.0))
        weight = leverage / float(len(tickers))
        row = {t: weight for t in tickers}
        df = pd.DataFrame([row], index=[current_date])
        return self._enforce_trade_direction_constraints(df)
