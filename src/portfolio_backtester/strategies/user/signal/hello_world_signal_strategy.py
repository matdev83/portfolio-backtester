from __future__ import annotations

from typing import Dict, Any, List, Optional, Union, Mapping, TYPE_CHECKING

import pandas as pd

from ..._core.base.base.signal_strategy import SignalStrategy

if TYPE_CHECKING:
    from portfolio_backtester.canonical_config import CanonicalScenarioConfig


class HelloWorldSignalStrategy(SignalStrategy):
    """Minimal user example strategy.
    ...
        - Intended to satisfy config validation and provide a simple runnable example.
    """

    def __init__(self, strategy_config: Union[Mapping[str, Any], "CanonicalScenarioConfig"]):
        super().__init__(strategy_config)

        params = strategy_config if strategy_config is not None else {}
        sp = params.get("strategy_params", params)
        self.leverage: float = float(sp.get("leverage", 1.0))

    @classmethod
    def tunable_parameters(_cls) -> Dict[str, Dict[str, Any]]:
        # Keep intentionally tiny for quick runs
        return {
            "leverage": {
                "type": "float",
                "default": 1.0,
                "min": 0.5,
                "max": 2.0,
                "step": 0.5,
            }
        }

    def get_non_universe_data_requirements(self) -> list[str]:
        return []

    def _extract_close_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            # Try to extract Close level if present
            for lvl in range(df.columns.nlevels):
                if "Close" in df.columns.get_level_values(lvl):
                    try:
                        close_obj = df.xs("Close", axis=1, level=lvl)
                        return (
                            close_obj
                            if isinstance(close_obj, pd.DataFrame)
                            else close_obj.to_frame()
                        )
                    except Exception:
                        continue
        return df

    def generate_signal_matrix(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: Optional[pd.DataFrame],
        rebalance_dates: pd.DatetimeIndex,
        universe_tickers: List[str],
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        use_sparse_nan_for_inactive_rows: bool = False,
    ) -> Optional[pd.DataFrame]:
        cols = list(universe_tickers)
        idx = pd.DatetimeIndex(rebalance_dates)
        fill_value = float("nan") if use_sparse_nan_for_inactive_rows else 0.0
        result = pd.DataFrame(fill_value, index=idx, columns=cols, dtype=float)
        if len(idx) == 0:
            return result

        close_prices = self._extract_close_frame(all_historical_data)
        present = [c for c in cols if c in close_prices.columns]
        if len(present) == 0:
            return result.fillna(0.0) if not use_sparse_nan_for_inactive_rows else result

        weight = self.leverage / float(len(present))
        result.loc[:, present] = float(weight)
        return result

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: Optional[pd.DataFrame] = None,
        current_date: Optional[pd.Timestamp] = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        if current_date is None:
            current_date = pd.Timestamp(all_historical_data.index[-1])

        close_prices = self._extract_close_frame(all_historical_data)
        tickers = list(close_prices.columns)
        result = pd.DataFrame(0.0, index=[current_date], columns=tickers)
        if len(tickers) > 0:
            weight = self.leverage / float(len(tickers))
            result.loc[current_date, tickers] = weight
        return result


__all__ = ["HelloWorldSignalStrategy"]
