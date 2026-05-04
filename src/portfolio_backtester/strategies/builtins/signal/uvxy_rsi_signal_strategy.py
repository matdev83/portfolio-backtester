from __future__ import annotations

from typing import Dict, List, Optional, Any, Union, Mapping, TYPE_CHECKING

import numpy as np
import pandas as pd

from ..._core.base.base.signal_strategy import SignalStrategy
from ..._core.target_generation import StrategyContext, default_benchmark_ticker

if TYPE_CHECKING:
    from portfolio_backtester.canonical_config import CanonicalScenarioConfig


class UvxyRsiSignalStrategy(SignalStrategy):
    """UVXY strategy using SPY RSI(2) signal. Simplified for tests.

    Full-period authoring API: :py:meth:`generate_target_weights`.
    ...
        - Tunable params: rsi_period, rsi_threshold
    """

    def __init__(self, strategy_config: Union[Mapping[str, Any], "CanonicalScenarioConfig"]):
        super().__init__(strategy_config)

        params = strategy_config.get("strategy_params", {}) if strategy_config else {}
        self.rsi_period: int = int(params.get("rsi_period", 2))
        self.rsi_threshold: float = float(params.get("rsi_threshold", 30.0))

    @classmethod
    def tunable_parameters(_cls) -> Dict[str, Dict[str, object]]:  # used by tests
        return {
            "rsi_period": {"type": "int", "default": 2, "min": 2, "max": 10},
            "rsi_threshold": {
                "type": "float",
                "default": 30.0,
                "min": 1.0,
                "max": 99.0,
            },
        }

    def get_non_universe_data_requirements(self) -> list[str]:  # used by tests
        return ["SPY"]

    def _compute_rsi2(self, prices: pd.Series) -> pd.Series:
        returns = prices.diff()
        gain = returns.clip(lower=0.0)
        loss = -returns.clip(upper=0.0)
        avg_gain = gain.rolling(self.rsi_period).mean()
        avg_loss = loss.rolling(self.rsi_period).mean()
        rs = (avg_gain / (avg_loss.replace(0, pd.NA))).fillna(0.0)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_target_weights(self, context: StrategyContext) -> pd.DataFrame:
        cols = list(context.universe_tickers)
        idx = pd.DatetimeIndex(context.rebalance_dates)
        result = pd.DataFrame(0.0, index=idx, columns=cols, dtype=float)
        if len(idx) == 0 or len(cols) == 0:
            return result

        nu = context.non_universe_data
        if len(nu.columns) == 0 or "SPY" not in nu.columns:
            return result

        spy_close = nu["SPY"].astype(float)
        rsi = self._compute_rsi2(spy_close)
        aligned = rsi.reindex(idx)
        active = aligned.lt(self.rsi_threshold).fillna(False).to_numpy(dtype=bool)
        ew = 1.0 / float(len(cols))
        dense = np.where(active[:, np.newaxis], ew, 0.0)
        result.iloc[:, :] = dense.astype(float)
        return result

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
        ctx = StrategyContext.from_standard_inputs(
            asset_data=all_historical_data,
            benchmark_data=benchmark_historical_data,
            non_universe_data=non_universe_historical_data,
            rebalance_dates=rebalance_dates,
            universe_tickers=universe_tickers,
            benchmark_ticker=default_benchmark_ticker(benchmark_historical_data, universe_tickers),
            wfo_start_date=start_date,
            wfo_end_date=end_date,
            use_sparse_nan_for_inactive_rows=use_sparse_nan_for_inactive_rows,
        )
        return self.generate_target_weights(ctx)

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

        tickers = list(all_historical_data.columns)
        result = pd.DataFrame(0.0, index=[current_date], columns=tickers)

        if (
            non_universe_historical_data is None
            or "SPY" not in non_universe_historical_data.columns
        ):
            return result

        spy_close = non_universe_historical_data["SPY"].astype(float)
        rsi = self._compute_rsi2(spy_close)

        if current_date in rsi.index and rsi.loc[current_date] < self.rsi_threshold:
            if len(tickers) > 0:
                equal_weight = 1.0 / len(tickers)
                result.loc[current_date, :] = equal_weight

        return result


__all__ = ["UvxyRsiSignalStrategy"]
