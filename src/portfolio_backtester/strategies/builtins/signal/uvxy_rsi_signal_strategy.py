from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from ..._core.base.base.signal_strategy import SignalStrategy


class UvxyRsiSignalStrategy(SignalStrategy):
    """UVXY strategy using SPY RSI(2) signal. Simplified for tests.

    - Universe: UVXY only (weights distributed equally across all columns if present)
    - Non-universe data requirement: ["SPY"] for RSI calculation
    - Tunable params: rsi_period, rsi_threshold
    """

    def __init__(self, strategy_config: Dict):
        super().__init__(strategy_config)
        params = strategy_config.get("strategy_params", {}) if strategy_config else {}
        self.rsi_period: int = int(params.get("rsi_period", 2))
        self.rsi_threshold: float = float(params.get("rsi_threshold", 30.0))

    @classmethod
    def tunable_parameters(cls) -> Dict[str, Dict[str, object]]:  # used by tests
        return {
            "rsi_period": {"type": "int", "default": 2, "min": 2, "max": 10},
            "rsi_threshold": {"type": "float", "default": 30.0, "min": 1.0, "max": 99.0},
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
