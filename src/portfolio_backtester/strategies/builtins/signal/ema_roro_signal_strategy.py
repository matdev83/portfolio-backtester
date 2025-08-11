from __future__ import annotations

from typing import Dict, Any

import pandas as pd

from .ema_crossover_signal_strategy import EmaCrossoverSignalStrategy


class EmaRoroSignalStrategy(EmaCrossoverSignalStrategy):
    """EMA crossover with RoRo-style leverage modulation (simplified).

    Exposes tunable_parameters including risk_off_leverage_multiplier as tests expect.
    """

    def __init__(self, strategy_config: dict):
        super().__init__(strategy_config)
        self.base_leverage = self.leverage
        params = strategy_config.get("strategy_params", {}) if strategy_config else {}
        self.risk_off_leverage_multiplier: float = float(
            params.get("risk_off_leverage_multiplier", 0.5)
        )

    @classmethod
    def tunable_parameters(_cls) -> Dict[str, Dict[str, Any]]:
        base = dict(EmaCrossoverSignalStrategy.tunable_parameters())
        base.update(
            {
                "risk_off_leverage_multiplier": {
                    "type": "float",
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                }
            }
        )
        return base

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: pd.DataFrame | None = None,
        current_date: pd.Timestamp | None = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        original = self.leverage
        try:
            if current_date is None:
                current_date = pd.Timestamp(all_historical_data.index[-1])
            if (current_date.day % 2) == 0:
                self.leverage = self.base_leverage * self.risk_off_leverage_multiplier
            else:
                self.leverage = self.base_leverage
            return super().generate_signals(
                all_historical_data,
                benchmark_historical_data,
                non_universe_historical_data,
                current_date,
                *args,
                **kwargs,
            )
        finally:
            self.leverage = original


__all__ = ["EmaRoroSignalStrategy"]
