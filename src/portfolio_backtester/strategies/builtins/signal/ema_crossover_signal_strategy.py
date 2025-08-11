from __future__ import annotations

from typing import Dict, Any, Optional

import pandas as pd

from ..._core.base.base.signal_strategy import SignalStrategy


class EmaCrossoverSignalStrategy(SignalStrategy):
    """Simple EMA crossover signal strategy (builtins implementation).

    Provides tunable parameters: fast_ema_days, slow_ema_days, leverage.
    """

    def __init__(self, strategy_config: dict):
        super().__init__(strategy_config)
        params = strategy_config if strategy_config is not None else {}
        # Some callers nest under strategy_params; support both
        sp = params.get("strategy_params", params)
        self.fast_ema_days: int = int(sp.get("fast_ema_days", 20))
        self.slow_ema_days: int = int(sp.get("slow_ema_days", 64))
        self.leverage: float = float(sp.get("leverage", 1.0))

    @classmethod
    def tunable_parameters(_cls) -> Dict[str, Dict[str, Any]]:
        return {
            "fast_ema_days": {"type": "int", "default": 20, "min": 10, "max": 200, "step": 5},
            "slow_ema_days": {"type": "int", "default": 64, "min": 20, "max": 300, "step": 10},
            "leverage": {"type": "float", "default": 1.0, "min": 1.0, "max": 1.0, "step": 0.1},
        }

    def get_non_universe_data_requirements(self) -> list[str]:
        return []

    def _extract_close_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            # Try each level to find 'Close'
            for lvl in range(df.columns.nlevels):
                if "Close" in df.columns.get_level_values(lvl):
                    try:
                        close_obj = df.xs("Close", axis=1, level=lvl)
                        if isinstance(close_obj, pd.Series):
                            close = close_obj.to_frame()
                        else:
                            close = close_obj
                        close.columns = close.columns.astype(str)
                        return close
                    except Exception:
                        continue
            # Fallback: if cannot find 'Close', reduce last level by first column
            reduced = df.copy()
            reduced.columns = [
                str(c[0]) if isinstance(c, tuple) and len(c) > 0 else str(c) for c in df.columns
            ]
            return reduced
        # Single-level columns; assume already prices
        return df

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

        close_prices = self._extract_close_frame(all_historical_data).astype(float)
        min_periods = max(self.fast_ema_days, self.slow_ema_days)
        valid_mask = close_prices.notna().sum() >= min_periods
        valid_cols = close_prices.columns[valid_mask]
        fast = close_prices[valid_cols].ewm(span=self.fast_ema_days).mean()
        slow = close_prices[valid_cols].ewm(span=self.slow_ema_days).mean()

        result = pd.DataFrame(0.0, index=[current_date], columns=close_prices.columns)
        if current_date in fast.index and current_date in slow.index:
            comp = fast.loc[current_date] > slow.loc[current_date]
            signal_series = comp.reindex(close_prices.columns, fill_value=False)
            if bool(signal_series.any()):
                selected_mask = signal_series.astype(bool)
                selected = [c for c, v in selected_mask.items() if v]
                num = int(signal_series.sum())
                weight = (self.leverage / num) if num > 0 else 0.0
                if selected:
                    result.loc[current_date, selected] = weight

        return result


__all__ = ["EmaCrossoverSignalStrategy"]
