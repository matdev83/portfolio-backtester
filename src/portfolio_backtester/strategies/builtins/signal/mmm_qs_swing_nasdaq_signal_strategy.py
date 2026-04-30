"""Pine port: MMM QS Swing (Nasdaq) — daily long-only with ADX/IBS/HLc3 filter and ATR brackets.

EMA filter from the original Pine script is intentionally omitted. Stop-loss is evaluated
before take-profit when both could hit the same daily bar (matches common broker fill
priority for longs).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Union

import numpy as np
import pandas as pd

from ..._core.base.base.signal_strategy import SignalStrategy
from ....risk_management.atr_service import calculate_atr_fast

if TYPE_CHECKING:
    from ....canonical_config import CanonicalScenarioConfig

logger = logging.getLogger(__name__)


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder / RMA smoothing (alpha = 1/period), matching TradingView-style RMA."""
    return series.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def _compute_adx_series(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    di_length: int,
    adx_smoothing: int,
) -> pd.Series:
    """ADX from DMI using two Wilder lengths (Pine ``ta.dmi(diLength, adxSmoothing)``)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up_move = high.diff().astype(float)
    down_move = (-low.diff()).astype(float)
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0),
        index=high.index,
        dtype=float,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0),
        index=high.index,
        dtype=float,
    )

    tr_s = _wilder_smooth(tr, di_length)
    plus_dm_s = _wilder_smooth(plus_dm, di_length)
    minus_dm_s = _wilder_smooth(minus_dm, di_length)

    plus_di = 100.0 * plus_dm_s / tr_s.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm_s / tr_s.replace(0.0, np.nan)
    di_sum = plus_di + minus_di
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum.replace(0.0, np.nan)
    adx = _wilder_smooth(dx, adx_smoothing)
    return adx


@dataclass
class _BracketState:
    """Mutable per-strategy-run position state for the primary symbol."""

    in_long: bool = False
    stop_price: Optional[float] = None
    take_profit_price: Optional[float] = None


class MmmQsSwingNasdaqSignalStrategy(SignalStrategy):
    """Daily long-only swing signals with calendar filters and ATR stop/limit brackets.

    Scenarios should set ``timing_config.trade_execution_timing: bar_close`` (on bar close)
    to match the Pine-style same-bar fill assumption; built-in YAMLs ship with that default.
    """

    def __init__(self, strategy_config: Union[Mapping[str, Any], "CanonicalScenarioConfig"]):
        super().__init__(strategy_config)

        from ....canonical_config import CanonicalScenarioConfig

        if isinstance(strategy_config, CanonicalScenarioConfig):
            sp = dict(strategy_config.strategy_params)
        else:
            raw = strategy_config if strategy_config is not None else {}
            sp = raw.get("strategy_params", raw)

        self.leverage: float = float(sp.get("leverage", 1.0))
        self.ibs_ratio: float = float(sp.get("ibs_ratio", 0.6))
        self.sl_atr_mult: float = float(sp.get("sl_atr_mult", 0.69))
        self.tp_atr_mult: float = float(sp.get("tp_atr_mult", 1.19))
        self.di_length: int = int(sp.get("di_length", 7))
        self.adx_smoothing: int = int(sp.get("adx_smoothing", 14))
        self.adx_threshold: float = float(sp.get("adx_threshold", 20.0))
        self.atr_length: int = int(sp.get("atr_length", 20))
        self.hlc3_avg_lookback: int = int(sp.get("hlc3_avg_lookback", 10))

        self._trade_day: Dict[int, bool] = {
            0: bool(sp.get("trade_day_mon", True)),
            1: bool(sp.get("trade_day_tue", True)),
            2: bool(sp.get("trade_day_wed", False)),
            3: bool(sp.get("trade_day_thu", False)),
            4: bool(sp.get("trade_day_fri", False)),
        }
        self._trade_month: Dict[int, bool] = {}
        for m in range(1, 13):
            key = f"trade_month_{m:02d}"
            if m in (2, 9):
                self._trade_month[m] = bool(sp.get(key, False))
            else:
                self._trade_month[m] = bool(sp.get(key, True))

        self._bracket = _BracketState()

    @classmethod
    def tunable_parameters(cls) -> Dict[str, Dict[str, Any]]:
        base: Dict[str, Dict[str, Any]] = dict(super().tunable_parameters())
        cat_bool: Dict[str, Any] = {"type": "categorical", "values": [True, False]}

        for name, default in [
            ("trade_day_mon", True),
            ("trade_day_tue", True),
            ("trade_day_wed", False),
            ("trade_day_thu", False),
            ("trade_day_fri", False),
        ]:
            base[name] = {**cat_bool, "default": default}

        month_defaults = {2: False, 9: False}
        for m in range(1, 13):
            base[f"trade_month_{m:02d}"] = {
                **cat_bool,
                "default": month_defaults.get(m, True),
            }

        base.update(
            {
                "leverage": {
                    "type": "float",
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.1,
                },
                "ibs_ratio": {
                    "type": "float",
                    "default": 0.6,
                    "min": 0.1,
                    "max": 0.95,
                    "step": 0.05,
                },
                "sl_atr_mult": {
                    "type": "float",
                    "default": 0.69,
                    "min": 0.5,
                    "max": 6.0,
                    "step": 0.02,
                },
                "tp_atr_mult": {
                    "type": "float",
                    "default": 1.19,
                    "min": 0.5,
                    "max": 8.0,
                    "step": 0.05,
                },
                "di_length": {
                    "type": "int",
                    "default": 7,
                    "min": 2,
                    "max": 30,
                    "step": 1,
                },
                "adx_smoothing": {
                    "type": "int",
                    "default": 14,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                },
                "adx_threshold": {
                    "type": "float",
                    "default": 20.0,
                    "min": 5.0,
                    "max": 60.0,
                    "step": 1.0,
                },
                "atr_length": {
                    "type": "int",
                    "default": 20,
                    "min": 5,
                    "max": 50,
                    "step": 1,
                },
                "hlc3_avg_lookback": {
                    "type": "int",
                    "default": 10,
                    "min": 3,
                    "max": 30,
                    "step": 1,
                },
            }
        )
        return base

    def get_non_universe_data_requirements(self) -> list[str]:
        return []

    def _list_tickers(self, df: pd.DataFrame) -> List[str]:
        if isinstance(df.columns, pd.MultiIndex):
            tickers = sorted(
                {str(c[0]) if isinstance(c, tuple) and len(c) > 0 else str(c) for c in df.columns}
            )
            return tickers
        return [str(c) for c in df.columns]

    def _extract_ohlc(
        self, df: pd.DataFrame, ticker: str
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        if isinstance(df.columns, pd.MultiIndex):
            high = df[(ticker, "High")].astype(float)
            low = df[(ticker, "Low")].astype(float)
            close = df[(ticker, "Close")].astype(float)
            return high, low, close
        raise ValueError(
            "MmmQsSwingNasdaqSignalStrategy requires MultiIndex OHLC columns (Ticker, Field)."
        )

    def _min_warmup_bars(self) -> int:
        return int(
            max(
                self.atr_length + 2,
                self.di_length + self.adx_smoothing + 5,
                self.hlc3_avg_lookback + 5,
                5,
            )
        )

    def _calendar_ok(self, ts: pd.Timestamp) -> bool:
        if not self._trade_month.get(int(ts.month), True):
            return False
        wd = int(ts.weekday())
        if wd > 4:
            return False
        return bool(self._trade_day.get(wd, True))

    def generate_signals(
        self,
        all_historical_data: pd.DataFrame,
        benchmark_historical_data: pd.DataFrame,
        non_universe_historical_data: Optional[pd.DataFrame] = None,
        current_date: Optional[pd.Timestamp] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if current_date is None:
            current_date = pd.Timestamp(all_historical_data.index[-1])
        current_date = pd.Timestamp(current_date)

        tickers = self._list_tickers(all_historical_data)
        if not tickers:
            return pd.DataFrame(0.0, index=[current_date], columns=[])

        result = pd.DataFrame(0.0, index=[current_date], columns=tickers)
        if not isinstance(all_historical_data.columns, pd.MultiIndex):
            logger.warning(
                "MmmQsSwingNasdaqSignalStrategy expected MultiIndex OHLC; returning zeros."
            )
            return result

        primary = sorted(tickers)[0]
        hist = all_historical_data.loc[:current_date]

        if len(hist) < self._min_warmup_bars():
            self._bracket = _BracketState()
            return result

        high, low, close = self._extract_ohlc(hist, primary)
        if current_date not in close.index or pd.isna(close.loc[current_date]):
            return result

        c = float(close.loc[current_date])
        h = float(high.loc[current_date])
        l_ = float(low.loc[current_date])
        prev_high_s = high.shift(1)
        prev_high = (
            float(prev_high_s.loc[current_date])
            if current_date in prev_high_s.index
            else float("nan")
        )

        rng = h - l_
        if rng <= 0.0 or not np.isfinite(rng):
            ibs = 0.5
        else:
            ibs = (c - l_) / rng

        hlc3 = (high + low + close) / 3.0
        lb = self.hlc3_avg_lookback
        parts = [hlc3.shift(i) for i in range(1, lb + 1)]
        hlc3_avg = pd.concat(parts, axis=1).mean(axis=1)
        hlc3_thr = (
            float(hlc3_avg.loc[current_date]) if current_date in hlc3_avg.index else float("nan")
        )

        adx_series = _compute_adx_series(high, low, close, self.di_length, self.adx_smoothing)
        adx_v = (
            float(adx_series.loc[current_date])
            if current_date in adx_series.index and pd.notna(adx_series.loc[current_date])
            else float("nan")
        )

        atr_series = calculate_atr_fast(hist, current_date, self.atr_length)
        atr_v = float(atr_series.get(primary, float("nan"))) if len(atr_series) else float("nan")

        cal_ok = self._calendar_ok(current_date)

        st = self._bracket
        if st.in_long:
            exit_bar = False
            if np.isfinite(prev_high) and c > prev_high:
                exit_bar = True
            elif st.stop_price is not None and np.isfinite(st.stop_price) and l_ <= st.stop_price:
                exit_bar = True
            elif (
                st.take_profit_price is not None
                and np.isfinite(st.take_profit_price)
                and h >= st.take_profit_price
            ):
                exit_bar = True
            if exit_bar:
                st.in_long = False
                st.stop_price = None
                st.take_profit_price = None

        if not st.in_long:
            long_cond = (
                cal_ok
                and np.isfinite(hlc3_thr)
                and c < hlc3_thr
                and ibs < self.ibs_ratio
                and np.isfinite(adx_v)
                and adx_v > self.adx_threshold
                and np.isfinite(atr_v)
                and atr_v > 0.0
            )
            if long_cond and self.get_trade_longs():
                st.in_long = True
                st.stop_price = c - atr_v * self.sl_atr_mult
                st.take_profit_price = c + atr_v * self.tp_atr_mult

        w = 0.0
        if st.in_long and self.get_trade_longs():
            w = float(self.leverage)

        result.loc[current_date, primary] = w
        return result


__all__ = ["MmmQsSwingNasdaqSignalStrategy"]
