"""Pine port: MMM QS Swing (Nasdaq) — daily long-only with ADX/IBS/HLc3 filter and ATR brackets.

EMA filter from the original Pine script is intentionally omitted. Stop-loss is evaluated
before take-profit when both could hit the same daily bar (matches common broker fill
priority for longs).

Optional ATH drawdown gates (see ``drawdown_from_ath_pct``): ``drawdown_from_ath_min_pct``
and ``drawdown_from_ath_max_dist_from_min_dd_pct``. Non-positive values disable the
corresponding side; ``dist`` is measured in percentage points above the active minimum
(or from zero when the minimum is disabled), so min/max cannot contradict.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Union

import numpy as np
import pandas as pd

from ..._core.base.base.signal_strategy import SignalStrategy
from ..._core.target_generation import StrategyContext
from ....risk_management.atr_service import calculate_atr_fast

if TYPE_CHECKING:
    from ....canonical_config import CanonicalScenarioConfig

logger = logging.getLogger(__name__)


def drawdown_from_ath_pct(high: pd.Series, close: pd.Series, as_of: pd.Timestamp) -> float:
    """Drawdown of ``close`` at ``as_of`` versus expanding all-time high of ``high`` (percent).

    Uses ``high`` and ``close`` only through ``as_of`` (inclusive). Defined as
    ``100 * (1 - close / ATH)`` when ``ATH > 0``; otherwise returns ``nan``.

    Args:
        high: High prices indexed by time.
        close: Close prices indexed by time.
        as_of: Evaluation timestamp (must exist in ``close``).

    Returns:
        Drawdown in percent, or ``nan`` if undefined.
    """
    high_to = high.loc[:as_of]
    if len(high_to) == 0 or as_of not in close.index:
        return float("nan")
    ath = float(high_to.max())
    c = float(close.loc[as_of])
    if ath <= 0.0 or not np.isfinite(ath) or not np.isfinite(c):
        return float("nan")
    return 100.0 * (1.0 - c / ath)


def ath_drawdown_entry_allowed(dd_pct: float, min_pct: float, dist_pct: float) -> bool:
    """Return whether ATH drawdown ``dd_pct`` passes entry band rules.

    Semantics (``> 0`` enables each role; ``<= 0`` disables):

    * ``min_pct > 0`` — require ``dd_pct >= min_pct``.
    * ``dist_pct > 0`` — upper edge: if ``min_pct > 0``, require ``dd_pct <= min_pct + dist_pct``;
      if ``min_pct <= 0``, require ``dd_pct <= dist_pct`` (ceiling-only).
    * If both ``min_pct <= 0`` and ``dist_pct <= 0``, the ATH drawdown filter is off.

    Args:
        dd_pct: Current drawdown in percent (``drawdown_from_ath_pct``).
        min_pct: Minimum drawdown from ATH in percent, or non-positive to disable.
        dist_pct: Span in percentage points for the upper edge (see above).

    Returns:
        ``True`` if the observation is allowed for a new entry from the ATH perspective.
    """
    if not np.isfinite(dd_pct):
        return False
    has_lo = min_pct > 0.0
    has_hi = dist_pct > 0.0
    if not has_lo and not has_hi:
        return True
    if has_lo and has_hi:
        return dd_pct >= min_pct and dd_pct <= (min_pct + dist_pct)
    if has_lo and not has_hi:
        return dd_pct >= min_pct
    return dd_pct <= dist_pct


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

    Full-period authoring API: :py:meth:`generate_target_weights`.

    Scenarios should set ``timing_config.trade_execution_timing: bar_close`` (on bar close)
    to match the Pine-style same-bar fill assumption; built-in YAMLs ship with that default.

    Optional ATH drawdown band uses ``drawdown_from_ath_min_pct`` and
    ``drawdown_from_ath_max_dist_from_min_dd_pct`` (see ``ath_drawdown_entry_allowed``).
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
        self.drawdown_from_ath_min_pct: float = float(sp.get("drawdown_from_ath_min_pct", 0.0))
        self.drawdown_from_ath_max_dist_from_min_dd_pct: float = float(
            sp.get("drawdown_from_ath_max_dist_from_min_dd_pct", 0.0)
        )

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
                "drawdown_from_ath_min_pct": {
                    "type": "float",
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.5,
                },
                "drawdown_from_ath_max_dist_from_min_dd_pct": {
                    "type": "float",
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.5,
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

    def _drawdown_from_ath_entry_ok(self, dd_pct: float) -> bool:
        """True if ATH drawdown passes optional band (see ``ath_drawdown_entry_allowed``)."""
        return ath_drawdown_entry_allowed(
            dd_pct,
            self.drawdown_from_ath_min_pct,
            self.drawdown_from_ath_max_dist_from_min_dd_pct,
        )

    def generate_target_weights(self, context: StrategyContext) -> pd.DataFrame:
        """Full-scan targets matching sequential legacy ``generate_signals`` semantics."""
        self._bracket = _BracketState()
        cols = list(context.universe_tickers)
        idx = pd.DatetimeIndex(context.rebalance_dates)
        fill_value = float("nan") if context.use_sparse_nan_for_inactive_rows else 0.0
        out = pd.DataFrame(fill_value, index=idx, columns=cols, dtype=float)
        if len(idx) == 0 or len(cols) == 0:
            return out
        nu_full = context.non_universe_data
        for d in idx:
            if d not in context.asset_data.index:
                continue
            nu_slice = nu_full.loc[:d] if len(nu_full.columns) > 0 else pd.DataFrame()
            sig = self.generate_signals(
                context.asset_data.loc[:d],
                context.benchmark_data.loc[:d],
                nu_slice,
                current_date=d,
            )
            if d not in sig.index:
                continue
            out.loc[d, :] = sig.loc[d].reindex(cols)

        return out

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

        dd_ath_pct = drawdown_from_ath_pct(high, close, current_date)
        ath_entry_ok = self._drawdown_from_ath_entry_ok(dd_ath_pct)

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
                and ath_entry_ok
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


__all__ = [
    "MmmQsSwingNasdaqSignalStrategy",
    "ath_drawdown_entry_allowed",
    "drawdown_from_ath_pct",
]
