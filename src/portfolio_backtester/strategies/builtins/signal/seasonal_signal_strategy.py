from __future__ import annotations

import calendar
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union, Mapping, TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from portfolio_backtester.risk_management.atr_service import calculate_atr_fast

from ..._core.base.base.signal_strategy import SignalStrategy
from ..._core.target_generation import StrategyContext, default_benchmark_ticker

if TYPE_CHECKING:
    from portfolio_backtester.canonical_config import CanonicalScenarioConfig

logger = logging.getLogger(__name__)

_ATR_LOOKBACK = 21
_DEFAULT_CARLOS_RORO_SYMBOL = "MDMP:RORO.CARLOS"


class SeasonalSignalStrategy(SignalStrategy):
    """Intramonth seasonality: Nth business-day entry and a fixed business-day hold window.

    Full-period authoring API: :py:meth:`generate_target_weights`.

    **Default (``month_local_seasonal_windows: false``):** For each evaluation date we walk
    backward from ``current_date``'s calendar month and take the **first** anchor whose
    seasonal window covers ``current_date`` (see :py:meth:`_resolve_active_entry_date` and
    :py:meth:`generate_signals`). Holdings may extend across calendar month boundaries, but
    overlapping candidate anchors are never union-blended—the first-scan anchor owns each
    row.

    **Legacy opt-in (``month_local_seasonal_windows: true``):** The anchor is always
    recomputed from **``current_date``'s** calendar month only (no carry from the prior
    month's entry), matching the older month-local behavior.

    **Exit rule precedence (same session, daily close):** simple high/low SL, then simple
    high/low TP, then ATR-based SL, then ATR-based TP. The first condition that fires on
    a given day locks the ticker for the remainder of that cycle.

    Strategy parameters (under ``strategy_params``); tunables listed in
    :py:meth:`tunable_parameters`:
        month_local_seasonal_windows: if ``true``, use month-local windows only;
            default ``false`` (cross-month holds).
        direction: ``"long"`` (default) or ``"short"``. Short uses negative equal weights;
            downstream constraints may still disallow shorts.
        entry_day: int in [-21, 21] (clamped to available business days in the month).
        hold_days: int >= 1; length of the in-window period in **business** days.
        entry_day_by_month: optional mapping from calendar month (1..12 and/or month
            names / abbreviations, case-insensitive) to ``entry_day`` int; months omitted
            use ``entry_day``.
        hold_days_by_month: optional mapping from calendar month to ``hold_days`` int;
            months omitted use ``hold_days``. Hold length applies to the anchor month's
            cycle (including when the window extends into the next calendar month).
        trade_month_1 .. trade_month_12: optional bools. In **cross-month** mode they
            gate the **entry anchor's** calendar month. In **month-local** mode they
            gate ``current_date``'s calendar month (unchanged from legacy).
        simple_high_low_stop_loss: if ``true``, exit when close crosses vs prior bar's
            high/low (long/short); default ``false``.
        simple_high_low_take_profit: if ``true``, take profit on the same vocabulary;
            default ``false``. Requires OHLC history (MultiIndex ``Ticker``/``Field``).
        stop_loss_atr_multiple: float >= 0; default ``0`` (disabled). When positive, exit
            when daily close breaches entry-based SL using a 21-day ATR at entry times this
            multiple (long: at or below ``entry - mult * ATR``; short: at or above
            ``entry + mult * ATR``).
        take_profit_atr_multiple: float >= 0; default ``0`` (disabled). Same ATR snapshot as
            stop loss; long TP at or above ``entry + mult * ATR``, short TP at or below
            ``entry - mult * ATR``.
        use_carlos_roro: if ``true``, load Carlos RoRo overlay from ``non_universe`` data and
            block exposure on days whose Close reads risk-off (1); default ``false``.
        carlos_roro_symbol: ticker key for overlay Close in ``non_universe_historical_data``;
            default ``MDMP:RORO.CARLOS``. Close convention: ``1`` = risk-off (flat /
            exit-only even inside a seasonal window), ``0`` = risk-on.
        max_dd_from_ath_pct: float ``>= 0``; default ``0.0`` disables. When positive and
            ``direction`` is ``long``, each ticker's allocation is gated on drawdown from
            the running ATH of its Close through the evaluation date (percent). Rows with
            drawdown strictly greater than this threshold behave as excluded from the long.

    Calendar note: entry is **Nth business day of the month**, not ``Timestamp.day``.
    Timestamps are compared on a **calendar wall-clock** basis after stripping timezone
    info so tz-aware price indices (e.g. US/Eastern) do not raise against naive anchors.
    """

    @staticmethod
    def _calendar_naive(ts: pd.Timestamp) -> pd.Timestamp:
        """Strip tz for Y/M/D and business-day window math (align with exchange calendar date)."""
        t = pd.Timestamp(ts)
        return t.tz_convert(None) if t.tzinfo is not None else t

    @staticmethod
    def _index_calendar_naive(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Naive wall-calendar timestamps parallel to ``index`` for compare with ``bdate_range`` edges.

        ``get_entry_date_for_month`` returns timezone-naive business days. When ``index`` is
        timezone-aware (e.g. US/Eastern), naive-vs-aware comparisons raise; we convert each
        label to the index timezone then drop tzinfo so ordering matches exchange-local dates.
        """
        if index.tz is None:
            return index
        return pd.DatetimeIndex(
            [pd.Timestamp(ts).tz_convert(index.tz).replace(tzinfo=None) for ts in index],
            dtype="datetime64[ns]",
        )

    def __init__(self, strategy_config: Union[Mapping[str, Any], "CanonicalScenarioConfig"]):
        super().__init__(strategy_config)

        params = strategy_config.get("strategy_params", {}) if strategy_config else {}
        self.direction: str = str(params.get("direction", "long"))
        self.entry_day: int = int(params.get("entry_day", 1))
        self.hold_days: int = int(params.get("hold_days", 3))
        self.month_local_seasonal_windows: bool = bool(
            params.get("month_local_seasonal_windows", False)
        )
        self.simple_high_low_stop_loss: bool = bool(params.get("simple_high_low_stop_loss", False))
        self.simple_high_low_take_profit: bool = bool(
            params.get("simple_high_low_take_profit", False)
        )
        self.stop_loss_atr_multiple: float = float(params.get("stop_loss_atr_multiple", 0.0))
        self.take_profit_atr_multiple: float = float(params.get("take_profit_atr_multiple", 0.0))
        if self.stop_loss_atr_multiple < 0:
            raise ValueError("stop_loss_atr_multiple must be >= 0")
        if self.take_profit_atr_multiple < 0:
            raise ValueError("take_profit_atr_multiple must be >= 0")

        # Month filters: default to True if not specified
        self.allowed_month: Dict[int, bool] = {
            m: bool(params.get(f"trade_month_{m}", True)) for m in range(1, 13)
        }

        self._entry_day_by_month_map: Dict[int, int] = self._parse_month_int_overrides(
            params.get("entry_day_by_month"), field_name="entry_day_by_month"
        )
        self._hold_days_by_month_map: Dict[int, int] = self._parse_month_int_overrides(
            params.get("hold_days_by_month"), field_name="hold_days_by_month"
        )

        self._warned_missing_ohlc_for_sl_tp: bool = False

        self.use_carlos_roro: bool = bool(params.get("use_carlos_roro", False))
        self.carlos_roro_symbol: str = str(
            params.get("carlos_roro_symbol", _DEFAULT_CARLOS_RORO_SYMBOL)
        )
        self._warned_missing_carlos_roro: bool = False

        self.max_dd_from_ath_pct: float = float(params.get("max_dd_from_ath_pct", 0.0))
        if self.max_dd_from_ath_pct < 0:
            raise ValueError("max_dd_from_ath_pct must be >= 0")

    def get_non_universe_data_requirements(self) -> List[str]:
        if self.use_carlos_roro:
            return [self.carlos_roro_symbol]
        return []

    def _warn_missing_carlos_roro_once(self) -> None:
        if self._warned_missing_carlos_roro:
            return
        self._warned_missing_carlos_roro = True
        logger.warning(
            "SeasonalSignalStrategy: use_carlos_roro enabled but Carlos RoRo Close series is "
            "missing or unreadable in non_universe_historical_data; overlay not applied."
        )

    def _extract_non_universe_close_frame(
        self, df: Optional[pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
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
            reduced = df.copy()
            reduced.columns = [
                str(c[0]) if isinstance(c, tuple) and len(c) > 0 else str(c) for c in df.columns
            ]
            return reduced
        return df

    def _carlos_roro_close_series(
        self, non_universe_historical_data: Optional[pd.DataFrame]
    ) -> Optional[pd.Series]:
        frame = self._extract_non_universe_close_frame(non_universe_historical_data)
        if frame is None or self.carlos_roro_symbol not in frame.columns:
            return None
        return pd.to_numeric(frame[self.carlos_roro_symbol], errors="coerce")

    def _carlos_value_is_risk_off(self, val: Any) -> bool:
        x = pd.to_numeric(val, errors="coerce")
        try:
            f = float(cast(Any, x))
        except (TypeError, ValueError):
            return False
        if f != f:
            return False
        return math.isclose(f, 1.0, rel_tol=0.0, abs_tol=1e-9)

    def _carlos_roro_is_risk_off_at(
        self, non_universe_historical_data: Optional[pd.DataFrame], asof: pd.Timestamp
    ) -> bool:
        if not self.use_carlos_roro:
            return False
        series = self._carlos_roro_close_series(non_universe_historical_data)
        if series is None or series.empty:
            self._warn_missing_carlos_roro_once()
            return False
        ts_naive = self._calendar_naive(pd.Timestamp(asof))
        by_day: Dict[pd.Timestamp, Any] = {}
        for lbl, raw in series.items():
            lbl_ts = pd.Timestamp(cast(Any, lbl))
            by_day[self._calendar_naive(lbl_ts)] = raw
        raw_final = by_day.get(ts_naive)
        if raw_final is None:
            return False
        return self._carlos_value_is_risk_off(raw_final)

    def _carlos_roro_risk_off_mask(
        self,
        calendar_naive_index: pd.DatetimeIndex,
        non_universe_historical_data: Optional[pd.DataFrame],
    ) -> Optional[np.ndarray]:
        if not self.use_carlos_roro:
            return None
        series = self._carlos_roro_close_series(non_universe_historical_data)
        if series is None or series.empty:
            self._warn_missing_carlos_roro_once()
            return np.zeros(len(calendar_naive_index), dtype=bool)
        by_day: Dict[pd.Timestamp, Any] = {}
        for lbl, raw in series.items():
            lbl_ts = pd.Timestamp(cast(Any, lbl))
            day = self._calendar_naive(lbl_ts)
            by_day[day] = raw
        out = np.zeros(len(calendar_naive_index), dtype=bool)
        for i, d in enumerate(calendar_naive_index):
            key = self._calendar_naive(pd.Timestamp(d))
            raw = by_day.get(key)
            if raw is None:
                continue
            out[i] = self._carlos_value_is_risk_off(raw)
        return out

    @staticmethod
    def _parse_month_int_overrides(raw: Any, *, field_name: str) -> Dict[int, int]:
        """Normalize ``entry_day_by_month`` / ``hold_days_by_month`` to month-index overrides."""
        if raw is None:
            return {}
        if not isinstance(raw, Mapping):
            raise TypeError(f"{field_name} must be a mapping or omitted, got {type(raw).__name__}")
        out: Dict[int, int] = {}
        for key, val in raw.items():
            month_index = SeasonalSignalStrategy._coerce_calendar_month_key(
                key, field_name=field_name
            )
            try:
                out[month_index] = int(val)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{field_name}: value for month {month_index} must be int-like"
                ) from exc
        return out

    @staticmethod
    def _coerce_calendar_month_key(key: Any, *, field_name: str) -> int:
        if isinstance(key, int):
            if 1 <= key <= 12:
                return key
            raise ValueError(f"{field_name}: invalid month index {key!r} (expected 1..12)")
        if isinstance(key, str):
            s = key.strip().lower()
            if s.isdigit():
                v = int(s)
                if 1 <= v <= 12:
                    return v
                raise ValueError(f"{field_name}: invalid month index {key!r} (expected 1..12)")
            for month_index in range(1, 13):
                full = calendar.month_name[month_index]
                abbr = calendar.month_abbr[month_index]
                if full and s == full.lower():
                    return month_index
                if abbr and s == abbr.lower():
                    return month_index
            raise ValueError(f"{field_name}: invalid month key {key!r}")
        raise TypeError(f"{field_name}: month keys must be int or str, got {type(key).__name__}")

    def _entry_day_for_calendar_month(self, calendar_month: int) -> int:
        return int(self._entry_day_by_month_map.get(int(calendar_month), self.entry_day))

    def _hold_days_for_calendar_month(self, calendar_month: int) -> int:
        return int(self._hold_days_by_month_map.get(int(calendar_month), self.hold_days))

    def _max_hold_days_for_scan(self) -> int:
        peak = int(self.hold_days)
        for hd in self._hold_days_by_month_map.values():
            peak = max(peak, int(hd))
        return peak

    @classmethod
    def tunable_parameters(cls) -> Dict[str, Dict[str, object]]:
        """Hyper-parameters exposed to scenario validation and optimization backends.

        Each ``trade_month_M`` (M=1..12) is a categorical on/off switch for that calendar
        month. Use ``optimize`` entries with ``type: categorical`` and
        ``values: [true, false]`` (or ``[True, False]``) unless overridden in
        ``OPTIMIZER_PARAMETER_DEFAULTS``.
        """
        month_bools: Dict[str, Dict[str, object]] = {
            f"trade_month_{m}": {
                "type": "categorical",
                "values": [True, False],
            }
            for m in range(1, 13)
        }
        base: Dict[str, Dict[str, object]] = {
            "month_local_seasonal_windows": {
                "type": "categorical",
                "values": [True, False],
                "default": False,
            },
            "direction": {
                "type": "categorical",
                "values": ["long", "short"],
                "default": "long",
            },
            "entry_day": {"type": "int", "default": 1, "min": -21, "max": 21},
            "hold_days": {"type": "int", "default": 5, "min": 5, "max": 20},
            "simple_high_low_stop_loss": {
                "type": "categorical",
                "values": [True, False],
                "default": False,
            },
            "simple_high_low_take_profit": {
                "type": "categorical",
                "values": [True, False],
                "default": False,
            },
            "stop_loss_atr_multiple": {
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 10.0,
                "step": 0.1,
            },
            "take_profit_atr_multiple": {
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 10.0,
                "step": 0.1,
            },
            "use_carlos_roro": {
                "type": "categorical",
                "values": [True, False],
                "default": False,
            },
            "carlos_roro_symbol": {
                "type": "str",
                "default": _DEFAULT_CARLOS_RORO_SYMBOL,
            },
            "max_dd_from_ath_pct": {
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 15.0,
                "step": 0.5,
            },
        }
        base.update(month_bools)
        return base

    def get_entry_date_for_month(self, date: pd.Timestamp, entry_day: int) -> pd.Timestamp:
        cal = self._calendar_naive(date)
        first_day = pd.Timestamp(year=cal.year, month=cal.month, day=1)
        last_day = (first_day + pd.offsets.MonthEnd(1)).normalize()
        bdays = pd.bdate_range(first_day, last_day)
        if entry_day == 0:
            entry_day = 1
        index = entry_day - 1 if entry_day > 0 else entry_day
        index = max(min(index, len(bdays) - 1), -len(bdays))
        return pd.Timestamp(bdays[index])

    def _is_within_hold_window(
        self,
        current_date: pd.Timestamp,
        entry_date: pd.Timestamp,
        hold_days: Optional[int] = None,
    ) -> bool:
        ent = self._calendar_naive(entry_date)
        hd = (
            int(hold_days)
            if hold_days is not None
            else self._hold_days_for_calendar_month(int(ent.month))
        )
        if hd <= 0:
            return False
        cur = self._calendar_naive(current_date)
        window_end = ent + BDay(hd - 1)
        return bool(ent <= cur <= window_end)

    def _month_allowed_for_calendar_month(self, month: int) -> bool:
        return bool(self.allowed_month.get(int(month), True))

    def _month_allowed(self, current_date: pd.Timestamp) -> bool:
        cal = self._calendar_naive(current_date)
        return self._month_allowed_for_calendar_month(int(cal.month))

    def _resolve_active_entry_date(self, current_date: pd.Timestamp) -> Optional[pd.Timestamp]:
        """Anchor entry date for the active seasonal cycle, or None if flat today."""
        if self.month_local_seasonal_windows:
            if not self._month_allowed(current_date):
                return None
            cal = self._calendar_naive(current_date)
            entry_day_m = self._entry_day_for_calendar_month(int(cal.month))
            hold_m = self._hold_days_for_calendar_month(int(cal.month))
            entry_date = self.get_entry_date_for_month(current_date, entry_day_m)
            if self._is_within_hold_window(current_date, entry_date, hold_m):
                return entry_date
            return None

        d_naive = self._calendar_naive(current_date)
        first_of_d_month = pd.Timestamp(year=d_naive.year, month=d_naive.month, day=1)
        k_max = min(24, max(3, (self._max_hold_days_for_scan() + 10) // 15 + 2))
        for k in range(k_max):
            anchor_month_start = first_of_d_month - pd.DateOffset(months=k)
            anchor_month = int(anchor_month_start.month)
            entry_day_anchor = self._entry_day_for_calendar_month(anchor_month)
            hold_anchor = self._hold_days_for_calendar_month(anchor_month)
            entry_m = self.get_entry_date_for_month(anchor_month_start, entry_day_anchor)
            if not self._month_allowed_for_calendar_month(int(entry_m.month)):
                continue
            if self._is_within_hold_window(current_date, entry_m, hold_anchor):
                return entry_m
        return None

    def _cross_month_scan_active(self, current_date: pd.Timestamp) -> bool:
        """True if current_date lies in any allowed anchor-month window (months scanned backward)."""
        return self._resolve_active_entry_date(current_date) is not None

    @staticmethod
    def _list_tickers_from_hist(all_historical_data: pd.DataFrame) -> List[str]:
        if isinstance(all_historical_data.columns, pd.MultiIndex):
            if "Field" in all_historical_data.columns.names:
                try:
                    close_cols = all_historical_data.xs("Close", level="Field", axis=1)
                    return [str(t) for t in close_cols.columns]
                except KeyError:
                    pass
            tickers = sorted(
                {
                    str(c[0]) if isinstance(c, tuple) and len(c) > 0 else str(c)
                    for c in all_historical_data.columns
                }
            )
            return tickers
        return [str(c) for c in all_historical_data.columns]

    def _universe_hist_close_subset(
        self,
        all_historical_data: pd.DataFrame,
        universe_tickers: List[str],
    ) -> pd.DataFrame:
        """Close columns aligned to universe tickers; missing tickers yield all-NaN columns."""
        if all_historical_data.empty:
            return pd.DataFrame(columns=list(universe_tickers))
        col_names = list(all_historical_data.columns.names or [])
        if isinstance(all_historical_data.columns, pd.MultiIndex) and "Field" in col_names:
            try:
                close_layer = cast(
                    pd.DataFrame,
                    all_historical_data.xs("Close", level="Field", axis=1),
                )
            except KeyError:
                return pd.DataFrame(index=all_historical_data.index, columns=list(universe_tickers))
        else:
            close_layer = all_historical_data
        aligned = pd.DataFrame(index=all_historical_data.index, columns=list(universe_tickers))
        present = [c for c in universe_tickers if c in close_layer.columns]
        if present:
            aligned.loc[:, present] = close_layer.loc[:, present].apply(
                pd.to_numeric, errors="coerce"
            )
        return aligned

    def _ticker_naive_close_through(
        self,
        all_historical_data: pd.DataFrame,
        ticker: str,
        end_ts: pd.Timestamp,
    ) -> pd.Series:
        end_naive = self._calendar_naive(end_ts)
        wide = self._universe_hist_close_subset(all_historical_data, [ticker])
        if wide.empty:
            return pd.Series(dtype=float)
        naive_hist = self._index_calendar_naive(pd.DatetimeIndex(wide.index))
        ser = pd.Series(wide[ticker].to_numpy(dtype=float), index=naive_hist, dtype=float)
        ser = ser[~ser.index.duplicated(keep="last")]
        return ser.loc[ser.index <= end_naive].sort_index()

    def _max_dd_vs_ath_blocks_long_allocation(self, close_through: pd.Series) -> bool:
        thr = float(self.max_dd_from_ath_pct)
        if self.direction != "long" or thr <= 0.0:
            return False
        vals = pd.to_numeric(close_through, errors="coerce").dropna()
        if vals.empty:
            return True
        ath = float(vals.expanding(min_periods=1).max().iloc[-1])
        cur = float(vals.iloc[-1])
        if ath <= 0.0 or math.isnan(ath) or math.isnan(cur):
            return True
        dd_pct = (ath - cur) / ath * 100.0
        return bool(dd_pct > thr)

    def _close_and_ath_on_scan_using_full_hist(
        self,
        hist_close: pd.DataFrame,
        calendar_naive_scan: pd.DatetimeIndex,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Close and ATH (running max close) at scan dates using full OHLC timeline."""
        if hist_close.empty:
            empty_close = pd.DataFrame(
                index=calendar_naive_scan, columns=hist_close.columns, dtype=float
            )
            empty_ath = empty_close.copy()
            return empty_close, empty_ath
        naive_hist = self._index_calendar_naive(pd.DatetimeIndex(hist_close.index))
        frame = pd.DataFrame(
            hist_close.values,
            index=naive_hist,
            columns=hist_close.columns,
            dtype=float,
        )
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        union_idx = frame.index.union(calendar_naive_scan).sort_values()
        filled = frame.reindex(union_idx, method="ffill")
        ath_on_union = filled.expanding(min_periods=1).max()
        close_at_scan = filled.reindex(calendar_naive_scan, method="ffill")
        ath_at_scan = ath_on_union.reindex(calendar_naive_scan, method="ffill")
        return close_at_scan, ath_at_scan

    def _try_extract_ohlc(
        self, all_historical_data: pd.DataFrame, tickers: List[str]
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """Return (close, high, low) aligned on tickers, or None if not available."""
        if not isinstance(all_historical_data.columns, pd.MultiIndex):
            return None
        if "Field" not in all_historical_data.columns.names:
            return None
        fields = set(all_historical_data.columns.get_level_values("Field").unique())
        if not {"Close", "High", "Low"}.issubset(fields):
            return None
        try:
            close = all_historical_data.xs("Close", level="Field", axis=1)
            high = all_historical_data.xs("High", level="Field", axis=1)
            low = all_historical_data.xs("Low", level="Field", axis=1)
        except KeyError:
            return None
        common = [
            t for t in tickers if t in close.columns and t in high.columns and t in low.columns
        ]
        if not common:
            return None
        out_close = close.loc[:, common]
        out_high = high.loc[:, common]
        out_low = low.loc[:, common]
        return cast(
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
            (out_close, out_high, out_low),
        )

    def _warn_missing_ohlc_once(self) -> None:
        if self._warned_missing_ohlc_for_sl_tp:
            return
        self._warned_missing_ohlc_for_sl_tp = True
        logger.warning(
            "SeasonalSignalStrategy: exit rules requested but High/Low/Close MultiIndex "
            "columns are unavailable; exit logic disabled for this run."
        )

    def _ticker_locked_this_cycle(
        self,
        *,
        close: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
        ticker: str,
        entry_date: pd.Timestamp,
        window_end: pd.Timestamp,
        current_date: pd.Timestamp,
        use_simple_sl: bool,
        use_simple_tp: bool,
        atr_series: Optional[pd.Series],
    ) -> bool:
        """Replay exits from entry through current_date; True if ticker exited."""
        idx = close.index
        ent_n = self._calendar_naive(entry_date)
        win_end_n = self._calendar_naive(window_end)
        cur_n = self._calendar_naive(current_date)
        last_n = min(cur_n, win_end_n)

        sl_atr_m = self.stop_loss_atr_multiple
        tp_atr_m = self.take_profit_atr_multiple

        entry_px: Optional[float] = None
        atr_v: Optional[float] = None
        try:
            ep_raw = close.loc[entry_date, ticker]
            if isinstance(ep_raw, pd.Series):
                ep_raw = ep_raw.iloc[0]
            entry_px = float(pd.to_numeric(ep_raw, errors="coerce"))
        except (KeyError, TypeError, ValueError):
            entry_px = None
        if atr_series is not None and ticker in atr_series.index:
            try:
                atr_v = float(atr_series[ticker])
            except (TypeError, ValueError):
                atr_v = float("nan")
        if atr_v is not None and pd.isna(atr_v):
            atr_v = None

        sl_price_atr: Optional[float] = None
        tp_price_atr: Optional[float] = None
        if (
            entry_px is not None
            and pd.notna(entry_px)
            and atr_v is not None
            and atr_v == atr_v
            and atr_v > 0
        ):
            if sl_atr_m > 0:
                if self.direction == "long":
                    sl_price_atr = entry_px - sl_atr_m * atr_v
                else:
                    sl_price_atr = entry_px + sl_atr_m * atr_v
            if tp_atr_m > 0:
                if self.direction == "long":
                    tp_price_atr = entry_px + tp_atr_m * atr_v
                else:
                    tp_price_atr = entry_px - tp_atr_m * atr_v

        locked = False
        for d in idx:
            if locked:
                break
            dn = self._calendar_naive(pd.Timestamp(d))
            if dn < ent_n or dn > last_n:
                continue
            loc = idx.get_indexer([pd.Timestamp(d)], method=None)[0]
            if loc < 0:
                continue

            try:
                c_raw = close.loc[d, ticker]
                if isinstance(c_raw, pd.Series):
                    c_raw = c_raw.iloc[0]
                c = float(pd.to_numeric(c_raw, errors="raise"))
            except (KeyError, TypeError, ValueError):
                continue
            if pd.isna(c):
                continue

            simple_sl_hit = False
            simple_tp_hit = False
            if loc > 0:
                prev_ts = idx[loc - 1]
                try:
                    hi_raw = high.loc[prev_ts, ticker]
                    lo_raw = low.loc[prev_ts, ticker]
                    if isinstance(hi_raw, pd.Series):
                        hi_raw = hi_raw.iloc[0]
                    if isinstance(lo_raw, pd.Series):
                        lo_raw = lo_raw.iloc[0]
                    hi_prev = float(pd.to_numeric(hi_raw, errors="raise"))
                    lo_prev = float(pd.to_numeric(lo_raw, errors="raise"))
                except (KeyError, TypeError, ValueError):
                    hi_prev = float("nan")
                    lo_prev = float("nan")

                if not pd.isna(hi_prev) and not pd.isna(lo_prev):
                    if self.direction == "long":
                        if use_simple_sl and c < lo_prev:
                            simple_sl_hit = True
                        if use_simple_tp and c > hi_prev:
                            simple_tp_hit = True
                    else:
                        if use_simple_sl and c > hi_prev:
                            simple_sl_hit = True
                        if use_simple_tp and c < lo_prev:
                            simple_tp_hit = True

            atr_sl_hit = False
            atr_tp_hit = False
            if sl_price_atr is not None:
                if self.direction == "long":
                    atr_sl_hit = c <= sl_price_atr
                else:
                    atr_sl_hit = c >= sl_price_atr
            if tp_price_atr is not None:
                if self.direction == "long":
                    atr_tp_hit = c >= tp_price_atr
                else:
                    atr_tp_hit = c <= tp_price_atr

            if use_simple_sl and simple_sl_hit:
                locked = True
            elif use_simple_tp and simple_tp_hit:
                locked = True
            elif sl_price_atr is not None and atr_sl_hit:
                locked = True
            elif tp_price_atr is not None and atr_tp_hit:
                locked = True

        return locked

    def generate_target_weights(self, context: StrategyContext) -> pd.DataFrame:
        idx = pd.DatetimeIndex(context.rebalance_dates)
        cols = list(context.universe_tickers)
        fv = np.nan if context.use_sparse_nan_for_inactive_rows else 0.0
        result = pd.DataFrame(fv, index=idx, columns=cols, dtype=float)
        if len(idx) == 0 or len(cols) == 0:
            return result

        asset_full = context.asset_data
        bench_full = context.benchmark_data
        nu_full = context.non_universe_data

        if asset_full.shape[1] == 0 or asset_full.empty:
            work_asset = pd.DataFrame(1.0, index=idx, columns=cols)
        else:
            work_asset = asset_full
        work_benchmark = bench_full

        for rd in idx:
            row_ts = pd.Timestamp(rd)
            sl_asset = work_asset.loc[work_asset.index <= row_ts]
            if work_benchmark.shape[1] > 0:
                sl_benchmark = work_benchmark.loc[work_benchmark.index <= row_ts]
            else:
                sl_benchmark = pd.DataFrame()
            nu_arg = None
            if not nu_full.empty and nu_full.shape[1] > 0:
                nu_slice = nu_full.loc[nu_full.index <= row_ts]
                nu_arg = nu_slice if nu_slice.shape[0] > 0 else None

            row = self.generate_signals(
                sl_asset,
                sl_benchmark,
                non_universe_historical_data=nu_arg,
                current_date=row_ts,
            ).iloc[0]
            aligned = row.reindex(cols)
            result.loc[row_ts, :] = aligned.fillna(0.0).to_numpy(dtype=float)

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
    ) -> pd.DataFrame:
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

        eval_ts = pd.Timestamp(current_date)

        tickers = self._list_tickers_from_hist(all_historical_data)
        result = pd.DataFrame(0.0, index=[eval_ts], columns=tickers)

        resolved_entry = self._resolve_active_entry_date(eval_ts)
        in_window = resolved_entry is not None

        use_simple_sl = self.simple_high_low_stop_loss
        use_simple_tp = self.simple_high_low_take_profit
        use_atr_exits = self.stop_loss_atr_multiple > 0 or self.take_profit_atr_multiple > 0
        need_exit_logic = use_simple_sl or use_simple_tp or use_atr_exits

        ohlc = self._try_extract_ohlc(all_historical_data, tickers) if need_exit_logic else None

        if need_exit_logic and ohlc is None:
            self._warn_missing_ohlc_once()
            need_exit_logic = False

        if not in_window:
            return result

        if self._carlos_roro_is_risk_off_at(non_universe_historical_data, eval_ts):
            return result

        if len(tickers) == 0:
            return result

        def usable_after_dd(tlist: List[str]) -> List[str]:
            if self.direction != "long" or float(self.max_dd_from_ath_pct) <= 0.0:
                return list(tlist)
            usable: List[str] = []
            for ti in tlist:
                thru = self._ticker_naive_close_through(all_historical_data, ti, eval_ts)
                if not self._max_dd_vs_ath_blocks_long_allocation(thru):
                    usable.append(ti)
            return usable

        if not need_exit_logic:
            tradable = usable_after_dd(list(tickers))
            if len(tradable) == 0:
                return result
            ew = 1.0 / len(tradable)
            for ti in tickers:
                alloc = ew if ti in tradable else 0.0
                result.loc[eval_ts, ti] = -alloc if self.direction == "short" else alloc
            return result

        assert resolved_entry is not None
        assert ohlc is not None
        close, high, low = ohlc

        ent_cal = self._calendar_naive(resolved_entry)
        hold_for_cycle = self._hold_days_for_calendar_month(int(ent_cal.month))
        window_end_ts = ent_cal + BDay(hold_for_cycle - 1)

        atr_series: Optional[pd.Series] = None
        if use_atr_exits:
            atr_series = calculate_atr_fast(all_historical_data, resolved_entry, _ATR_LOOKBACK)

        active: List[str] = []
        for t in tickers:
            if t not in close.columns:
                continue
            locked = self._ticker_locked_this_cycle(
                close=close,
                high=high,
                low=low,
                ticker=t,
                entry_date=resolved_entry,
                window_end=window_end_ts,
                current_date=eval_ts,
                use_simple_sl=use_simple_sl,
                use_simple_tp=use_simple_tp,
                atr_series=atr_series,
            )
            if not locked:
                active.append(t)

        active_allowed = usable_after_dd(active)
        if not active_allowed:
            return result

        w = 1.0 / len(active_allowed)
        allowed_set = set(active_allowed)
        for t in tickers:
            if t in allowed_set:
                result.loc[eval_ts, t] = -w if self.direction == "short" else w
            else:
                result.loc[eval_ts, t] = 0.0

        return result


__all__ = ["SeasonalSignalStrategy"]
