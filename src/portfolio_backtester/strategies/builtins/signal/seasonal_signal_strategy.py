from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Mapping, TYPE_CHECKING, cast

import pandas as pd
from pandas.tseries.offsets import BDay

from portfolio_backtester.risk_management.atr_service import calculate_atr_fast

from ..._core.base.base.signal_strategy import SignalStrategy

if TYPE_CHECKING:
    from portfolio_backtester.canonical_config import CanonicalScenarioConfig

logger = logging.getLogger(__name__)

_ATR_LOOKBACK = 21


class SeasonalSignalStrategy(SignalStrategy):
    """Intramonth seasonality: Nth business-day entry and a fixed business-day hold window.

    **Default (``month_local_seasonal_windows: false``):** Each session checks whether
    ``current_date`` falls inside **any** recent calendar month's anchored window
    ``[entry_m, entry_m + BDay(hold_days - 1)]``, where ``entry_m`` is that month's
    prescribed business-day entry. Holds therefore **continue across month boundaries**
    when the hold window spans them.

    **Cross-month cycle identity:** When multiple backward-scanned anchor months could
    apply, the **first** match in the same order as the internal month scan (current
    calendar month first, then prior months) defines the active cycle for SL/TP replay.

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

    Calendar note: entry is **Nth business day of the month**, not ``Timestamp.day``.
    Timestamps are compared on a **calendar wall-clock** basis after stripping timezone
    info so tz-aware price indices (e.g. US/Eastern) do not raise against naive anchors.
    """

    @staticmethod
    def _calendar_naive(ts: pd.Timestamp) -> pd.Timestamp:
        """Strip tz for Y/M/D and business-day window math (align with exchange calendar date)."""
        t = pd.Timestamp(ts)
        return t.tz_convert(None) if t.tzinfo is not None else t

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

        self._warned_missing_ohlc_for_sl_tp: bool = False

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
            "hold_days": {"type": "int", "default": 3, "min": 1, "max": 20},
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

    def _is_within_hold_window(self, current_date: pd.Timestamp, entry_date: pd.Timestamp) -> bool:
        if self.hold_days <= 0:
            return False
        cur = self._calendar_naive(current_date)
        ent = self._calendar_naive(entry_date)
        window_end = ent + BDay(self.hold_days - 1)
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
            entry_date = self.get_entry_date_for_month(current_date, self.entry_day)
            if self._is_within_hold_window(current_date, entry_date):
                return entry_date
            return None

        d_naive = self._calendar_naive(current_date)
        first_of_d_month = pd.Timestamp(year=d_naive.year, month=d_naive.month, day=1)
        k_max = min(24, max(3, (self.hold_days + 10) // 15 + 2))
        for k in range(k_max):
            anchor_month_start = first_of_d_month - pd.DateOffset(months=k)
            entry_m = self.get_entry_date_for_month(anchor_month_start, self.entry_day)
            if not self._month_allowed_for_calendar_month(int(entry_m.month)):
                continue
            if self._is_within_hold_window(current_date, entry_m):
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

        tickers = self._list_tickers_from_hist(all_historical_data)
        result = pd.DataFrame(0.0, index=[current_date], columns=tickers)

        resolved_entry = self._resolve_active_entry_date(current_date)
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

        if len(tickers) == 0:
            return result

        if not need_exit_logic:
            equal_weight = 1.0 / len(tickers)
            result.loc[current_date, :] = (
                -equal_weight if self.direction == "short" else equal_weight
            )
            return result

        assert resolved_entry is not None
        assert ohlc is not None
        close, high, low = ohlc

        ent_cal = self._calendar_naive(resolved_entry)
        window_end_ts = ent_cal + BDay(self.hold_days - 1)

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
                current_date=current_date,
                use_simple_sl=use_simple_sl,
                use_simple_tp=use_simple_tp,
                atr_series=atr_series,
            )
            if not locked:
                active.append(t)

        if not active:
            return result

        w = 1.0 / len(active)
        active_set = set(active)
        for t in tickers:
            if t in active_set:
                result.loc[current_date, t] = -w if self.direction == "short" else w
            else:
                result.loc[current_date, t] = 0.0

        return result


__all__ = ["SeasonalSignalStrategy"]
