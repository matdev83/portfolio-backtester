"""Cached full-series features built on :class:`MarketDataPanel`.

Rolling mean convention
-----------------------
:class:`FeatureStore.get_rolling_mean` uses a **trailing** window with
``pandas.Series.rolling(window=window, min_periods=window).mean()`` on the
requested source series (default: close). No centering; the value at row ``t``
uses observations ``t - window + 1 .. t`` inclusive.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from ..numba_optimized import atr_fast_fixed
from .market_data_panel import MarketDataPanel


@dataclass(frozen=True)
class _FeatureStoreCacheKey:
    panel_fingerprint: str
    ticker: str
    feature: str
    params: tuple[tuple[str, Any], ...]
    row_bounds: Optional[tuple[int, int]]


def market_data_panel_fingerprint(panel: MarketDataPanel) -> str:
    """Stable digest of panel identity for cache namespacing."""

    h = hashlib.blake2b(digest_size=16, usedforsecurity=False)
    h.update(repr(panel.tickers).encode())
    h.update(str(len(panel.daily_index_naive)).encode())
    h.update(np.asarray(panel.daily_index_naive.values, dtype="datetime64[ns]").tobytes())
    h.update(np.ascontiguousarray(panel.daily_close_np, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(panel.returns_np, dtype=np.float32).tobytes())
    if panel.open_np is not None:
        h.update(np.ascontiguousarray(panel.open_np, dtype=np.float32).tobytes())
    if panel.high_np is not None:
        h.update(np.ascontiguousarray(panel.high_np, dtype=np.float32).tobytes())
    if panel.low_np is not None:
        h.update(np.ascontiguousarray(panel.low_np, dtype=np.float32).tobytes())
    return h.hexdigest()


class FeatureStore:
    """Full-length per-ticker feature cache backed by a :class:`MarketDataPanel`."""

    def __init__(self, panel: MarketDataPanel) -> None:
        self._panel = panel
        self._cache: dict[_FeatureStoreCacheKey, np.ndarray] = {}

    def clear_cache(self) -> None:
        """Drop all cached arrays."""

        self._cache.clear()

    def _make_key(
        self,
        ticker: str,
        feature: str,
        params: Mapping[str, Any],
        row_bounds: Optional[tuple[int, int]],
    ) -> _FeatureStoreCacheKey:
        params_tuple = tuple(sorted(params.items(), key=lambda kv: kv[0]))
        return _FeatureStoreCacheKey(
            panel_fingerprint=market_data_panel_fingerprint(self._panel),
            ticker=ticker,
            feature=feature,
            params=params_tuple,
            row_bounds=row_bounds,
        )

    def _col(self, ticker: str) -> int:
        if ticker not in self._panel.ticker_to_column:
            raise KeyError(f"Unknown ticker {ticker!r} for FeatureStore panel")
        return int(self._panel.ticker_to_column[ticker])

    def _apply_bounds(self, arr: np.ndarray, bounds: Optional[tuple[int, int]]) -> np.ndarray:
        if bounds is None:
            return arr
        lo, hi = bounds
        return arr[lo:hi]

    def get_returns(
        self,
        ticker: str,
        *,
        row_bounds: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        """Copy of aligned simple returns for ``ticker`` (panel fill rules)."""

        key = self._make_key(ticker, "returns", {}, row_bounds)
        if key in self._cache:
            return self._cache[key].copy()
        j = self._col(ticker)
        raw = np.asarray(self._panel.returns_np[:, j], dtype=np.float64)
        out = self._apply_bounds(raw, row_bounds).copy()
        miss_key = self._make_key(ticker, "returns", {}, None)
        if row_bounds is None:
            self._cache[miss_key] = out
        return out.copy()

    def get_atr(
        self,
        ticker: str,
        *,
        period: int,
        row_bounds: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        """ATR series via :func:`~portfolio_backtester.numba_optimized.atr_fast_fixed`."""

        key = self._make_key(ticker, "atr", {"period": period}, row_bounds)
        if key in self._cache:
            return self._cache[key].copy()
        if self._panel.high_np is None or self._panel.low_np is None:
            raise ValueError("ATR requires High/Low OHLC arrays on MarketDataPanel")
        j = self._col(ticker)
        high = np.asarray(self._panel.high_np[:, j], dtype=np.float64)
        low = np.asarray(self._panel.low_np[:, j], dtype=np.float64)
        close = np.asarray(self._panel.daily_close_np[:, j], dtype=np.float64)
        full = np.asarray(atr_fast_fixed(high, low, close, int(period)), dtype=np.float64)
        out = self._apply_bounds(full, row_bounds).copy()
        if row_bounds is None:
            store_key = self._make_key(ticker, "atr", {"period": period}, None)
            self._cache[store_key] = out.copy()
        return out.copy()

    def get_adx(
        self,
        ticker: str,
        *,
        di_length: int,
        adx_smoothing: int,
        row_bounds: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        """ADX/DMI (Wilder) consistent with ``_compute_adx_series``."""

        key = self._make_key(
            ticker,
            "adx",
            {"adx_smoothing": adx_smoothing, "di_length": di_length},
            row_bounds,
        )
        if key in self._cache:
            return self._cache[key].copy()
        if self._panel.high_np is None or self._panel.low_np is None:
            raise ValueError("ADX requires High/Low OHLC arrays on MarketDataPanel")

        from ..strategies.builtins.signal.mmm_qs_swing_nasdaq_signal_strategy import (
            _compute_adx_series,
        )

        j = self._col(ticker)
        ix = self._panel.daily_index_naive
        high = pd.Series(np.asarray(self._panel.high_np[:, j], dtype=float), index=ix)
        low = pd.Series(np.asarray(self._panel.low_np[:, j], dtype=float), index=ix)
        close = pd.Series(np.asarray(self._panel.daily_close_np[:, j], dtype=float), index=ix)
        adx = _compute_adx_series(high, low, close, int(di_length), int(adx_smoothing))
        full = np.asarray(adx.to_numpy(dtype=np.float64), dtype=np.float64)
        out = self._apply_bounds(full, row_bounds).copy()
        if row_bounds is None:
            store_key = self._make_key(
                ticker,
                "adx",
                {"adx_smoothing": adx_smoothing, "di_length": di_length},
                None,
            )
            self._cache[store_key] = out.copy()
        return out.copy()

    def get_rolling_mean(
        self,
        ticker: str,
        *,
        window: int,
        source: str = "close",
        row_bounds: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        """Trailing rolling mean (``min_periods=window``)."""

        key = self._make_key(
            ticker, "rolling_mean", {"source": source, "window": window}, row_bounds
        )
        if key in self._cache:
            return self._cache[key].copy()
        j = self._col(ticker)
        src_l = source.lower().strip()
        if src_l == "close":
            values = np.asarray(self._panel.daily_close_np[:, j], dtype=float)
        elif src_l == "high":
            if self._panel.high_np is None:
                raise ValueError("rolling mean on high requires High array")
            values = np.asarray(self._panel.high_np[:, j], dtype=float)
        elif src_l == "low":
            if self._panel.low_np is None:
                raise ValueError("rolling mean on low requires Low array")
            values = np.asarray(self._panel.low_np[:, j], dtype=float)
        elif src_l == "open":
            if self._panel.open_np is None:
                raise ValueError("rolling mean on open requires Open array")
            values = np.asarray(self._panel.open_np[:, j], dtype=float)
        else:
            raise ValueError(f"Unsupported source {source!r}")
        ser = pd.Series(values, index=self._panel.daily_index_naive)
        w = int(window)
        full = ser.rolling(window=w, min_periods=w).mean().to_numpy(dtype=np.float64)
        out = self._apply_bounds(full, row_bounds).copy()
        if row_bounds is None:
            store_key = self._make_key(
                ticker, "rolling_mean", {"source": source, "window": window}, None
            )
            self._cache[store_key] = out.copy()
        return out.copy()

    def get_ath_drawdown_pct(
        self,
        ticker: str,
        *,
        row_bounds: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        """Percent drawdown: ``100 * (1 - close / expanding_max(high))``."""

        key = self._make_key(ticker, "ath_drawdown_pct", {}, row_bounds)
        if key in self._cache:
            return self._cache[key].copy()
        if self._panel.high_np is None:
            raise ValueError("ATH drawdown requires High array on MarketDataPanel")
        j = self._col(ticker)
        high = np.asarray(self._panel.high_np[:, j], dtype=np.float64)
        close = np.asarray(self._panel.daily_close_np[:, j], dtype=np.float64)
        ath = np.maximum.accumulate(high)
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = 100.0 * (1.0 - close / ath)
        dd = np.where((ath > 0.0) & np.isfinite(ath) & np.isfinite(close), dd, np.nan)
        full = np.asarray(dd, dtype=np.float64)
        out = self._apply_bounds(full, row_bounds).copy()
        if row_bounds is None:
            store_key = self._make_key(ticker, "ath_drawdown_pct", {}, None)
            self._cache[store_key] = out.copy()
        return out.copy()
