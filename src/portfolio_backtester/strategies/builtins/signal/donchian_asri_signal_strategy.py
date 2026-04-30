from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, TYPE_CHECKING, Union

import numpy as np
import pandas as pd

from ..._core.base.base.signal_strategy import SignalStrategy

if TYPE_CHECKING:
    from portfolio_backtester.canonical_config import CanonicalScenarioConfig


def donchian_position_close_breakout(
    close: pd.Series,
    *,
    entry_lookback: int = 20,
    exit_lookback: int = 10,
) -> pd.Series:
    """Long-only Donchian channel position based on closes (no look-ahead)."""
    if close.empty:
        return pd.Series(dtype="float64", name="donchian_pos")
    c = pd.to_numeric(close, errors="coerce").astype(float).copy()
    c = c[~c.index.duplicated(keep="last")].sort_index()

    e = int(entry_lookback)
    x = int(exit_lookback)
    if e <= 0 or x <= 0:
        raise ValueError("entry_lookback and exit_lookback must be positive.")

    high_n = c.shift(1).rolling(e, min_periods=e).max()
    low_n = c.shift(1).rolling(x, min_periods=x).min()
    entry = (c > high_n) & c.notna()
    exit_ = (c < low_n) & c.notna()

    pos = pd.Series(0.0, index=c.index, name="donchian_pos")
    state = 0.0
    for i, dt in enumerate(c.index):
        if i == 0:
            pos.iloc[i] = 0.0
            continue
        if state <= 0.0 and bool(entry.loc[dt]):
            state = 1.0
        elif state > 0.0 and bool(exit_.loc[dt]):
            state = 0.0
        pos.iloc[i] = state

    return pos


def donchian_position_with_risk_filter(
    close: pd.Series,
    *,
    risk_on: pd.Series,
    entry_lookback: int = 20,
    exit_lookback: int = 10,
) -> pd.Series:
    """Donchian channel position with risk-on filter + risk-off override."""
    if close.empty:
        return pd.Series(dtype="float64", name="donchian_pos")
    c = pd.to_numeric(close, errors="coerce").astype(float).copy()
    c = c[~c.index.duplicated(keep="last")].sort_index()

    e = int(entry_lookback)
    x = int(exit_lookback)
    if e <= 0 or x <= 0:
        raise ValueError("entry_lookback and exit_lookback must be positive.")

    ron = risk_on.reindex(c.index)
    ron = ron.fillna(False).astype(bool)

    high_n = c.shift(1).rolling(e, min_periods=e).max()
    low_n = c.shift(1).rolling(x, min_periods=x).min()
    breakout = (c > high_n) & c.notna()
    stop = (c < low_n) & c.notna()

    entry = breakout & ron
    exit_ = stop | (~ron)

    pos = pd.Series(0.0, index=c.index, name="donchian_pos")
    state = 0.0
    for i, dt in enumerate(c.index):
        if i == 0:
            pos.iloc[i] = 0.0
            continue
        if state <= 0.0 and bool(entry.loc[dt]):
            state = 1.0
        elif state > 0.0 and bool(exit_.loc[dt]):
            state = 0.0
        pos.iloc[i] = state

    return pos


def _logistic_weight(sig_0_100: pd.Series, *, center: float, slope: float) -> pd.Series:
    sig = pd.to_numeric(sig_0_100, errors="coerce")
    x = float(slope) * (sig - float(center))
    exp_x = np.exp(x.clip(-50, 50).to_numpy())
    raw = np.minimum(np.maximum(1.0 / (1.0 + exp_x), 0.0), 1.0)
    return pd.Series(raw, index=sig.index, name="asri_weight", dtype="float64")


class DonchianAsriSignalStrategy(SignalStrategy):
    """Donchian channel strategy with optional ASRI filter and sizing."""

    def __init__(self, strategy_config: Union[Mapping[str, Any], "CanonicalScenarioConfig"]):
        super().__init__(strategy_config)

        from portfolio_backtester.canonical_config import CanonicalScenarioConfig

        if isinstance(strategy_config, CanonicalScenarioConfig):
            sp = dict(strategy_config.strategy_params)
        else:
            params = strategy_config if strategy_config is not None else {}
            sp = params.get("strategy_params", params)

        self.entry_lookback: int = int(sp.get("entry_lookback", 20))
        self.exit_lookback: int = int(sp.get("exit_lookback", 10))
        self.use_asri_filter: bool = bool(sp.get("use_asri_filter", True))
        self.use_asri_sizing: bool = bool(sp.get("use_asri_sizing", False))
        self.asri_symbol: str = str(sp.get("asri_symbol", "MDMP:ASRI"))
        self.asri_threshold_quantile: float = float(sp.get("asri_threshold_quantile", 0.7))
        self.asri_size_center_quantile: float = float(sp.get("asri_size_center_quantile", 0.9))
        self.asri_size_slope: float = float(sp.get("asri_size_slope", 0.3))
        self.leverage: float = float(sp.get("leverage", 1.0))

    @classmethod
    def tunable_parameters(_cls) -> Dict[str, Dict[str, Any]]:
        return {
            "entry_lookback": {"type": "int", "default": 20, "min": 10, "max": 60, "step": 1},
            "exit_lookback": {"type": "int", "default": 10, "min": 5, "max": 40, "step": 1},
            "use_asri_filter": {"type": "bool", "default": True},
            "use_asri_sizing": {"type": "bool", "default": False},
            "asri_symbol": {"type": "str", "default": "MDMP:ASRI"},
            "asri_threshold_quantile": {
                "type": "float",
                "default": 0.7,
                "min": 0.6,
                "max": 0.95,
                "step": 0.05,
            },
            "asri_size_center_quantile": {
                "type": "float",
                "default": 0.9,
                "min": 0.6,
                "max": 0.9,
                "step": 0.1,
            },
            "asri_size_slope": {
                "type": "float",
                "default": 0.3,
                "min": 0.05,
                "max": 0.3,
                "step": 0.05,
            },
            "leverage": {"type": "float", "default": 1.0, "min": 1.0, "max": 1.0, "step": 0.1},
        }

    def get_non_universe_data_requirements(self) -> list[str]:
        if self.use_asri_filter or self.use_asri_sizing:
            return [self.asri_symbol]
        return []

    def _extract_close_frame(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def _asri_series(
        self, non_universe_historical_data: Optional[pd.DataFrame]
    ) -> pd.Series | None:
        if non_universe_historical_data is None or non_universe_historical_data.empty:
            return None
        close = self._extract_close_frame(non_universe_historical_data)
        if self.asri_symbol not in close.columns:
            return None
        return pd.to_numeric(close[self.asri_symbol], errors="coerce")

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
        tickers = list(close_prices.columns)
        result = pd.DataFrame(0.0, index=[current_date], columns=tickers)

        asri_series = self._asri_series(non_universe_historical_data)
        if (self.use_asri_filter or self.use_asri_sizing) and asri_series is None:
            return result

        asri_hist = (
            asri_series.loc[:current_date].dropna() if asri_series is not None else pd.Series(dtype="float64")
        )
        if (self.use_asri_filter or self.use_asri_sizing) and asri_hist.empty:
            return result

        risk_on = None
        size_weight = None
        if asri_series is not None:
            thr = float(asri_hist.quantile(self.asri_threshold_quantile))
            risk_on = (asri_series < thr).reindex(close_prices.index).fillna(False)
            if self.use_asri_sizing:
                center = float(asri_hist.quantile(self.asri_size_center_quantile))
                size_weight = _logistic_weight(asri_series, center=center, slope=self.asri_size_slope)

        for ticker in tickers:
            close = close_prices[ticker].loc[:current_date]
            if close.empty:
                continue
            if self.use_asri_filter and risk_on is not None:
                pos_series = donchian_position_with_risk_filter(
                    close,
                    risk_on=risk_on,
                    entry_lookback=self.entry_lookback,
                    exit_lookback=self.exit_lookback,
                )
            else:
                pos_series = donchian_position_close_breakout(
                    close,
                    entry_lookback=self.entry_lookback,
                    exit_lookback=self.exit_lookback,
                )
            if current_date not in pos_series.index:
                continue
            pos = float(pos_series.loc[current_date])
            if self.use_asri_sizing and size_weight is not None and current_date in size_weight.index:
                pos *= float(size_weight.loc[current_date])
            result.loc[current_date, ticker] = pos * self.leverage

        return result


class AsriThresholdSignalStrategy(SignalStrategy):
    """Simple ASRI threshold risk-on signal (no Donchian logic)."""

    def __init__(self, strategy_config: Union[Mapping[str, Any], "CanonicalScenarioConfig"]):
        super().__init__(strategy_config)

        from portfolio_backtester.canonical_config import CanonicalScenarioConfig

        if isinstance(strategy_config, CanonicalScenarioConfig):
            sp = dict(strategy_config.strategy_params)
        else:
            params = strategy_config if strategy_config is not None else {}
            sp = params.get("strategy_params", params)

        self.asri_symbol: str = str(sp.get("asri_symbol", "MDMP:ASRI"))
        self.asri_threshold_quantile: float = float(sp.get("asri_threshold_quantile", 0.7))
        self.leverage: float = float(sp.get("leverage", 1.0))

    @classmethod
    def tunable_parameters(_cls) -> Dict[str, Dict[str, Any]]:
        return {
            "asri_symbol": {"type": "str", "default": "MDMP:ASRI"},
            "asri_threshold_quantile": {
                "type": "float",
                "default": 0.7,
                "min": 0.6,
                "max": 0.95,
                "step": 0.05,
            },
            "leverage": {"type": "float", "default": 1.0, "min": 1.0, "max": 1.0, "step": 0.1},
        }

    def get_non_universe_data_requirements(self) -> list[str]:
        return [self.asri_symbol]

    def _extract_close_frame(self, df: pd.DataFrame) -> pd.DataFrame:
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

        tickers = list(self._extract_close_frame(all_historical_data).columns)
        result = pd.DataFrame(0.0, index=[current_date], columns=tickers)

        if non_universe_historical_data is None or non_universe_historical_data.empty:
            return result
        asri_close = self._extract_close_frame(non_universe_historical_data)
        if self.asri_symbol not in asri_close.columns:
            return result

        asri_series = pd.to_numeric(asri_close[self.asri_symbol], errors="coerce")
        hist = asri_series.loc[:current_date].dropna()
        if hist.empty:
            return result
        thr = float(hist.quantile(self.asri_threshold_quantile))
        risk_on = bool(asri_series.loc[current_date] < thr) if current_date in asri_series.index else False
        if risk_on and tickers:
            weight = self.leverage / float(len(tickers))
            result.loc[current_date, :] = weight
        return result


__all__ = [
    "AsriThresholdSignalStrategy",
    "DonchianAsriSignalStrategy",
    "donchian_position_close_breakout",
    "donchian_position_with_risk_filter",
]
