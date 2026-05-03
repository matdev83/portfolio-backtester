"""Treasury yield levels to implied risk-free returns for performance metrics."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, Mapping, Optional, Union, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from portfolio_backtester.canonical_config import CanonicalScenarioConfig

logger = logging.getLogger(__name__)

RfYieldConvention = Literal["simple_split"]

_SYNTHETIC_BENCHMARKS = frozenset({"SP500_EQUAL_WEIGHT"})

# Case-insensitive tokens meaning "no yield ticker for this resolution scope".
_LEGACY_YIELD_TICKER_TOKENS = frozenset({"legacy", "none", "off"})

DEFAULT_RISK_FREE_YIELD_TICKER = "^IRX"


def _is_synthetic_ticker(ticker: object) -> bool:
    if ticker is None:
        return False
    return str(ticker).strip().upper() in _SYNTHETIC_BENCHMARKS


def _get_extras_mapping(
    scenario_config: Optional[Union["CanonicalScenarioConfig", Mapping[str, object]]],
) -> Optional[Mapping[str, object]]:
    if scenario_config is None:
        return None
    if hasattr(scenario_config, "extras"):
        ex = getattr(scenario_config, "extras", None)
        return cast(Mapping[str, object], ex) if isinstance(ex, Mapping) else None
    if isinstance(scenario_config, Mapping):
        raw_ex = scenario_config.get("extras")
        return cast(Mapping[str, object], raw_ex) if isinstance(raw_ex, Mapping) else None
    return None


def _scenario_value_explicit(
    scenario_config: Optional[Union["CanonicalScenarioConfig", Mapping[str, object]]],
    key: str,
) -> tuple[Optional[object], bool]:
    """Return (value, explicit) if ``key`` is set on scenario extras or scenario top-level."""
    if scenario_config is None:
        return None, False
    extras = _get_extras_mapping(scenario_config)
    if extras is not None and key in extras:
        return extras.get(key), True
    if isinstance(scenario_config, Mapping) and key in scenario_config:
        return scenario_config.get(key), True
    return None, False


def _coerce_bool_flag(val: object) -> Optional[bool]:
    """Interpret YAML-friendly booleans; return None if unset or unknown."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        if val == 1:
            return True
        if val == 0:
            return False
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off"):
            return False
    return None


def _risk_free_metrics_enabled(
    global_config: Mapping[str, object],
    scenario_config: Optional[Union["CanonicalScenarioConfig", Mapping[str, object]]],
) -> bool:
    """Whether Treasury-based excess metrics are enabled (default True)."""
    s_val, s_explicit = _scenario_value_explicit(scenario_config, "risk_free_metrics_enabled")
    if s_explicit:
        coerced = _coerce_bool_flag(s_val) if s_val is not None else None
        if coerced is False:
            return False
        if coerced is True:
            return True
        # Key present but null / invalid → treat as unset for this layer
    g_raw = global_config.get("risk_free_metrics_enabled")
    g_coerced = _coerce_bool_flag(g_raw) if g_raw is not None else None
    if g_coerced is False:
        return False
    return True


def _normalize_resolved_yield_ticker(val: object) -> Optional[str]:
    """Return a real ticker string, or None if disabled / empty / sentinel / synthetic."""
    if val is None:
        return None
    if not isinstance(val, str):
        return None
    s = val.strip()
    if not s:
        return None
    if s.lower() in _LEGACY_YIELD_TICKER_TOKENS:
        return None
    if _is_synthetic_ticker(s):
        return None
    return s


def resolve_risk_free_yield_ticker(
    global_config: Mapping[str, object],
    scenario_config: Optional[Union["CanonicalScenarioConfig", Mapping[str, object]]] = None,
) -> Optional[str]:
    """Return configured Treasury yield ticker for risk-free metrics, if any.

    Resolution order:

    * If ``risk_free_metrics_enabled`` is explicitly false on the scenario (extras or
      top-level) or on ``global_config``, returns ``None`` (legacy Sharpe path).
    * Scenario ``risk_free_yield_ticker`` (extras first, then top-level on dict
      scenarios): if the key is **present** with null, empty string, or tokens
      ``legacy`` / ``none`` / ``off`` (case-insensitive), returns ``None`` without
      falling back to global (per-scenario opt-out).
    * Otherwise falls back to ``global_config["risk_free_yield_ticker"]`` with the same
      normalization. Shipped defaults are applied in :mod:`portfolio_backtester.config_loader`.
    """
    if not _risk_free_metrics_enabled(global_config, scenario_config):
        return None

    s_val, s_explicit = _scenario_value_explicit(scenario_config, "risk_free_yield_ticker")
    if s_explicit:
        return _normalize_resolved_yield_ticker(s_val)

    g_val = global_config.get("risk_free_yield_ticker")
    return _normalize_resolved_yield_ticker(g_val)


def extract_yield_levels(price_data: pd.DataFrame, ticker: str, index: pd.Index) -> pd.Series:
    """Read yield **levels** (% per annum) from OHLC; do not apply pct_change."""
    if price_data is None or price_data.empty:
        return pd.Series(np.nan, index=index, dtype=float)

    try:
        if isinstance(price_data.columns, pd.MultiIndex):
            close_prices = price_data.xs(ticker, level="Ticker", axis=1)["Close"]
        elif ticker in price_data.columns:
            ticker_data = price_data[ticker]
            close_prices = (
                ticker_data["Close"]
                if isinstance(ticker_data, pd.DataFrame) and "Close" in ticker_data.columns
                else ticker_data
            )
        elif "Close" in price_data.columns:
            close_prices = price_data["Close"]
        else:
            close_prices = price_data.iloc[:, 0]
    except (KeyError, IndexError):
        logger.warning(
            "Risk-free yield ticker %s close data missing; risk-free metrics will be NaN.", ticker
        )
        return pd.Series(np.nan, index=index, dtype=float)

    if isinstance(close_prices, pd.DataFrame):
        if close_prices.empty or len(close_prices.columns) == 0:
            return pd.Series(np.nan, index=index, dtype=float)
        close_prices = close_prices.iloc[:, 0]

    ser = close_prices.reindex(index).astype(float)
    return cast(pd.Series, ser)


def yield_levels_to_implied_rf_returns(
    levels: pd.Series,
    steps_per_year: int,
    convention: RfYieldConvention = "simple_split",
) -> pd.Series:
    """Convert annualized yield % levels to implied per-bar simple risk-free returns.

    Default **simple_split**: rf_t = (Y_t / 100) / N with N = ``steps_per_year``
    (e.g. 252 for daily), treating the quoted yield as a simple annual rate.

    Args:
        levels: Annualized yield in percent (e.g. 5.0 for 5%).
        steps_per_year: Bars per year for the return frequency.
        convention: Only ``simple_split`` is implemented.

    Returns:
        Series of implied simple returns per bar, same index as ``levels``.
    """
    if convention != "simple_split":
        raise NotImplementedError(f"Unsupported risk-free convention: {convention!r}")
    n = max(int(steps_per_year), 1)
    y = levels.astype(float)
    return y / 100.0 / float(n)


def build_risk_free_return_series(
    price_data: pd.DataFrame,
    ticker: str,
    index: pd.Index,
    steps_per_year: int,
    convention: RfYieldConvention = "simple_split",
) -> pd.Series:
    """Extract yield levels from OHLC and convert to implied rf return series."""
    levels = extract_yield_levels(price_data, ticker, index)
    if levels.isna().all():
        return pd.Series(np.nan, index=index, dtype=float)
    return yield_levels_to_implied_rf_returns(levels, steps_per_year, convention=convention)


def align_risk_free_to_strategy_returns(
    strategy_rets: pd.Series, risk_free_rets: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Align rf to strategy index (ffill/bfill) and return (strategy, rf_aligned) same index."""
    idx = strategy_rets.index
    rf = risk_free_rets.reindex(idx).ffill().bfill()
    return strategy_rets, rf


def excess_returns(strategy_rets: pd.Series, risk_free_rets: pd.Series) -> pd.Series:
    """Strategy minus aligned rf; drop rows where excess is NaN."""
    _, rf_al = align_risk_free_to_strategy_returns(strategy_rets, risk_free_rets)
    ex = strategy_rets.astype(float) - rf_al.astype(float)
    return ex.dropna()


def risk_free_cumulative_growth(rf_simple_rets: pd.Series) -> float:
    """Cumulative growth ∏(1+rf_t)-1 over the series (NaNs dropped)."""
    x = rf_simple_rets.dropna().astype(float)
    if x.empty:
        return float("nan")
    return float(np.prod(1.0 + x.to_numpy(dtype=np.float64)) - 1.0)


def build_optional_risk_free_series(
    daily_ohlc: pd.DataFrame,
    global_config: Mapping[str, object],
    index: pd.Index,
    scenario_config: Optional[Union["CanonicalScenarioConfig", Mapping[str, object]]] = None,
) -> Optional[pd.Series]:
    """Build implied rf returns when ``risk_free_yield_ticker`` is configured; else None."""
    from portfolio_backtester.reporting.performance_metrics import _infer_steps_per_year

    ticker = resolve_risk_free_yield_ticker(global_config, scenario_config)
    if ticker is None:
        return None
    steps = _infer_steps_per_year(pd.DatetimeIndex(index))
    return build_risk_free_return_series(daily_ohlc, ticker, index, steps_per_year=steps)
