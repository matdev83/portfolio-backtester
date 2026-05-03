"""Baseline: SeasonalSignalStrategy mask 4086 with vs without Carlos RoRo overlay.

Uses final ``best_subset_month_params.tsv``, BacktestRunner with
``generate_signal_matrix`` disabled (detailed signal path), IBKR-like costs,
slippage 0.5 bps, risk-free metrics off.

Data: ``prepare_data_for_backtesting([scenario_off, scenario_on])`` so
``MDMP:RORO.CARLOS`` is always in the prefetch set.
"""

from __future__ import annotations

# ruff: noqa: E402

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from portfolio_backtester import config_loader
from portfolio_backtester.backtester_logic.backtest_runner import BacktestRunner
from portfolio_backtester.backtester_logic.data_fetcher import DataFetcher
from portfolio_backtester.backtester_logic.strategy_manager import StrategyManager
from portfolio_backtester.interfaces import create_cache_manager, create_data_source
from portfolio_backtester.reporting.fast_objective_metrics import calculate_optimizer_metrics_fast
from portfolio_backtester.scenario_normalizer import ScenarioNormalizer
from portfolio_backtester.strategies.builtins.signal.seasonal_signal_strategy import (
    SeasonalSignalStrategy,
)

PORT = 100_000.0
CPS = 0.005
CMIN = 1.0
CMAX = 0.005
SLIP = 0.5
START = "2005-01-01"
END = "2024-12-31"
TSV_SRC = (
    _REPO
    / "dev"
    / "spy_intramonth_detailed_ibkr_rf_off_2005_2024_final"
    / "best_subset_month_params.tsv"
)
CARLOS = "MDMP:RORO.CARLOS"
OUT_DIR = _REPO / "dev" / "spy_seasonal_mask4086_roro_baseline_ibkr_rf_off"


def _load_tsv(path: Path) -> tuple[int, dict[int, int], dict[int, int]]:
    df = pd.read_csv(path, sep="\t")
    bitmask = 0
    entry: dict[int, int] = {}
    hold: dict[int, int] = {}
    for _, row in df.iterrows():
        m = int(row["calendar_month"])
        bitmask |= int(row["active"]) << (m - 1)
        entry[m] = int(row["entry_day"])
        hold[m] = int(row["hold_days"])
    return bitmask, entry, hold


def _strategy_params(
    bitmask: int, entry_by_month: dict[int, int], hold_by_month: dict[int, int]
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "direction": "long",
        "month_local_seasonal_windows": False,
        "entry_day": 1,
        "hold_days": max(hold_by_month.values()),
        "entry_day_by_month": dict(entry_by_month),
        "hold_days_by_month": dict(hold_by_month),
        "simple_high_low_stop_loss": False,
        "simple_high_low_take_profit": False,
        "stop_loss_atr_multiple": 0.0,
        "take_profit_atr_multiple": 0.0,
    }
    for month in range(1, 13):
        params[f"trade_month_{month}"] = bool((bitmask >> (month - 1)) & 1)
    return params


def scenario(*, label: str, use_carlos_roro: bool, base_params: dict[str, Any]) -> dict[str, Any]:
    p = dict(base_params)
    p["use_carlos_roro"] = bool(use_carlos_roro)
    return {
        "name": label,
        "strategy": "SeasonalSignalStrategy",
        "start_date": START,
        "end_date": END,
        "benchmark_ticker": "SPY",
        "rebalance_frequency": "ME",
        "position_sizer": "equal_weight",
        "timing_config": {
            "mode": "signal_based",
            "scan_frequency": "D",
            "min_holding_period": 1,
            "trade_execution_timing": "bar_close",
        },
        "universe_config": {"type": "fixed", "tickers": ["SPY"]},
        "train_window_months": 36,
        "test_window_months": 12,
        "extras": {"is_wfo": False, "risk_free_metrics_enabled": False},
        "optimization_targets": [{"name": "Sortino", "direction": "maximize"}],
        "strategy_params": p,
    }


def _period_filter(index: pd.Index) -> pd.Series:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        cmp_idx = pd.DatetimeIndex(
            [pd.Timestamp(ts).tz_convert(idx.tz).replace(tzinfo=None) for ts in idx]
        )
    else:
        cmp_idx = idx
    return pd.Series((cmp_idx >= pd.Timestamp(START)) & (cmp_idx <= pd.Timestamp(END)), index=index)


def _metrics(
    returns: pd.Series,
    *,
    spy_close_on_index: pd.Series,
) -> dict[str, float]:
    benchmark = spy_close_on_index.pct_change(fill_method=None).fillna(0.0).astype(float)
    raw = calculate_optimizer_metrics_fast(
        returns.astype(float),
        benchmark,
        "SPY",
        risk_free_rets=None,
        requested_metrics={
            "Sortino",
            "Sharpe",
            "Calmar",
            "Total Return",
            "Max Drawdown",
            "Ann. Return",
        },
    )
    return {str(k): float(v) for k, v in raw.items()}


def _underwater(eq: pd.Series) -> pd.Series:
    peak = eq.cummax()
    return eq / peak - 1.0


def main() -> int:
    bitmask, entry_m, hold_m = _load_tsv(TSV_SRC)
    expected = 4086
    if bitmask != expected:
        raise RuntimeError(f"Bitmask from TSV {bitmask} != expected {expected}")
    base_sp = _strategy_params(bitmask, entry_m, hold_m)

    config_loader.load_config()
    gc = config_loader.GLOBAL_CONFIG
    gc.setdefault("data_source_config", {})
    dsc = gc["data_source_config"]
    if isinstance(dsc, dict):
        dsc["cache_only"] = False

    gc.update(
        {
            "portfolio_value": PORT,
            "commission_per_share": CPS,
            "commission_min_per_order": CMIN,
            "commission_max_percent_of_trade": CMAX,
            "slippage_bps": SLIP,
        }
    )

    normalizer = ScenarioNormalizer()
    can_off = normalizer.normalize(
        scenario=scenario(
            label="spy_mask4086_roro_off", use_carlos_roro=False, base_params=base_sp
        ),
        global_config=gc,
    )
    can_on = normalizer.normalize(
        scenario=scenario(label="spy_mask4086_roro_on", use_carlos_roro=True, base_params=base_sp),
        global_config=gc,
    )

    strategy_manager = StrategyManager()
    data_cache = create_cache_manager()
    fetcher = DataFetcher(gc, create_data_source(gc))
    daily_ohlc, monthly_data, daily_closes = fetcher.prepare_data_for_backtesting(
        [can_off, can_on],
        strategy_manager.get_strategy,
    )
    cached_rets = data_cache.get_cached_returns(daily_closes, "full_period_returns")
    rets_full = (
        cached_rets.to_frame() if isinstance(cached_rets, pd.Series) else pd.DataFrame(cached_rets)
    )

    non_uni_hist_cols = SeasonalSignalStrategy(
        {"strategy_params": dict(base_sp, **{"use_carlos_roro": True})}
    ).get_non_universe_data_requirements()
    if non_uni_hist_cols != [CARLOS]:
        raise RuntimeError(f"Unexpected non-universe requirements: {non_uni_hist_cols}")

    risky_idx = pd.DatetimeIndex([d for d in daily_ohlc.index])
    naive_idx = (
        risky_idx.tz_convert(None)
        if getattr(risky_idx, "tz", None) is not None
        else pd.DatetimeIndex(risky_idx)
    )

    strat_roro_only = SeasonalSignalStrategy(
        {"strategy_params": dict(base_sp, **{"use_carlos_roro": True})}
    )
    nu_view = None
    if isinstance(daily_ohlc.columns, pd.MultiIndex):
        flds = list(daily_ohlc.columns.get_level_values("Field").unique())
        cols = pd.MultiIndex.from_product([[CARLOS], flds], names=["Ticker", "Field"]).intersection(
            daily_ohlc.columns
        )
        if len(cols):
            nu_view = daily_ohlc[cols]

    overlay_risk_off = pd.Series(False, index=naive_idx)
    if nu_view is not None and len(nu_view.columns):
        mask_naive = strat_roro_only._carlos_roro_risk_off_mask(naive_idx, nu_view)  # noqa: SLF001
        if mask_naive is None:
            raise RuntimeError("RoRo overlay mask is None unexpectedly")
        overlay_risk_off = (
            pd.Series(mask_naive, index=risky_idx).reindex(daily_ohlc.index).fillna(False)
        )
    elif isinstance(daily_ohlc.columns, pd.MultiIndex):
        pass

    spy_close_full = daily_ohlc.xs("Close", level="Field", axis=1)["SPY"]

    original_matrix = SeasonalSignalStrategy.generate_signal_matrix

    def _disable_signal_matrix(_self: SeasonalSignalStrategy, *_a: Any, **_k: Any) -> Any:
        return None

    SeasonalSignalStrategy.generate_signal_matrix = _disable_signal_matrix  # type: ignore[assignment]

    runner = BacktestRunner(gc, data_cache, strategy_manager, lambda: False)
    try:
        raw_off = runner.run_scenario(can_off, monthly_data, daily_ohlc, rets_full, verbose=False)
        raw_on = runner.run_scenario(can_on, monthly_data, daily_ohlc, rets_full, verbose=False)
    finally:
        SeasonalSignalStrategy.generate_signal_matrix = original_matrix  # type: ignore[method-assign]

    if raw_off is None or raw_on is None:
        raise RuntimeError("Backtest returned empty series")

    pm = _period_filter(raw_off.index)
    pm2 = _period_filter(raw_on.index)
    r_off = raw_off.loc[pm.to_numpy()].astype(float)
    r_on = raw_on.loc[pm2.to_numpy()].astype(float)

    ix = r_off.index.intersection(r_on.index)
    ix = ix.intersection(spy_close_full.index)
    r_off = r_off.reindex(ix).fillna(0.0).astype(float)
    r_on = r_on.reindex(ix).fillna(0.0).astype(float)
    spy_aligned = spy_close_full.reindex(ix)

    metrics_off = _metrics(r_off, spy_close_on_index=spy_aligned)
    metrics_on = _metrics(r_on, spy_close_on_index=spy_aligned)

    eq_off = (1.0 + r_off).cumprod()
    eq_on = (1.0 + r_on).cumprod()
    uw_off = _underwater(eq_off)
    uw_on = _underwater(eq_on)

    diff = r_off - r_on
    carlos_daily = overlay_risk_off.reindex(ix).fillna(False)
    ticker_level = (
        isinstance(daily_ohlc.columns, pd.MultiIndex) and "Ticker" in daily_ohlc.columns.names
    )
    tickers = (
        list(daily_ohlc.columns.get_level_values("Ticker").unique())
        if ticker_level
        else [str(x) for x in list(daily_ohlc.columns)]
    )
    rr_like = [
        str(t)
        for t in tickers
        if isinstance(t, str) and ("RORO" in t.upper() or "CARLOS" in t.upper())
    ]
    overlay_stats: dict[str, Any] = {
        "mdmp_symbol": CARLOS,
        "daily_ohlc_has_carlos_ticker_level": CARLOS in tickers,
        "sessions_in_calendar": int(len(overlay_risk_off)),
        "sessions_in_output_window": int(len(ix)),
        "risk_off_sessions_window": int(carlos_daily.astype(bool).sum()),
        "risk_off_share_window": float(carlos_daily.astype(float).mean()) if len(ix) else 0.0,
        "roro_like_tickers_in_ohlc": rr_like[:20],
        "max_abs_daily_return_delta_off_minus_on": float(diff.abs().max()) if len(diff) else 0.0,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    r_off.to_csv(OUT_DIR / "detailed_daily_returns_use_carlos_roro_false.csv", header=["return"])
    r_on.to_csv(OUT_DIR / "detailed_daily_returns_use_carlos_roro_true.csv", header=["return"])
    spread_uw = (uw_off.astype(float) - uw_on.astype(float)).to_numpy(dtype=float)
    pd.DataFrame(
        {
            "underwater_false": uw_off.astype(float).to_numpy(dtype=float),
            "underwater_true": uw_on.astype(float).to_numpy(dtype=float),
            "spread_underwater_false_minus_true": spread_uw,
        },
        index=ix,
    ).to_csv(OUT_DIR / "drawdown_underwater_compare.csv")

    months_on = [m for m in range(1, 13) if (bitmask >> (m - 1)) & 1]
    summary = {
        "params_path": str(TSV_SRC),
        "bitmask": bitmask,
        "active_calendar_months": months_on,
        "entry_day_by_month": entry_m,
        "hold_days_by_month": hold_m,
        "period": {"start": START, "end": END},
        "costs": {
            "portfolio_value": PORT,
            "commission_per_share": CPS,
            "commission_min_per_order": CMIN,
            "commission_max_percent_of_trade": CMAX,
            "slippage_bps": SLIP,
        },
        "risk_free_metrics_enabled": False,
        "detailed_path": "SeasonalSignalStrategy.generate_signal_matrix disabled",
        "metrics_use_carlos_roro_false": metrics_off,
        "metrics_use_carlos_roro_true": metrics_on,
        "drawdown_summary": {
            "min_underwater_false": float(uw_off.min()),
            "min_underwater_true": float(uw_on.min()),
            "max_drawdown_metric_false": metrics_off.get("Max Drawdown", float("nan")),
            "max_drawdown_metric_true": metrics_on.get("Max Drawdown", float("nan")),
        },
        "carlos_overlay": overlay_stats,
    }
    (OUT_DIR / "baseline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"WROTE={OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
