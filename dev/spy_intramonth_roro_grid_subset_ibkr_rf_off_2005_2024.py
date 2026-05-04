"""SPY intramonth seasonal: RoRo-aware Task A + subset Sortino (IBKR-like costs, RF off).

Task A: For each calendar month, grid ``entry_day`` 1..20, ``hold_days`` 5..20,
``use_carlos_roro`` False/True; maximize Sortino on net daily returns (fast path).

Task B: Exhaust non-empty month subsets (4095). Uses first-anchor allocation scan
(matching ``SeasonalSignalStrategy`` cross-month semantics).

Variants:
  - ``mixed_params``: per-month winners from the unrestricted Task A grid; subset eval with
    global ``use_carlos_roro`` False vs True (same entry/hold tuples).
  - ``consistent_off`` / ``consistent_on``: Task A restricted to a fixed RoRo flag; subset
    eval with matching global flag (deployable tuning).

Final verification: ``BacktestRunner`` with ``generate_signal_matrix`` disabled for top
subset candidates under global RoRo off/on.

Period default: 2005-01-01 .. 2024-12-31. Slippage 0.5 bps. Carlos symbol MDMP:RORO.CARLOS.
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
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
CARLOS = "MDMP:RORO.CARLOS"
START_DEF = "2005-01-01"
END_DEF = "2024-12-31"

_MONTH_EN = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def _scenario_template(
    *,
    name: str,
    strategy_params: dict[str, Any],
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "strategy": "SeasonalSignalStrategy",
        "start_date": start_date,
        "end_date": end_date,
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
        "strategy_params": strategy_params,
    }


def _params_single_month(
    calendar_month: int,
    entry_day: int,
    hold_days: int,
    *,
    use_carlos_roro: bool,
) -> dict[str, Any]:
    sp: dict[str, Any] = {
        "direction": "long",
        "month_local_seasonal_windows": False,
        "entry_day": 1,
        "hold_days": 5,
        "entry_day_by_month": {int(calendar_month): int(entry_day)},
        "hold_days_by_month": {int(calendar_month): int(hold_days)},
        "simple_high_low_stop_loss": False,
        "simple_high_low_take_profit": False,
        "stop_loss_atr_multiple": 0.0,
        "take_profit_atr_multiple": 0.0,
        "use_carlos_roro": bool(use_carlos_roro),
        "carlos_roro_symbol": CARLOS,
    }
    for m in range(1, 13):
        sp[f"trade_month_{m}"] = m == int(calendar_month)
    return sp


def _params_subset(
    bitmask: int,
    entry_by_month: dict[int, int],
    hold_by_month: dict[int, int],
    *,
    use_carlos_roro: bool,
) -> dict[str, Any]:
    max_hold = max(int(hold_by_month[m]) for m in range(1, 13))
    sp: dict[str, Any] = {
        "direction": "long",
        "month_local_seasonal_windows": False,
        "hold_days": int(max_hold),
        "entry_day": 1,
        "entry_day_by_month": {m: int(entry_by_month[m]) for m in range(1, 13)},
        "hold_days_by_month": {m: int(hold_by_month[m]) for m in range(1, 13)},
        "simple_high_low_stop_loss": False,
        "simple_high_low_take_profit": False,
        "stop_loss_atr_multiple": 0.0,
        "take_profit_atr_multiple": 0.0,
        "use_carlos_roro": bool(use_carlos_roro),
        "carlos_roro_symbol": CARLOS,
    }
    for m in range(1, 13):
        sp[f"trade_month_{m}"] = bool((bitmask >> (m - 1)) & 1)
    return sp


def load_global_shared(
    *,
    start_date: str,
    end_date: str,
    cache_only: bool,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[
    dict[str, Any],
    pd.DatetimeIndex,
    pd.DatetimeIndex,
    np.ndarray,
    pd.Series,
    np.ndarray,
    np.ndarray,
    tuple[Any, Any, Any, Any, Any],
]:
    config_loader.load_config()
    gc = config_loader.GLOBAL_CONFIG
    if cache_only:
        gc.setdefault("data_source_config", {})["cache_only"] = True
    gc.update(
        {
            "portfolio_value": PORT,
            "commission_per_share": CPS,
            "commission_min_per_order": CMIN,
            "commission_max_percent_of_trade": CMAX,
            "slippage_bps": SLIP,
        }
    )

    probe_off = ScenarioNormalizer().normalize(
        scenario=_scenario_template(
            name="probe_roro_off",
            strategy_params=_params_single_month(6, 1, 5, use_carlos_roro=False),
            start_date=start_date,
            end_date=end_date,
        ),
        global_config=gc,
    )
    probe_on = ScenarioNormalizer().normalize(
        scenario=_scenario_template(
            name="probe_roro_on",
            strategy_params=_params_single_month(6, 1, 5, use_carlos_roro=True),
            start_date=start_date,
            end_date=end_date,
        ),
        global_config=gc,
    )

    fetcher = DataFetcher(gc, create_data_source(gc))
    strat_mgr = StrategyManager()
    daily_ohlc, monthly_data, daily_closes = fetcher.prepare_data_for_backtesting(
        [probe_off, probe_on],
        strat_mgr.get_strategy,
    )
    data_cache = create_cache_manager()
    _rets_cached = data_cache.get_cached_returns(daily_closes, "full_period_returns")
    rets_full = (
        _rets_cached.to_frame()
        if isinstance(_rets_cached, pd.Series)
        else pd.DataFrame(_rets_cached)
    )

    close_s = daily_ohlc.xs("Close", level="Field", axis=1)["SPY"]
    ix = pd.DatetimeIndex(close_s.index)
    loc_ix = (
        pd.DatetimeIndex([pd.Timestamp(t).tz_convert(ix.tz).replace(tzinfo=None) for t in ix])
        if ix.tz is not None
        else ix
    )
    sel = (loc_ix >= start) & (loc_ix <= end)
    ix_f = ix[sel]
    loc_f = loc_ix[sel]
    close_a = close_s.loc[sel].to_numpy(float)
    rets = np.r_[0.0, close_a[1:] / close_a[:-1] - 1.0]
    bench_rets = pd.Series(rets, index=ix_f)

    risky_idx = pd.DatetimeIndex([d for d in daily_ohlc.index])
    naive_idx = (
        risky_idx.tz_convert(None)
        if getattr(risky_idx, "tz", None) is not None
        else pd.DatetimeIndex(risky_idx)
    )
    strat_roro = SeasonalSignalStrategy({"strategy_params": {"use_carlos_roro": True}})
    nu_view = None
    if isinstance(daily_ohlc.columns, pd.MultiIndex):
        flds = list(daily_ohlc.columns.get_level_values("Field").unique())
        cols = pd.MultiIndex.from_product([[CARLOS], flds], names=["Ticker", "Field"]).intersection(
            daily_ohlc.columns
        )
        if len(cols):
            nu_view = daily_ohlc[cols]

    carlos_off_full = np.zeros(len(ix), dtype=np.float64)
    if nu_view is not None and len(nu_view.columns):
        mask_naive = strat_roro._carlos_roro_risk_off_mask(naive_idx, nu_view)  # noqa: SLF001
        if mask_naive is not None:
            carlos_off_full = mask_naive.astype(np.float64)
    carlos_off_win = carlos_off_full[sel]

    shared = (daily_ohlc, monthly_data, daily_closes, rets_full, data_cache)
    return gc, ix_f, loc_f, close_a, bench_rets, rets, carlos_off_win, shared


def task_a_month_entry_hold(
    calendar_month: int,
    entry_day: int,
    hold_days: int,
) -> tuple[dict[int, int], dict[int, int]]:
    entry = {m: 1 for m in range(1, 13)}
    hold = {m: 5 for m in range(1, 13)}
    entry[int(calendar_month)] = int(entry_day)
    hold[int(calendar_month)] = int(hold_days)
    return entry, hold


def subset_entry_hold(
    entry_best: dict[int, int],
    hold_best: dict[int, int],
) -> tuple[dict[int, int], dict[int, int]]:
    return (
        {m: int(entry_best[m]) for m in range(1, 13)},
        {m: int(hold_best[m]) for m in range(1, 13)},
    )


def _nth_bday(year: int, month: int, n: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    last = (first + pd.offsets.MonthEnd(1)).normalize()
    bdays = pd.bdate_range(first, last)
    idx = n - 1 if n > 0 else n
    idx = max(min(idx, len(bdays) - 1), -len(bdays))
    return pd.Timestamp(bdays[idx])


def _entry_hold_allowed(
    strategy_params: dict[str, Any],
) -> tuple[dict[int, int], dict[int, int], dict[int, bool]]:
    entry_default = int(strategy_params.get("entry_day", 1))
    hold_default = int(strategy_params.get("hold_days", 5))
    raw_entry = strategy_params.get("entry_day_by_month", {}) or {}
    raw_hold = strategy_params.get("hold_days_by_month", {}) or {}
    entry = {m: int(raw_entry.get(m, raw_entry.get(str(m), entry_default))) for m in range(1, 13)}
    hold = {m: int(raw_hold.get(m, raw_hold.get(str(m), hold_default))) for m in range(1, 13)}
    allowed = {m: bool(strategy_params.get(f"trade_month_{m}", True)) for m in range(1, 13)}
    return entry, hold, allowed


def precompute_anchor_scan(
    loc_ix: pd.DatetimeIndex,
    entry_by_month: dict[int, int],
    hold_by_month: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, int]:
    max_hold = max(int(hold_by_month[m]) for m in range(1, 13))
    k_max = min(24, max(3, (max_hold + 10) // 15 + 2))
    n = len(loc_ix)
    month_at_k = np.zeros((n, k_max), dtype=np.int8)
    in_win = np.zeros((n, k_max), dtype=np.bool_)
    cache: dict[tuple[int, int, int, int], tuple[pd.Timestamp, pd.Timestamp]] = {}
    for i in range(n):
        d = pd.Timestamp(loc_ix[i])
        first = pd.Timestamp(year=d.year, month=d.month, day=1)
        for k in range(k_max):
            anchor = first - pd.DateOffset(months=k)
            y, m = int(anchor.year), int(anchor.month)
            ed = int(entry_by_month[m])
            hd = int(hold_by_month[m])
            key = (y, m, ed, hd)
            if key not in cache:
                ent = _nth_bday(y, m, ed)
                cache[key] = (ent, ent + pd.tseries.offsets.BDay(hd - 1))
            ent, wend = cache[key]
            month_at_k[i, k] = np.int8(m)
            in_win[i, k] = bool(ent <= d <= wend)
    return month_at_k, in_win, k_max


def allocate_from_scan(month_at_k: np.ndarray, in_win: np.ndarray, bitmask: int) -> np.ndarray:
    n, k_max = month_at_k.shape
    active = np.zeros(n, dtype=np.bool_)
    bits = int(bitmask)
    for k in range(k_max):
        m = month_at_k[:, k].astype(np.int32, copy=False)
        allowed_m = (bits >> (m - 1)) & 1
        hit = (~active) & in_win[:, k] & (allowed_m != 0)
        active |= hit
    return active.astype(np.float64)


def single_month_long_mask(
    loc_ix: pd.DatetimeIndex,
    *,
    calendar_month: int,
    entry_day: int,
    hold_days: int,
) -> np.ndarray:
    y0 = int(pd.Timestamp(loc_ix[0]).year)
    y1 = int(pd.Timestamp(loc_ix[-1]).year)
    active = np.zeros(len(loc_ix), dtype=np.bool_)
    cm = int(calendar_month)
    ed = int(entry_day)
    hd = int(hold_days)
    for y in range(y0 - 1, y1 + 1):
        ent = _nth_bday(y, cm, ed)
        wend = ent + pd.tseries.offsets.BDay(hd - 1)
        active |= (loc_ix >= ent) & (loc_ix < wend)
    return active.astype(np.float64)


def apply_roro_to_target(
    target: np.ndarray, *, carlos_off: np.ndarray, use_roro: bool
) -> np.ndarray:
    if not use_roro:
        return np.asarray(target, dtype=np.float64)
    clear = 1.0 - carlos_off.astype(np.float64)
    return np.asarray(target * clear, dtype=np.float64)


def eval_target_net_returns(
    target: np.ndarray,
    *,
    close: np.ndarray,
    rets: np.ndarray,
    ix: pd.DatetimeIndex,
    bench_rets: pd.Series,
) -> dict[str, Any]:
    from portfolio_backtester.numba_kernels import (
        detailed_commission_slippage_kernel,
        drifting_weights_returns_kernel,
    )

    weights = target.reshape(-1, 1).astype(np.float64)
    close_2d = close.reshape(-1, 1).astype(np.float64)
    price_mask = (np.isfinite(close_2d) & (close_2d > 0.0)).astype(np.bool_)
    rets_2d = rets.reshape(-1, 1).astype(np.float32)
    rets_mask = np.isfinite(rets_2d).astype(np.bool_)
    w_for_returns = pd.Series(target).shift(1).fillna(0.0).to_numpy(dtype=np.float32).reshape(-1, 1)
    gross = drifting_weights_returns_kernel(w_for_returns, rets_2d, rets_mask)
    costs, _ = detailed_commission_slippage_kernel(
        weights_current=weights,
        close_prices=close_2d,
        portfolio_value=PORT,
        commission_per_share=CPS,
        commission_min_per_order=CMIN,
        commission_max_percent=CMAX,
        slippage_bps=SLIP,
        price_mask=price_mask,
    )
    delta = np.abs(target - np.r_[0.0, target[:-1]])
    pr = pd.Series(gross - costs, index=ix)
    f = calculate_optimizer_metrics_fast(
        pr,
        bench_rets,
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
    return {
        "Sortino": float(f.get("Sortino", np.nan)),
        "Sharpe": float(f.get("Sharpe", np.nan)),
        "Calmar": float(f.get("Calmar", np.nan)),
        "Total Return": float(f.get("Total Return", np.nan)),
        "Max Drawdown": float(f.get("Max Drawdown", np.nan)),
        "Ann Return": float(f.get("Ann. Return", np.nan)),
        "Trades": int(np.count_nonzero(delta > 1e-12)),
        "Total_Cost_frac": float(costs.sum()),
    }


def run_detailed_runner(
    *,
    global_config: dict[str, Any],
    shared: tuple[Any, Any, Any, Any, Any],
    strategy_params: dict[str, Any],
    name: str,
    start_date: str,
    end_date: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.Series, dict[str, float]]:
    daily_ohlc, monthly_data, daily_closes, rets_full, data_cache = shared
    runner = BacktestRunner(global_config, data_cache, StrategyManager(), lambda: False)
    canon = ScenarioNormalizer().normalize(
        scenario=_scenario_template(
            name=name,
            strategy_params=strategy_params,
            start_date=start_date,
            end_date=end_date,
        ),
        global_config=global_config,
    )
    original_matrix = SeasonalSignalStrategy.generate_signal_matrix

    def _disable_signal_matrix(self: SeasonalSignalStrategy, *args: Any, **kwargs: Any) -> None:
        return None

    SeasonalSignalStrategy.generate_signal_matrix = _disable_signal_matrix  # type: ignore[method-assign]
    try:
        pr = runner.run_scenario(canon, monthly_data, daily_ohlc, rets_full, verbose=False)
    finally:
        SeasonalSignalStrategy.generate_signal_matrix = original_matrix  # type: ignore[method-assign]
    if pr is None or pr.empty:
        raise RuntimeError("BacktestRunner returned empty series")
    idx = pd.DatetimeIndex(pr.index)
    if idx.tz is not None:
        cmp = pd.DatetimeIndex(
            [pd.Timestamp(ts).tz_convert(idx.tz).replace(tzinfo=None) for ts in idx]
        )
    else:
        cmp = idx
    m = (cmp >= start) & (cmp <= end)
    pr = pr.loc[m].astype(float)
    close = daily_ohlc.xs("Close", level="Field", axis=1)["SPY"].reindex(pr.index).astype(float)
    bench = close.pct_change(fill_method=None).fillna(0.0)
    metrics_raw = calculate_optimizer_metrics_fast(
        pr,
        bench.astype(float),
        "SPY",
        risk_free_rets=None,
        requested_metrics={"Sortino", "Total Return", "Max Drawdown", "Ann. Return"},
    )
    metrics = {str(k): float(v) for k, v in metrics_raw.items()}
    return pr, metrics


def parity_max_abs_diff(
    *,
    sim_net: pd.Series,
    runner_net: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> float:
    idx_a = pd.DatetimeIndex(sim_net.index)
    idx_b = pd.DatetimeIndex(runner_net.index)
    if idx_a.tz is not None:
        cmp_a = pd.DatetimeIndex(
            [pd.Timestamp(ts).tz_convert(idx_a.tz).replace(tzinfo=None) for ts in idx_a]
        )
    else:
        cmp_a = idx_a
    if idx_b.tz is not None:
        cmp_b = pd.DatetimeIndex(
            [pd.Timestamp(ts).tz_convert(idx_b.tz).replace(tzinfo=None) for ts in idx_b]
        )
    else:
        cmp_b = idx_b
    ma = (cmp_a >= start) & (cmp_a <= end)
    mb = (cmp_b >= start) & (cmp_b <= end)
    a = sim_net.loc[ma]
    b = runner_net.loc[mb]
    common = a.index.intersection(b.index)
    return float((a.loc[common].astype(float) - b.loc[common].astype(float)).abs().max())


def _sanitize_json_obj(obj: object) -> object:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {str(k): _sanitize_json_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json_obj(v) for v in obj]
    return obj


def _run_task_a_for_month(
    month: int,
    *,
    loc_f: pd.DatetimeIndex,
    ix_f: pd.DatetimeIndex,
    close_a: np.ndarray,
    rets: np.ndarray,
    bench_rets: pd.Series,
    carlos_off: np.ndarray,
    roro_fixed: bool | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    best_s = float("-inf")
    best_pack: dict[str, Any] = {}
    roro_values = [False, True] if roro_fixed is None else [roro_fixed]
    for ed in range(1, 21):
        for hd in range(5, 21):
            for ur in roro_values:
                tgt0 = single_month_long_mask(
                    loc_f,
                    calendar_month=month,
                    entry_day=ed,
                    hold_days=hd,
                )
                tgt = apply_roro_to_target(tgt0, carlos_off=carlos_off, use_roro=ur)
                mets = eval_target_net_returns(
                    tgt, close=close_a, rets=rets, ix=ix_f, bench_rets=bench_rets
                )
                s = float(mets["Sortino"])
                row = {
                    "calendar_month": month,
                    "month_name": _MONTH_EN[month - 1],
                    "entry_day": ed,
                    "hold_days": hd,
                    "use_carlos_roro": ur,
                    **mets,
                }
                rows.append(row)
                if math.isfinite(s) and s > best_s:
                    best_s = s
                    best_pack = {
                        "calendar_month": month,
                        "month_name": _MONTH_EN[month - 1],
                        "entry_day": ed,
                        "hold_days": hd,
                        "use_carlos_roro": ur,
                        "Sortino": s,
                        **{k: v for k, v in mets.items() if k != "Sortino"},
                    }
    return rows, best_pack


def exhaust_subsets(
    *,
    month_at_k: np.ndarray,
    in_win: np.ndarray,
    ix_f: pd.DatetimeIndex,
    close_a: np.ndarray,
    rets: np.ndarray,
    bench_rets: pd.Series,
    carlos_off: np.ndarray,
    global_roro: bool,
    variant_label: str,
) -> tuple[list[dict[str, Any]], int, float]:
    best_mask = -1
    best_sort = float("-inf")
    out: list[dict[str, Any]] = []
    for mask in range(1, (1 << 12)):
        tgt0 = allocate_from_scan(month_at_k, in_win, mask)
        tgt = apply_roro_to_target(tgt0, carlos_off=carlos_off, use_roro=global_roro)
        mets = eval_target_net_returns(
            tgt, close=close_a, rets=rets, ix=ix_f, bench_rets=bench_rets
        )
        s = float(mets["Sortino"])
        out.append(
            {
                "variant": variant_label,
                "global_use_carlos_roro": global_roro,
                "bitmask": mask,
                "n_months": int(bin(mask).count("1")),
                "months_csv": ";".join(_MONTH_EN[i] for i in range(12) if (mask >> i) & 1),
                **mets,
            }
        )
        if math.isfinite(s) and s > best_sort:
            best_sort = s
            best_mask = mask
    return out, best_mask, best_sort


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=START_DEF)
    parser.add_argument("--end-date", default=END_DEF)
    parser.add_argument("--cache-only", action="store_true", default=True)
    parser.add_argument("--no-cache-only", dest="cache_only", action="store_false")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO / "dev" / "spy_intramonth_roro_grid_subset_ibkr_rf_off_2005_2024",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-task-a-unrestricted", action="store_true")
    parser.add_argument("--skip-task-a-off", action="store_true")
    parser.add_argument("--skip-task-a-on", action="store_true")
    parser.add_argument(
        "--skip-backtest-parity",
        action="store_true",
        help="Skip BacktestRunner vs fast-sim checks on sampled masks.",
    )
    parser.add_argument(
        "--skip-runner-top-n",
        action="store_true",
        help="Skip detailed BacktestRunner on top subset candidates.",
    )
    args = parser.parse_args()

    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    rng = random.Random(int(args.seed))

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    print("loading_data ...", flush=True)
    gc, ix_f, loc_f, close_a, bench_rets, rets, carlos_off, shared = load_global_shared(
        start_date=args.start_date,
        end_date=args.end_date,
        cache_only=args.cache_only,
        start=start,
        end=end,
    )
    print(f"loaded_bars n={len(ix_f)} carlos_off_days={int(carlos_off.sum())}", flush=True)

    task_a_all: list[dict[str, Any]] = []
    unrestricted_winners: dict[int, dict[str, Any]] = {}
    consistent_off_winners: dict[int, dict[str, Any]] = {}
    consistent_on_winners: dict[int, dict[str, Any]] = {}

    print("task_a_unrestricted ...", flush=True)
    if args.skip_task_a_unrestricted:
        print("task_a_unrestricted skipped", flush=True)
    else:
        pass
    for month in [] if args.skip_task_a_unrestricted else range(1, 13):
        rows, bp = _run_task_a_for_month(
            month,
            loc_f=loc_f,
            ix_f=ix_f,
            close_a=close_a,
            rets=rets,
            bench_rets=bench_rets,
            carlos_off=carlos_off,
            roro_fixed=None,
        )
        task_a_all.extend(rows)
        unrestricted_winners[month] = bp

    print("task_a_consistent_off ...", flush=True)
    if args.skip_task_a_off:
        print("task_a_consistent_off skipped", flush=True)
    for month in [] if args.skip_task_a_off else range(1, 13):
        rows, bp = _run_task_a_for_month(
            month,
            loc_f=loc_f,
            ix_f=ix_f,
            close_a=close_a,
            rets=rets,
            bench_rets=bench_rets,
            carlos_off=carlos_off,
            roro_fixed=False,
        )
        consistent_off_winners[month] = bp

    print("task_a_consistent_on ...", flush=True)
    if args.skip_task_a_on:
        print("task_a_consistent_on skipped", flush=True)
    for month in [] if args.skip_task_a_on else range(1, 13):
        rows, bp = _run_task_a_for_month(
            month,
            loc_f=loc_f,
            ix_f=ix_f,
            close_a=close_a,
            rets=rets,
            bench_rets=bench_rets,
            carlos_off=carlos_off,
            roro_fixed=True,
        )
        consistent_on_winners[month] = bp

    pd.DataFrame(task_a_all).to_csv(out_dir / "task_a_full_grid_metrics.csv", index=False)

    def pack_to_entry_hold(
        pack_by_month: dict[int, dict[str, Any]],
    ) -> tuple[dict[int, int], dict[int, int]]:
        ent = {m: int(pack_by_month[m]["entry_day"]) for m in range(1, 13)}
        hod = {m: int(pack_by_month[m]["hold_days"]) for m in range(1, 13)}
        return ent, hod

    if args.skip_task_a_unrestricted:
        unrestricted_winners = dict(consistent_on_winners or consistent_off_winners)
    if args.skip_task_a_off:
        consistent_off_winners = dict(consistent_on_winners or unrestricted_winners)
    if args.skip_task_a_on:
        consistent_on_winners = dict(consistent_off_winners or unrestricted_winners)
    ent_u, hod_u = pack_to_entry_hold(unrestricted_winners)
    ent_co, hod_co = pack_to_entry_hold(consistent_off_winners)
    ent_cn, hod_cn = pack_to_entry_hold(consistent_on_winners)

    per_month_table = []
    for m in range(1, 13):
        u = unrestricted_winners[m]
        per_month_table.append(
            {
                "calendar_month": m,
                "month_name": _MONTH_EN[m - 1],
                "unrestricted_entry": u["entry_day"],
                "unrestricted_hold": u["hold_days"],
                "unrestricted_use_carlos_roro": u["use_carlos_roro"],
                "unrestricted_Sortino": u["Sortino"],
                "consistent_off_entry": consistent_off_winners[m]["entry_day"],
                "consistent_off_hold": consistent_off_winners[m]["hold_days"],
                "consistent_off_Sortino": consistent_off_winners[m]["Sortino"],
                "consistent_on_entry": consistent_on_winners[m]["entry_day"],
                "consistent_on_hold": consistent_on_winners[m]["hold_days"],
                "consistent_on_Sortino": consistent_on_winners[m]["Sortino"],
            }
        )
    pd.DataFrame(per_month_table).to_csv(out_dir / "per_month_best_summary.csv", index=False)

    roro_pref_counts = {"False": 0, "True": 0}
    for m in range(1, 13):
        key = str(bool(unrestricted_winners[m]["use_carlos_roro"]))
        roro_pref_counts[key] = roro_pref_counts.get(key, 0) + 1

    print("task_b_subsets ...", flush=True)
    ent_sub_u, hd_sub_u = subset_entry_hold(ent_u, hod_u)
    month_at_u, in_win_u, _ = precompute_anchor_scan(loc_f, ent_sub_u, hd_sub_u)
    rows_u_off, best_m_u_off, best_s_u_off = exhaust_subsets(
        month_at_k=month_at_u,
        in_win=in_win_u,
        ix_f=ix_f,
        close_a=close_a,
        rets=rets,
        bench_rets=bench_rets,
        carlos_off=carlos_off,
        global_roro=False,
        variant_label="mixed_params_global_roro_off",
    )
    rows_u_on, best_m_u_on, best_s_u_on = exhaust_subsets(
        month_at_k=month_at_u,
        in_win=in_win_u,
        ix_f=ix_f,
        close_a=close_a,
        rets=rets,
        bench_rets=bench_rets,
        carlos_off=carlos_off,
        global_roro=True,
        variant_label="mixed_params_global_roro_on",
    )

    ent_sub_co, hd_sub_co = subset_entry_hold(ent_co, hod_co)
    month_at_co, in_win_co, _ = precompute_anchor_scan(loc_f, ent_sub_co, hd_sub_co)
    rows_co_off, best_m_co_off, best_s_co_off = exhaust_subsets(
        month_at_k=month_at_co,
        in_win=in_win_co,
        ix_f=ix_f,
        close_a=close_a,
        rets=rets,
        bench_rets=bench_rets,
        carlos_off=carlos_off,
        global_roro=False,
        variant_label="consistent_off",
    )

    ent_sub_cn, hd_sub_cn = subset_entry_hold(ent_cn, hod_cn)
    month_at_cn, in_win_cn, _ = precompute_anchor_scan(loc_f, ent_sub_cn, hd_sub_cn)
    rows_cn_on, best_m_cn_on, best_s_cn_on = exhaust_subsets(
        month_at_k=month_at_cn,
        in_win=in_win_cn,
        ix_f=ix_f,
        close_a=close_a,
        rets=rets,
        bench_rets=bench_rets,
        carlos_off=carlos_off,
        global_roro=True,
        variant_label="consistent_on",
    )

    all_subset_rows = rows_u_off + rows_u_on + rows_co_off + rows_cn_on
    pd.DataFrame(all_subset_rows).to_csv(out_dir / "task_b_all_subsets_metrics.csv", index=False)

    def top_k(rows: list[dict[str, Any]], k: int = 15) -> list[dict[str, Any]]:
        return sorted(
            (r for r in rows if math.isfinite(float(r["Sortino"]))),
            key=lambda r: float(r["Sortino"]),
            reverse=True,
        )[:k]

    summary_subset = {
        "mixed_params_global_roro_off": {"bitmask": best_m_u_off, "Sortino_fast": best_s_u_off},
        "mixed_params_global_roro_on": {"bitmask": best_m_u_on, "Sortino_fast": best_s_u_on},
        "consistent_off": {"bitmask": best_m_co_off, "Sortino_fast": best_s_co_off},
        "consistent_on": {"bitmask": best_m_cn_on, "Sortino_fast": best_s_cn_on},
    }

    parity_checks: list[dict[str, Any]] = []
    if not args.skip_backtest_parity:
        print("fast_vs_runner_sample ...", flush=True)
        checks = [
            ("mixed_off", best_m_u_off, ent_sub_u, hd_sub_u, False),
            ("mixed_on", best_m_u_on, ent_sub_u, hd_sub_u, True),
            ("consistent_off", best_m_co_off, ent_sub_co, hd_sub_co, False),
            ("consistent_on", best_m_cn_on, ent_sub_cn, hd_sub_cn, True),
        ]
        for label, mask, ent_d, hd_d, ur in checks:
            if mask < 1:
                continue
            month_at, in_win, _ = precompute_anchor_scan(loc_f, ent_d, hd_d)
            tgt0 = allocate_from_scan(month_at, in_win, mask)
            tgt = apply_roro_to_target(tgt0, carlos_off=carlos_off, use_roro=ur)
            lag_target = np.r_[0.0, tgt[:-1]]
            gross = lag_target * rets
            delta = np.abs(tgt - np.r_[0.0, tgt[:-1]])
            tv = delta * PORT
            shares = np.where((tv > 0) & np.isfinite(close_a) & (close_a > 0), tv / close_a, 0.0)
            comm = np.zeros_like(tv)
            nz = shares > 0
            comm[nz] = np.minimum(np.maximum(shares[nz] * CPS, CMIN), tv[nz] * CMAX)
            costs = (comm + tv * (SLIP / 10000.0)) / PORT
            sim_net = pd.Series(gross - costs, index=ix_f)
            sp = _params_subset(mask, ent_d, hd_d, use_carlos_roro=ur)
            try:
                run_net, run_m = run_detailed_runner(
                    global_config=gc,
                    shared=shared,
                    strategy_params=sp,
                    name=f"parity_{label}_{mask}",
                    start_date=args.start_date,
                    end_date=args.end_date,
                    start=start,
                    end=end,
                )
                mad = parity_max_abs_diff(sim_net=sim_net, runner_net=run_net, start=start, end=end)
                parity_checks.append(
                    {
                        "label": label,
                        "bitmask": mask,
                        "global_use_carlos_roro": ur,
                        "max_abs_return_diff": mad,
                        "sortino_runner": float(run_m.get("Sortino", np.nan)),
                    }
                )
            except Exception as exc:
                parity_checks.append({"label": label, "bitmask": mask, "error": str(exc)})

        rnd_masks = [int(rng.randrange(1, 1 << 12)) for _ in range(2)]
        for rm in dict.fromkeys(rnd_masks):
            month_at, in_win, _ = precompute_anchor_scan(loc_f, ent_sub_u, hd_sub_u)
            for ur in (False, True):
                tgt0 = allocate_from_scan(month_at, in_win, rm)
                tgt = apply_roro_to_target(tgt0, carlos_off=carlos_off, use_roro=ur)
                lag_target = np.r_[0.0, tgt[:-1]]
                gross = lag_target * rets
                delta = np.abs(tgt - np.r_[0.0, tgt[:-1]])
                tv = delta * PORT
                shares = np.where(
                    (tv > 0) & np.isfinite(close_a) & (close_a > 0), tv / close_a, 0.0
                )
                comm = np.zeros_like(tv)
                nz = shares > 0
                comm[nz] = np.minimum(np.maximum(shares[nz] * CPS, CMIN), tv[nz] * CMAX)
                costs = (comm + tv * (SLIP / 10000.0)) / PORT
                sim_net = pd.Series(gross - costs, index=ix_f)
                sp = _params_subset(rm, ent_sub_u, hd_sub_u, use_carlos_roro=ur)
                try:
                    run_net, _run_m = run_detailed_runner(
                        global_config=gc,
                        shared=shared,
                        strategy_params=sp,
                        name=f"parity_rand_{rm}_{ur}",
                        start_date=args.start_date,
                        end_date=args.end_date,
                        start=start,
                        end=end,
                    )
                    mad = parity_max_abs_diff(
                        sim_net=sim_net, runner_net=run_net, start=start, end=end
                    )
                    parity_checks.append(
                        {
                            "label": "random_mixed_params",
                            "bitmask": rm,
                            "global_use_carlos_roro": ur,
                            "max_abs_return_diff": mad,
                        }
                    )
                except Exception as exc:
                    parity_checks.append(
                        {
                            "label": "random_mixed_params",
                            "bitmask": rm,
                            "global_use_carlos_roro": ur,
                            "error": str(exc),
                        }
                    )

    runner_top_details: list[dict[str, Any]] = []
    if not args.skip_runner_top_n:
        print("backtest_runner_top_candidates ...", flush=True)
        top_specs = [
            ("mixed_global_off", best_m_u_off, ent_sub_u, hd_sub_u, False, rows_u_off),
            ("mixed_global_on", best_m_u_on, ent_sub_u, hd_sub_u, True, rows_u_on),
            ("consistent_off", best_m_co_off, ent_sub_co, hd_sub_co, False, rows_co_off),
            ("consistent_on", best_m_cn_on, ent_sub_cn, hd_sub_cn, True, rows_cn_on),
        ]
        for label, primary_mask, ent_d, hd_d, ur, row_pool in top_specs:
            pool_masks = [primary_mask] + [int(r["bitmask"]) for r in top_k(row_pool, k=5)[1:6]]
            seen: set[int] = set()
            for bm in pool_masks:
                if bm in seen or bm < 1:
                    continue
                seen.add(bm)
                sp = _params_subset(bm, ent_d, hd_d, use_carlos_roro=ur)
                try:
                    run_net, run_m = run_detailed_runner(
                        global_config=gc,
                        shared=shared,
                        strategy_params=sp,
                        name=f"top_{label}_{bm}",
                        start_date=args.start_date,
                        end_date=args.end_date,
                        start=start,
                        end=end,
                    )
                    runner_top_details.append(
                        {
                            "label": label,
                            "bitmask": bm,
                            "global_use_carlos_roro": ur,
                            **{
                                k: float(run_m.get(k, np.nan))
                                for k in ("Sortino", "Total Return", "Max Drawdown", "Ann. Return")
                            },
                            "runner_daily_bars": int(run_net.shape[0]),
                        }
                    )
                except Exception as exc:
                    runner_top_details.append(
                        {
                            "label": label,
                            "bitmask": bm,
                            "global_use_carlos_roro": ur,
                            "error": str(exc),
                        }
                    )

    doc = {
        "period": {"start": args.start_date, "end": args.end_date},
        "carlos_symbol": CARLOS,
        "costs": {
            "portfolio_value": PORT,
            "commission_per_share": CPS,
            "commission_min_per_order": CMIN,
            "commission_max_percent_of_trade": CMAX,
            "slippage_bps": SLIP,
        },
        "risk_free": "off",
        "carlos_risk_off_sessions_in_window": int(carlos_off.sum()),
        "unrestricted_month_use_carlos_roro_counts": roro_pref_counts,
        "subset_best_fast": summary_subset,
        "runner_top_candidates": runner_top_details,
        "parity_checks_sample": parity_checks,
        "elapsed_seconds": time.perf_counter() - t0,
        "limitations": [
            "Single global use_carlos_roro applies to full strategy; per-month optimal RoRo flags from Task A cannot all be honored simultaneously.",
            "mixed_params_global_roro_* uses entry/hold from unrestricted per-month winners (RoRo chosen per month in isolation).",
            "Fast sim uses bar_close lag on 0/1 targets + IBKR-style costs; parity_sample compares vs BacktestRunner canonical path.",
            "4095 subsets × 4 variant tracks recorded in task_b_all_subsets_metrics.csv.",
        ],
    }
    (out_dir / "pipeline_summary.json").write_text(
        json.dumps(_sanitize_json_obj(doc), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    def write_tsv(path: Path, bitmask: int, ent: dict[int, int], hod: dict[int, int]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as f:
            f.write("calendar_month\tmonth_name\tactive\tentry_day\thold_days\n")
            for m in range(1, 13):
                act = bool((bitmask >> (m - 1)) & 1)
                f.write(f"{m}\t{_MONTH_EN[m - 1]}\t{int(act)}\t{ent[m]}\t{hod[m]}\n")

    write_tsv(out_dir / "best_subset_mixed_global_roro_off.tsv", best_m_u_off, ent_sub_u, hd_sub_u)
    write_tsv(out_dir / "best_subset_mixed_global_roro_on.tsv", best_m_u_on, ent_sub_u, hd_sub_u)
    write_tsv(out_dir / "best_subset_consistent_off.tsv", best_m_co_off, ent_sub_co, hd_sub_co)
    write_tsv(out_dir / "best_subset_consistent_on.tsv", best_m_cn_on, ent_sub_cn, hd_sub_cn)

    pd.DataFrame(top_k(rows_u_off, 25)).to_csv(
        out_dir / "top_subsets_mixed_global_off.csv", index=False
    )
    pd.DataFrame(top_k(rows_u_on, 25)).to_csv(
        out_dir / "top_subsets_mixed_global_on.csv", index=False
    )
    pd.DataFrame(top_k(rows_co_off, 25)).to_csv(
        out_dir / "top_subsets_consistent_off.csv", index=False
    )
    pd.DataFrame(top_k(rows_cn_on, 25)).to_csv(
        out_dir / "top_subsets_consistent_on.csv", index=False
    )

    print(json.dumps(_sanitize_json_obj(doc), indent=2, allow_nan=False))
    print(f"WROTE_OUT_DIR={out_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
