"""Task A+B: SPY intramonth seasonal — detailed-path Sortino (legacy RF-off), IBKR-like costs.

Task A: For each calendar month independently, grid entry_day 1..20 and hold_days 5..20 with
only that month's ``trade_month_*`` enabled; objective is Sortino from net daily returns.

Task B: Fix per-month (entry, hold) at Task A winners; exhaust non-empty month subsets (4095).
Allocation each day follows production ``SeasonalSignalStrategy._resolve_active_entry_date`` /
``generate_signals`` semantics (first-anchor resolution), not ``generate_signal_matrix`` union.

Costs / period match ``dev/run_spy_intramonth_detailed_backtest.py`` defaults.

Verification: compares max |Δr| vs ``BacktestRunner`` with ``generate_signal_matrix`` disabled
for the global best bitmask and for two deterministic pseudo-random masks.
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import csv
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


def load_global_and_ohlc(
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

    probe = ScenarioNormalizer().normalize(
        scenario=_scenario_template(
            name="slice_probe",
            strategy_params=_params_single_month(6, 1, 5),
            start_date=start_date,
            end_date=end_date,
        ),
        global_config=gc,
    )
    fetcher = DataFetcher(gc, create_data_source(gc))
    strat_mgr = StrategyManager()
    daily_ohlc, monthly_data, daily_closes = fetcher.prepare_data_for_backtesting(
        [probe], strat_mgr.get_strategy
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

    shared = (daily_ohlc, monthly_data, daily_closes, rets_full, data_cache)
    return gc, ix_f, loc_f, close_a, bench_rets, rets, shared


def _params_single_month(
    calendar_month: int,
    entry_day: int,
    hold_days: int,
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
    }
    for m in range(1, 13):
        sp[f"trade_month_{m}"] = m == int(calendar_month)
    return sp


def _params_subset(
    bitmask: int,
    entry_by_month: dict[int, int],
    hold_by_month: dict[int, int],
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
    }
    for m in range(1, 13):
        sp[f"trade_month_{m}"] = bool((bitmask >> (m - 1)) & 1)
    return sp


def single_month_long_mask(
    loc_ix: pd.DatetimeIndex,
    *,
    calendar_month: int,
    entry_day: int,
    hold_days: int,
) -> np.ndarray:
    """Long mask for one intramonth cycle per year (inclusive window end).

    When only a single ``trade_month_*`` is enabled (Task A), this matches
    ``SeasonalSignalStrategy._resolve_active_entry_date`` / first-anchor scan:
    disallowed anchor months are skipped, so the first qualifying in-window month
    is always the configured calendar month.

    """
    y0 = int(pd.Timestamp(loc_ix[0]).year)
    y1 = int(pd.Timestamp(loc_ix[-1]).year)
    active = np.zeros(len(loc_ix), dtype=np.bool_)
    cm = int(calendar_month)
    ed = int(entry_day)
    hd = int(hold_days)
    for y in range(y0 - 1, y1 + 1):
        ent = _nth_bday(y, cm, ed)
        wend = ent + pd.tseries.offsets.BDay(hd - 1)
        active |= (loc_ix >= ent) & (loc_ix <= wend)
    return active.astype(np.float64)


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


def detailed_allocation_mask(
    strategy_params: dict[str, Any],
    loc_ix: pd.DatetimeIndex,
) -> np.ndarray:
    entry, hold, allowed = _entry_hold_allowed(strategy_params)
    bitmask = 0
    for m in range(1, 13):
        if allowed[m]:
            bitmask |= 1 << (m - 1)
    month_at_k, in_win, _ = precompute_anchor_scan(loc_ix, entry, hold)
    return allocate_from_scan(month_at_k, in_win, bitmask)


def _assert_task_a_fast_mask_parity(
    *,
    loc_ix: pd.DatetimeIndex,
    rng: random.Random,
) -> float:
    worst = 0.0
    for _ in range(24):
        m = rng.randint(1, 12)
        ed = rng.randint(1, 20)
        hd = rng.randint(5, 20)
        ent_map, hd_map = task_a_month_entry_hold(m, ed, hd)
        month_at_k, in_win, _ = precompute_anchor_scan(loc_ix, ent_map, hd_map)
        slow = allocate_from_scan(month_at_k, in_win, 1 << (m - 1))
        fast = single_month_long_mask(loc_ix, calendar_month=m, entry_day=ed, hold_days=hd)
        worst = max(worst, float(np.max(np.abs(slow - fast))))
    return worst


def eval_target_net_returns(
    target: np.ndarray,
    *,
    close: np.ndarray,
    rets: np.ndarray,
    ix: pd.DatetimeIndex,
    bench_rets: pd.Series,
) -> dict[str, Any]:
    lag_target = np.r_[0.0, target[:-1]]
    gross = lag_target * rets
    delta = np.abs(target - np.r_[0.0, target[:-1]])
    tv = delta * PORT
    shares = np.where((tv > 0) & np.isfinite(close) & (close > 0), tv / close, 0.0)
    comm = np.zeros_like(tv)
    nz = shares > 0
    comm[nz] = np.minimum(np.maximum(shares[nz] * CPS, CMIN), tv[nz] * CMAX)
    costs = (comm + tv * (SLIP / 10000.0)) / PORT
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


def _period_mask(index: pd.Index, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        cmp = pd.DatetimeIndex(
            [pd.Timestamp(ts).tz_convert(idx.tz).replace(tzinfo=None) for ts in idx]
        )
    else:
        cmp = idx
    return pd.Series((cmp >= start) & (cmp <= end), index=index)


def parity_max_abs_diff(
    *,
    sim_net: pd.Series,
    runner_net: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> float:
    a = sim_net.loc[_period_mask(sim_net.index, start, end).to_numpy()]
    b = runner_net.loc[_period_mask(runner_net.index, start, end).to_numpy()]
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2005-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--cache-only", action="store_true", default=True)
    parser.add_argument("--no-cache-only", dest="cache_only", action="store_false")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO / "dev" / "spy_intramonth_detailed_ibkr_rf_off_2005_2024",
    )
    parser.add_argument("--skip-task-a", action="store_true")
    parser.add_argument(
        "--task-a-csv",
        type=Path,
        default=None,
        help="Load per-month best from CSV instead of Task A (calendar_month,entry_day,hold_days,Sortino).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-backtest-parity",
        action="store_true",
        help="Skip BacktestRunner vs sim return checks (faster; metrics already from fast sim).",
    )
    parser.add_argument(
        "--skip-task-a-fast-verify",
        action="store_true",
        help="Skip randomized Task-A single-month parity (fast vs anchor-scan allocator).",
    )
    args = parser.parse_args()

    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    rng = random.Random(int(args.seed))

    print("loading_data ...", flush=True)
    gc, ix_f, loc_f, close_a, bench_rets, rets, shared = load_global_and_ohlc(
        start_date=args.start_date,
        end_date=args.end_date,
        cache_only=args.cache_only,
        start=start,
        end=end,
    )
    print(f"loaded_bars n={len(ix_f)}", flush=True)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t_all0 = time.perf_counter()

    month_rows: list[dict[str, Any]] = []
    entry_best: dict[int, int] = {}
    hold_best: dict[int, int] = {}

    task_a_fast_parity_worst: float | None = None
    if args.task_a_csv is not None:
        with args.task_a_csv.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                m = int(row["calendar_month"])
                entry_best[m] = int(row["entry_day"])
                hold_best[m] = int(row["hold_days"])
        if set(entry_best.keys()) != set(range(1, 13)):
            raise ValueError("task-a-csv must define months 1..12")
    elif not args.skip_task_a:
        if not args.skip_task_a_fast_verify:
            task_a_fast_parity_worst = _assert_task_a_fast_mask_parity(loc_ix=loc_f, rng=rng)
            print(f"task_a_fast_parity_worst={task_a_fast_parity_worst}", flush=True)
            if task_a_fast_parity_worst > 1e-9:
                raise RuntimeError(
                    "Task A fast mask parity failed vs precompute_anchor_scan/allocate_from_scan"
                )
        t_a0 = time.perf_counter()
        print("task_a_start", flush=True)
        for month in range(1, 13):
            best_s = float("-inf")
            best_e = 1
            best_h = 5
            for ed in range(1, 21):
                for hd in range(5, 21):
                    tgt = single_month_long_mask(
                        loc_f,
                        calendar_month=month,
                        entry_day=ed,
                        hold_days=hd,
                    )
                    mets = eval_target_net_returns(
                        tgt, close=close_a, rets=rets, ix=ix_f, bench_rets=bench_rets
                    )
                    s = float(mets["Sortino"])
                    row = {
                        "calendar_month": month,
                        "month_name": _MONTH_EN[month - 1],
                        "entry_day": ed,
                        "hold_days": hd,
                        **mets,
                    }
                    month_rows.append(row)
                    if math.isfinite(s) and s > best_s:
                        best_s = s
                        best_e = ed
                        best_h = hd
            entry_best[month] = best_e
            hold_best[month] = best_h
        t_a1 = time.perf_counter()
        pd.DataFrame(month_rows).to_csv(out_dir / "task_a_full_grid_metrics.csv", index=False)
        best_summary = []
        for month in range(1, 13):
            sub = [r for r in month_rows if int(r["calendar_month"]) == month]
            winner = max(
                sub,
                key=lambda r: (
                    float(r["Sortino"]) if math.isfinite(float(r["Sortino"])) else float("-inf")
                ),
            )
            best_summary.append(winner)
        pd.DataFrame(best_summary).to_csv(out_dir / "per_month_best.csv", index=False)
        print(f"task_a_done_seconds={t_a1 - t_a0:.3f}", flush=True)
    else:
        raise SystemExit("Provide --task-a-csv or run Task A (default).")

    subset_rows: list[dict[str, Any]] = []
    t_b0 = time.perf_counter()
    print("task_b_start", flush=True)
    best_mask = -1
    best_sort = float("-inf")
    ent_sub, hd_sub = subset_entry_hold(entry_best, hold_best)
    month_at_k_b, in_win_b, _ = precompute_anchor_scan(loc_f, ent_sub, hd_sub)
    for mask in range(1, (1 << 12)):
        tgt = allocate_from_scan(month_at_k_b, in_win_b, mask)
        mets = eval_target_net_returns(
            tgt, close=close_a, rets=rets, ix=ix_f, bench_rets=bench_rets
        )
        subset_rows.append(
            {
                "bitmask": mask,
                "n_months": int(bin(mask).count("1")),
                "months_csv": ";".join(_MONTH_EN[i] for i in range(12) if (mask >> i) & 1),
                **mets,
            }
        )
        s = float(mets["Sortino"])
        if math.isfinite(s) and s > best_sort:
            best_sort = s
            best_mask = mask
    t_b1 = time.perf_counter()
    print(f"task_b_done_seconds={t_b1 - t_b0:.3f}", flush=True)

    pd.DataFrame(subset_rows).to_csv(out_dir / "task_b_all_subsets_metrics.csv", index=False)
    top_alternatives = sorted(
        (r for r in subset_rows if math.isfinite(float(r["Sortino"]))),
        key=lambda r: float(r["Sortino"]),
        reverse=True,
    )[:20]

    verify_masks = [int(best_mask), int(rng.randrange(1, 1 << 12)), int(rng.randrange(1, 1 << 12))]
    verify_masks = list(dict.fromkeys(verify_masks))
    parity_checks: list[dict[str, Any]] = []
    if not args.skip_backtest_parity:
        print("backtest_parity_start", flush=True)
        for vm in verify_masks:
            sp = _params_subset(vm, entry_best, hold_best)
            tgt = allocate_from_scan(month_at_k_b, in_win_b, vm)
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
            try:
                run_net, run_m = run_detailed_runner(
                    global_config=gc,
                    shared=shared,
                    strategy_params=sp,
                    name=f"parity_mask_{vm}",
                    start_date=args.start_date,
                    end_date=args.end_date,
                    start=start,
                    end=end,
                )
                mad = parity_max_abs_diff(sim_net=sim_net, runner_net=run_net, start=start, end=end)
                parity_checks.append(
                    {
                        "bitmask": vm,
                        "max_abs_return_diff": mad,
                        "sortino_sim_common": float(
                            calculate_optimizer_metrics_fast(
                                sim_net,
                                bench_rets,
                                "SPY",
                                risk_free_rets=None,
                                requested_metrics={"Sortino"},
                            ).get("Sortino", np.nan)
                        ),
                        "sortino_runner": float(run_m.get("Sortino", np.nan)),
                    }
                )
            except Exception as exc:
                parity_checks.append({"bitmask": vm, "error": str(exc)})
        print("backtest_parity_done", flush=True)

    best_row = next((r for r in subset_rows if int(r["bitmask"]) == best_mask), None)
    doc = {
        "period": {"start": args.start_date, "end": args.end_date},
        "costs": {
            "portfolio_value": PORT,
            "commission_per_share": CPS,
            "commission_min_per_order": CMIN,
            "commission_max_percent_of_trade": CMAX,
            "slippage_bps": SLIP,
        },
        "risk_free": "off (scenario extras + risk_free_rets=None in Sortino)",
        "detail_path": (
            "precompute_anchor_scan + allocate_from_scan matches "
            "SeasonalSignalStrategy._resolve_active_entry_date scan order (k=0 first)"
        ),
        "task_b_elapsed_seconds": t_b1 - t_b0,
        "total_elapsed_seconds": time.perf_counter() - t_all0,
        "best_bitmask": best_mask,
        "best_sortino_fast_first_anchor_scan": best_sort,
        "best_row": best_row,
        "top_alternatives": top_alternatives,
        "per_month_best": {
            str(m): {"entry_day": entry_best[m], "hold_days": hold_best[m]} for m in range(1, 13)
        },
        "task_a_fast_parity_worst": task_a_fast_parity_worst,
        "parity_runner_vs_sim": parity_checks,
        "limitations": [
            "PnL uses bar_close lag on 0/1 targets + IBKR-style per-change costs (meta script pattern).",
            "Optional BacktestRunner parity: omit --skip-backtest-parity (3 full detailed runs; slow).",
            "Task A uses single_month_long_mask after randomized equivalence check vs first-anchor scan.",
            "Task B does one precompute_anchor_scan then 4095 allocate_from_scan passes.",
        ],
    }
    (out_dir / "pipeline_summary.json").write_text(
        json.dumps(_sanitize_json_obj(doc), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    with (out_dir / "best_subset_month_params.tsv").open("w", encoding="utf-8", newline="\n") as f:
        f.write("calendar_month\tmonth_name\tactive\tentry_day\thold_days\n")
        for m in range(1, 13):
            act = bool((best_mask >> (m - 1)) & 1)
            f.write(f"{m}\t{_MONTH_EN[m - 1]}\t{int(act)}\t{entry_best[m]}\t{hold_best[m]}\n")

    print(json.dumps(_sanitize_json_obj(doc), indent=2, allow_nan=False))
    print(f"WROTE_OUT_DIR={out_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
