"""Per calendar month: grid (entry_day, hold_days) maximizing full-period Sortino.

Default grid: entry_day 1..20, hold_days 5..20 (IBKR-like costs, 0.5 bps slip). Vectorized
single-asset path (no SL/TP/ATR exits). Objective: ``calculate_optimizer_metrics_fast`` with
``risk_free_rets=None``. Optional framework parity vs ``BacktestRunner`` on a few cells.
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

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


def nth_bday(year: int, month: int, n: int) -> pd.Timestamp:
    b = pd.bdate_range(
        pd.Timestamp(year, month, 1),
        (pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(1)).normalize(),
    )
    i = n - 1 if n > 0 else n
    i = max(min(i, len(b) - 1), -len(b))
    return pd.Timestamp(b[i])


def atr_series(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, lookback: int = 21
) -> np.ndarray:
    prev = np.r_[close[0], close[:-1]]
    tr = np.maximum.reduce([high - low, np.abs(high - prev), np.abs(low - prev)])
    return cast(np.ndarray, pd.Series(tr).rolling(lookback, min_periods=1).mean().to_numpy())


def _scenario_fetch_probe(
    *,
    start_date: str,
    end_date: str,
    hold_days: int,
    entry_day: int,
) -> dict[str, Any]:
    sp: dict[str, Any] = {
        "direction": "long",
        "month_local_seasonal_windows": False,
        "hold_days": int(hold_days),
        "entry_day": int(entry_day),
    }
    for m in range(1, 13):
        sp[f"trade_month_{m}"] = m == 6
    sp["simple_high_low_stop_loss"] = False
    sp["simple_high_low_take_profit"] = False
    sp["stop_loss_atr_multiple"] = 0.0
    sp["take_profit_atr_multiple"] = 0.0
    return {
        "name": "grid_data_probe",
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
        "strategy_params": sp,
    }


def load_ohlc_slice(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    start_date: str,
    end_date: str,
    cache_only: bool,
) -> tuple[
    pd.DatetimeIndex,
    pd.DatetimeIndex,
    np.ndarray,
    np.ndarray,
    np.ndarray,
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
    norm = ScenarioNormalizer()
    canon = norm.normalize(
        scenario=_scenario_fetch_probe(
            start_date=start_date,
            end_date=end_date,
            hold_days=10,
            entry_day=1,
        ),
        global_config=gc,
    )
    ohlc, _, _ = DataFetcher(gc, create_data_source(gc)).prepare_data_for_backtesting(
        [canon], StrategyManager().get_strategy
    )
    close_s = ohlc.xs("Close", level="Field", axis=1)["SPY"]
    high_s = ohlc.xs("High", level="Field", axis=1)["SPY"]
    low_s = ohlc.xs("Low", level="Field", axis=1)["SPY"]
    ix = pd.DatetimeIndex(close_s.index)
    loc_ix = (
        pd.DatetimeIndex([pd.Timestamp(t).tz_convert(ix.tz).replace(tzinfo=None) for t in ix])
        if ix.tz is not None
        else ix
    )
    sel = (loc_ix >= start) & (loc_ix <= end)
    return (
        ix[sel],
        loc_ix[sel],
        close_s.loc[sel].to_numpy(float),
        high_s.loc[sel].to_numpy(float),
        low_s.loc[sel].to_numpy(float),
    )


def build_cycles(
    cal_month: int,
    entry_day: int,
    hold_days: int,
    loc_ix: pd.DatetimeIndex,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    cycles: list[dict[str, Any]] = []
    for y in range(start.year - 1, end.year + 1):
        ent = nth_bday(y, cal_month, entry_day)
        exit_end = ent + BDay(hold_days - 1)
        if exit_end < start or ent > end:
            continue
        pos = np.where((loc_ix >= ent) & (loc_ix <= exit_end))[0]
        if len(pos) == 0:
            continue
        ei_arr = np.where(loc_ix == ent)[0]
        ei = int(ei_arr[0]) if len(ei_arr) else None
        cycles.append(
            {
                "entry": ent,
                "month": cal_month,
                "pos": pos,
                "ei": ei,
                "entry_px": close[ei] if ei is not None else math.nan,
                "atr": atr[ei] if ei is not None else math.nan,
                "cl": close[pos],
                "ph": high[np.maximum(pos - 1, 0)],
                "pl": low[np.maximum(pos - 1, 0)],
            }
        )
    cycles.sort(key=lambda x: x["entry"])
    resolved = np.full(len(close), -1, dtype=np.int32)
    for ci, c in enumerate(cycles):
        for p in c["pos"]:
            if resolved[p] < 0 or c["entry"] > cycles[int(resolved[p])]["entry"]:
                resolved[p] = ci
    return cycles, resolved


def eval_vectorized(
    cycles: list[dict[str, Any]],
    resolved: np.ndarray,
    close: np.ndarray,
    rets: np.ndarray,
    bench_rets: pd.Series,
    ix: pd.DatetimeIndex,
) -> dict[str, Any]:
    target = np.zeros(len(close))
    for ci, c in enumerate(cycles):
        pos = c["pos"]
        hit = np.zeros(len(pos), dtype=bool)
        active = pos[: int(np.argmax(hit))] if hit.any() else pos
        mask = resolved[active] == ci
        target[active[mask]] = 1.0
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


def _scenario_for_framework(
    *,
    name: str,
    start_date: str,
    end_date: str,
    cal_month: int,
    entry_day: int,
    hold_days: int,
) -> dict[str, Any]:
    sp: dict[str, Any] = {
        "direction": "long",
        "month_local_seasonal_windows": False,
        "hold_days": int(hold_days),
        "entry_day": int(entry_day),
        "simple_high_low_stop_loss": False,
        "simple_high_low_take_profit": False,
        "stop_loss_atr_multiple": 0.0,
        "take_profit_atr_multiple": 0.0,
    }
    for m in range(1, 13):
        sp[f"trade_month_{m}"] = m == cal_month
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
        "strategy_params": sp,
    }


def run_framework_returns(
    *,
    cal_month: int,
    entry_day: int,
    hold_days: int,
    start_date: str,
    end_date: str,
    gc: dict[str, Any],
    runner: BacktestRunner,
    norm: ScenarioNormalizer,
    monthly_data: pd.DataFrame,
    daily_ohlc: pd.DataFrame,
    rets_full: pd.DataFrame,
) -> pd.Series | None:
    raw = _scenario_for_framework(
        name="parity",
        start_date=start_date,
        end_date=end_date,
        cal_month=cal_month,
        entry_day=entry_day,
        hold_days=hold_days,
    )
    canon = norm.normalize(scenario=raw, global_config=gc)
    return runner.run_scenario(canon, monthly_data, daily_ohlc, rets_full, verbose=False)


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
    parser.add_argument("--entry-min", type=int, default=1)
    parser.add_argument("--entry-max", type=int, default=20)
    parser.add_argument("--hold-min", type=int, default=5)
    parser.add_argument("--hold-max", type=int, default=20)
    parser.add_argument("--cache-only", action="store_true", default=True)
    parser.add_argument("--no-cache-only", dest="cache_only", action="store_false")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO / "dev" / "spy_intramonth_per_month_grid_ibkr_legacy_hd5_20_2005_2024",
    )
    parser.add_argument("--parity-samples", type=int, default=6)
    parser.add_argument("--no-parity", action="store_true")
    args = parser.parse_args()

    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    ix, loc_ix, close, high, low = load_ohlc_slice(
        start=start,
        end=end,
        start_date=args.start_date,
        end_date=args.end_date,
        cache_only=args.cache_only,
    )
    rets = np.r_[0.0, close[1:] / close[:-1] - 1.0]
    atr = atr_series(high, low, close)
    bench_rets = pd.Series(rets, index=ix)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    per_month_best: dict[int, dict[str, Any]] = {}
    t0 = time.perf_counter()

    for cal_m in range(1, 13):
        for e in range(args.entry_min, args.entry_max + 1):
            for h in range(args.hold_min, args.hold_max + 1):
                cycles, resolved = build_cycles(
                    cal_m, e, h, loc_ix, close, high, low, atr, start, end
                )
                if not cycles:
                    metrics = {
                        "Sortino": float("nan"),
                        "Sharpe": float("nan"),
                        "Calmar": float("nan"),
                        "Total Return": float("nan"),
                        "Max Drawdown": float("nan"),
                        "Ann Return": float("nan"),
                        "Trades": 0,
                        "Total_Cost_frac": 0.0,
                    }
                else:
                    metrics = eval_vectorized(cycles, resolved, close, rets, bench_rets, ix)
                row = {
                    "calendar_month": cal_m,
                    "month_name": _MONTH_EN[cal_m - 1],
                    "entry_day": e,
                    "hold_days": h,
                    **metrics,
                }
                rows.append(row)
                s = metrics["Sortino"]
                prev = per_month_best.get(cal_m)
                if prev is None or (
                    math.isfinite(s) and (not math.isfinite(prev["Sortino"]) or s > prev["Sortino"])
                ):
                    per_month_best[cal_m] = row

    elapsed = time.perf_counter() - t0

    parity_out: list[dict[str, Any]] = []
    if not args.no_parity and args.parity_samples > 0:
        config_loader.load_config()
        gc = config_loader.GLOBAL_CONFIG
        if args.cache_only:
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
        strat_mgr = StrategyManager()
        fetcher = DataFetcher(gc, create_data_source(gc))
        data_cache = create_cache_manager()
        runner = BacktestRunner(gc, data_cache, strat_mgr, lambda: False)
        norm = ScenarioNormalizer()
        probe = norm.normalize(
            scenario=_scenario_for_framework(
                name="parity_probe",
                start_date=args.start_date,
                end_date=args.end_date,
                cal_month=3,
                entry_day=5,
                hold_days=5,
            ),
            global_config=gc,
        )
        daily_ohlc, monthly_data, daily_closes = fetcher.prepare_data_for_backtesting(
            [probe], strat_mgr.get_strategy
        )
        _rets_cached = data_cache.get_cached_returns(daily_closes, "full_period_returns")
        rets_full = (
            _rets_cached.to_frame()
            if isinstance(_rets_cached, pd.Series)
            else pd.DataFrame(_rets_cached)
        )

        canned = [
            (1, 4, 7),
            (2, 11, 3),
            (4, 19, 12),
            (6, 6, 5),
            (8, 14, 8),
            (11, 2, 15),
            (12, 17, 4),
            (3, 9, 11),
            (7, 1, 20),
            (10, 20, 2),
        ][: max(0, args.parity_samples)]

        for cal_m, e, h in canned:
            pr_fw = run_framework_returns(
                cal_month=cal_m,
                entry_day=e,
                hold_days=h,
                start_date=args.start_date,
                end_date=args.end_date,
                gc=gc,
                runner=runner,
                norm=norm,
                monthly_data=monthly_data,
                daily_ohlc=daily_ohlc,
                rets_full=rets_full,
            )
            cycles, resolved = build_cycles(cal_m, e, h, loc_ix, close, high, low, atr, start, end)
            if pr_fw is None or pr_fw.empty or not cycles:
                parity_out.append(
                    {
                        "calendar_month": cal_m,
                        "entry_day": e,
                        "hold_days": h,
                        "skipped": True,
                        "reason": "framework_empty_or_no_cycles_vectorized",
                    }
                )
                continue
            metrics = eval_vectorized(cycles, resolved, close, rets, bench_rets, ix)
            target = np.zeros(len(close))
            for ci, c in enumerate(cycles):
                pos = c["pos"]
                hit = np.zeros(len(pos), dtype=bool)
                active = pos[: int(np.argmax(hit))] if hit.any() else pos
                mask = resolved[active] == ci
                target[active[mask]] = 1.0
            delta = np.abs(target - np.r_[0.0, target[:-1]])
            tv = delta * PORT
            close_a = close.astype(float)
            shares = np.where((tv > 0) & np.isfinite(close_a) & (close_a > 0), tv / close_a, 0.0)
            comm = np.zeros_like(tv)
            nz = shares > 0
            comm[nz] = np.minimum(np.maximum(shares[nz] * CPS, CMIN), tv[nz] * CMAX)
            costs = (comm + tv * (SLIP / 10000.0)) / PORT
            lag_target = np.r_[0.0, target[:-1]]
            vec_net = pd.Series(lag_target * rets - costs, index=ix)
            common = pr_fw.index.intersection(vec_net.index)
            dvec = vec_net.loc[common].astype(float)
            dfw = pr_fw.loc[common].astype(float)
            diff = float((dvec - dfw).abs().max())
            bcom = bench_rets.reindex(common).astype(float)
            sort_fw = float(
                calculate_optimizer_metrics_fast(
                    dfw,
                    bcom,
                    "SPY",
                    risk_free_rets=None,
                    requested_metrics={"Sortino"},
                ).get("Sortino", np.nan)
            )
            sort_vec_common = float(
                calculate_optimizer_metrics_fast(
                    dvec,
                    bcom,
                    "SPY",
                    risk_free_rets=None,
                    requested_metrics={"Sortino"},
                ).get("Sortino", np.nan)
            )
            parity_out.append(
                {
                    "calendar_month": cal_m,
                    "entry_day": e,
                    "hold_days": h,
                    "max_abs_ret_diff": diff,
                    "sortino_vectorized_full_window": metrics["Sortino"],
                    "sortino_vectorized_common_index": sort_vec_common,
                    "sortino_framework_fast": sort_fw,
                    "skipped": False,
                }
            )

    df = pd.DataFrame(rows)
    csv_all = args.out_dir / "full_grid_metrics.csv"
    df.to_csv(csv_all, index=False)

    best_rows = [per_month_best[m] for m in sorted(per_month_best)]
    df_best = pd.DataFrame(best_rows)
    csv_best = args.out_dir / "per_month_best.csv"
    df_best.to_csv(csv_best, index=False)

    finite_best = [r for r in best_rows if math.isfinite(float(r["Sortino"]))]
    overall = (
        max(finite_best, key=lambda r: float(r["Sortino"]))
        if finite_best
        else (best_rows[0] if best_rows else {})
    )
    summary = {
        "period": {"start": args.start_date, "end": args.end_date},
        "grid": {
            "entry_day_range": [args.entry_min, args.entry_max],
            "hold_days_range": [args.hold_min, args.hold_max],
            "evaluations": len(rows),
        },
        "costs": {
            "portfolio_value": PORT,
            "commission_per_share": CPS,
            "commission_min_per_order": CMIN,
            "commission_max_percent_of_trade": CMAX,
            "slippage_bps": SLIP,
        },
        "sortino_definition": "calculate_optimizer_metrics_fast raw returns (risk_free_rets=None); scenario extras risk_free_metrics_disabled_for_objective_alignment",
        "elapsed_seconds": elapsed,
        "per_month_best": best_rows,
        "overall_best_month": int(overall["calendar_month"]) if overall else None,
        "overall_best_row": overall if overall else None,
        "parity_checks": parity_out,
        "limitations": [
            "Vectorized path assumes no simple_high_low or ATR stop/take-profit (matches default-off strategy).",
            "Single-asset long 0/1 target; commission model matches per-share/min/max/slip scaling used in portfolio_logic for full investment.",
            "PnL uses weights lagged one session vs returns, matching portfolio_simulation_input.build_weights_arrays (shift(1)).",
            "Month-only strategies: one entry anchor per calendar year; overlapping cycles across years do not occur for a fixed month.",
            "Parity probes: max |r_vec - r_fw| on index intersection typ. ~1e-9; full-window Sortino can differ slightly from common-index Sortino if edges differ.",
        ],
    }
    (args.out_dir / "grid_summary.json").write_text(
        json.dumps(_sanitize_json_obj(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(_sanitize_json_obj(summary), indent=2, allow_nan=False))
    print(f"WROTE_FULL_GRID={csv_all.resolve()}", flush=True)
    print(f"WROTE_PER_MONTH_BEST={csv_best.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
