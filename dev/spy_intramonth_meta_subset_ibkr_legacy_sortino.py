"""Meta baseline: exhaustive non-empty month subsets with IBKR-like costs.

Uses per-calendar-month (entry_day, hold_days) from ``per_month_best.csv`` (from
``spy_intramonth_per_month_grid_ibkr_legacy_2005_2024.py``, default grid hold_days 5..20).
Subset runs set ``entry_day_by_month`` / ``hold_days_by_month`` plus ``trade_month_*`` on
``SeasonalSignalStrategy`` (production path; no monkeypatch). Union of monthly windows matches
``generate_signal_matrix`` when SL/TP/ATR exits are off.

Objective: full-period Sortino via ``calculate_optimizer_metrics_fast`` with
``risk_free_rets=None`` and ``risk_free_metrics_enabled: false``.

Costs: same as grid script (PORT 100k, CPS 0.005, CMIN 1.0, CMAX 0.005, SLIP 0.5 bps).
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

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


def load_ohlc_slice(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    start_date: str,
    end_date: str,
    cache_only: bool,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, np.ndarray]:
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

    sp: dict[str, Any] = {
        "direction": "long",
        "month_local_seasonal_windows": False,
        "hold_days": 10,
        "entry_day": 1,
        "simple_high_low_stop_loss": False,
        "simple_high_low_take_profit": False,
        "stop_loss_atr_multiple": 0.0,
        "take_profit_atr_multiple": 0.0,
    }
    for m in range(1, 13):
        sp[f"trade_month_{m}"] = m == 6

    probe_raw = {
        "name": "meta_subset_probe",
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

    norm = ScenarioNormalizer()
    canon = norm.normalize(scenario=probe_raw, global_config=gc)
    ohlc, _, _ = DataFetcher(gc, create_data_source(gc)).prepare_data_for_backtesting(
        [canon], StrategyManager().get_strategy
    )
    close_s = ohlc.xs("Close", level="Field", axis=1)["SPY"]
    ix = pd.DatetimeIndex(close_s.index)
    loc_ix = (
        pd.DatetimeIndex([pd.Timestamp(t).tz_convert(ix.tz).replace(tzinfo=None) for t in ix])
        if ix.tz is not None
        else ix
    )
    sel = (loc_ix >= start) & (loc_ix <= end)
    return ix[sel], loc_ix[sel], close_s.loc[sel].to_numpy(float)


def compute_month_masks(
    *,
    loc_ix: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    entry_by_month: dict[int, int],
    hold_by_month: dict[int, int],
) -> dict[int, np.ndarray]:
    n = len(loc_ix)
    out: dict[int, np.ndarray] = {}
    for m in range(1, 13):
        ed = entry_by_month[m]
        hd = hold_by_month[m]
        mask = np.zeros(n, dtype=np.uint8)
        for y in range(start.year - 1, end.year + 1):
            ent = nth_bday(y, m, ed)
            wend = ent + BDay(hd - 1)
            if wend < start or ent > end:
                continue
            pos = np.where((loc_ix >= ent) & (loc_ix <= wend))[0]
            mask[pos] = 1
        out[m] = mask
    return out


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


def load_per_month_best_csv(path: Path) -> tuple[dict[int, int], dict[int, int]]:
    entry_by_month: dict[int, int] = {}
    hold_by_month: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            mi = int(row["calendar_month"])
            entry_by_month[mi] = int(row["entry_day"])
            hold_by_month[mi] = int(row["hold_days"])
    if set(entry_by_month.keys()) != set(range(1, 13)):
        raise ValueError(f"Expected rows for months 1..12 in {path}")
    return entry_by_month, hold_by_month


def mask_combine(month_masks: dict[int, np.ndarray], bitmask: int) -> np.ndarray:
    n = len(next(iter(month_masks.values())))
    acc = np.zeros(n, dtype=np.uint8)
    for m in range(1, 13):
        if (bitmask >> (m - 1)) & 1:
            acc |= month_masks[m]
    return acc.astype(float)


def build_production_strategy_params(
    *,
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


def scenario_for_subset(
    *,
    name: str,
    bitmask: int,
    entry_by_month: dict[int, int],
    hold_by_month: dict[int, int],
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
        "strategy_params": build_production_strategy_params(
            bitmask=bitmask,
            entry_by_month=entry_by_month,
            hold_by_month=hold_by_month,
        ),
    }


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
    parser.add_argument(
        "--per-month-csv",
        type=Path,
        default=_REPO
        / "dev"
        / "spy_intramonth_per_month_grid_ibkr_legacy_hd5_20_2005_2024"
        / "per_month_best.csv",
        help="Rows calendar_month 1..12 with entry_day, hold_days (IBKR grid output).",
    )
    parser.add_argument("--cache-only", action="store_true", default=True)
    parser.add_argument("--no-cache-only", dest="cache_only", action="store_false")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO / "dev" / "spy_intramonth_meta_subset_ibkr_legacy_hd5_20_2005_2024",
    )
    parser.add_argument("--max-subsets", type=int, default=0, help="0 = all 4095 non-empty masks")
    parser.add_argument(
        "--parity-mask",
        type=int,
        default=None,
        help="Optional bitmask for BacktestRunner parity vs vectorized union.",
    )
    parser.add_argument(
        "--parity-only",
        action="store_true",
        help="Only run --parity-mask framework check; do not sweep subsets or overwrite CSV/TSV/JSON.",
    )
    args = parser.parse_args()
    if args.parity_only and args.parity_mask is None:
        raise SystemExit("--parity-only requires --parity-mask")

    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)

    if not args.per_month_csv.is_file():
        raise SystemExit(
            f"Missing {args.per_month_csv}; run dev/spy_intramonth_per_month_grid_ibkr_legacy_2005_2024.py first "
            f"(default hold_days grid 5..20, out-dir dev/spy_intramonth_per_month_grid_ibkr_legacy_hd5_20_2005_2024)."
        )

    entry_by_month, hold_by_month = load_per_month_best_csv(args.per_month_csv)

    ix, loc_ix, close = load_ohlc_slice(
        start=start,
        end=end,
        start_date=args.start_date,
        end_date=args.end_date,
        cache_only=args.cache_only,
    )
    rets = np.r_[0.0, close[1:] / close[:-1] - 1.0]
    bench_rets = pd.Series(rets, index=ix)

    month_masks = compute_month_masks(
        loc_ix=loc_ix,
        start=start,
        end=end,
        entry_by_month=entry_by_month,
        hold_by_month=hold_by_month,
    )

    max_m = args.max_subsets if args.max_subsets and args.max_subsets > 0 else (1 << 12) - 1
    masks_to_run = list(range(1, max_m + 1))

    rows: list[dict[str, Any]] = []
    best_mask = -1
    best_sort = float("-inf")
    t0 = time.perf_counter()

    if not args.parity_only:
        for mask in masks_to_run:
            tgt = mask_combine(month_masks, mask)
            mets = eval_target_net_returns(
                tgt, close=close, rets=rets, ix=ix, bench_rets=bench_rets
            )
            row = {
                "bitmask": mask,
                "n_months": int(bin(mask).count("1")),
                "months_csv": ";".join(_MONTH_EN[i] for i in range(12) if (mask >> i) & 1),
                **mets,
            }
            rows.append(row)
            s = mets["Sortino"]
            if math.isfinite(s) and s > best_sort:
                best_sort = s
                best_mask = mask

    elapsed = time.perf_counter() - t0

    parity_block: dict[str, Any] | None = None
    if args.parity_mask is not None:
        pm = int(args.parity_mask)
        if pm < 1 or pm >= (1 << 12):
            raise SystemExit("--parity-mask must satisfy 1 <= mask <= 4095")

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
        probe_raw = scenario_for_subset(
            name="parity_probe_meta",
            bitmask=pm,
            entry_by_month=entry_by_month,
            hold_by_month=hold_by_month,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        probe = norm.normalize(scenario=probe_raw, global_config=gc)
        daily_ohlc, monthly_data, daily_closes = fetcher.prepare_data_for_backtesting(
            [probe], strat_mgr.get_strategy
        )
        _rets_cached = data_cache.get_cached_returns(daily_closes, "full_period_returns")
        rets_full = (
            _rets_cached.to_frame()
            if isinstance(_rets_cached, pd.Series)
            else pd.DataFrame(_rets_cached)
        )

        pr_fw = runner.run_scenario(probe, monthly_data, daily_ohlc, rets_full, verbose=False)

        tgt_vec = mask_combine(month_masks, pm)
        lag_target = np.r_[0.0, tgt_vec[:-1]]
        gross = lag_target * rets
        delta = np.abs(tgt_vec - np.r_[0.0, tgt_vec[:-1]])
        tv = delta * PORT
        shares = np.where((tv > 0) & np.isfinite(close) & (close > 0), tv / close, 0.0)
        comm = np.zeros_like(tv)
        nz = shares > 0
        comm[nz] = np.minimum(np.maximum(shares[nz] * CPS, CMIN), tv[nz] * CMAX)
        costs = (comm + tv * (SLIP / 10000.0)) / PORT
        vec_net = pd.Series(gross - costs, index=ix)

        if pr_fw is None or pr_fw.empty:
            parity_block = {"skipped": True, "reason": "framework_empty"}
        else:
            common = pr_fw.index.intersection(vec_net.index)
            diff = float(
                (vec_net.loc[common].astype(float) - pr_fw.loc[common].astype(float)).abs().max()
            )
            sort_fw = float(
                calculate_optimizer_metrics_fast(
                    pr_fw.loc[common].astype(float),
                    bench_rets.loc[common],
                    "SPY",
                    risk_free_rets=None,
                    requested_metrics={"Sortino"},
                ).get("Sortino", np.nan)
            )
            sort_vec = float(
                calculate_optimizer_metrics_fast(
                    vec_net.loc[common].astype(float),
                    bench_rets.loc[common],
                    "SPY",
                    risk_free_rets=None,
                    requested_metrics={"Sortino"},
                ).get("Sortino", np.nan)
            )
            sort_vec_full = float(
                calculate_optimizer_metrics_fast(
                    vec_net.astype(float),
                    bench_rets,
                    "SPY",
                    risk_free_rets=None,
                    requested_metrics={"Sortino"},
                ).get("Sortino", np.nan)
            )
            parity_block = {
                "bitmask": pm,
                "max_abs_ret_diff_common_index": diff,
                "sortino_framework_common_index": sort_fw,
                "sortino_vectorized_common_index": sort_vec,
                "sortino_vectorized_full_index": sort_vec_full,
                "skipped": False,
            }

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.parity_only:
        out_p = args.out_dir / f"parity_mask_{int(args.parity_mask)}.json"
        doc = {
            "per_month_csv": str(args.per_month_csv.resolve()),
            "period": {"start": args.start_date, "end": args.end_date},
            "parity_check": parity_block,
        }
        out_p.write_text(
            json.dumps(_sanitize_json_obj(doc), indent=2, allow_nan=False), encoding="utf-8"
        )
        print(json.dumps(_sanitize_json_obj(doc), indent=2, allow_nan=False))
        print(f"WROTE_PARITY_JSON={out_p.resolve()}", flush=True)
        return 0

    csv_out = args.out_dir / "all_subsets_metrics.csv"
    pd.DataFrame(rows).to_csv(csv_out, index=False)

    finite_rows = [r for r in rows if math.isfinite(float(r["Sortino"]))]
    top_alternatives = sorted(finite_rows, key=lambda r: float(r["Sortino"]), reverse=True)[:15]

    best_row = next((r for r in rows if r["bitmask"] == best_mask), None)
    summary = {
        "best_bitmask": best_mask,
        "best_sortino": best_sort,
        "top_alternatives": top_alternatives,
        "elapsed_seconds": elapsed,
        "evaluated_subsets": len(rows),
        "per_month_csv": str(args.per_month_csv.resolve()),
        "period": {"start": args.start_date, "end": args.end_date},
        "costs": {
            "portfolio_value": PORT,
            "commission_per_share": CPS,
            "commission_min_per_order": CMIN,
            "commission_max_percent_of_trade": CMAX,
            "slippage_bps": SLIP,
        },
        "per_month_entry_hold": {
            str(m): {"entry_day": entry_by_month[m], "hold_days": hold_by_month[m]}
            for m in range(1, 13)
        },
        "best_row": best_row,
        "parity_check": parity_block,
        "limitations": [
            "Vectorized union matches SeasonalSignalStrategy.generate_signal_matrix (exits off); "
            "differs from generate_signals which uses first-anchor resolution when windows overlap.",
            "Optional --parity-mask: BacktestRunner uses production entry_day_by_month / hold_days_by_month; "
            "daily net returns vs vectorized union typ. ~1e-9 on index overlap; Sortino may differ slightly "
            "from full-window vectorized Sortino due to index edges / metric helpers.",
            "Per-month params from cost-aware grid Sortino unless CSV overridden.",
            "Commission model matches spy_intramonth_per_month_grid_ibkr_legacy_2005_2024.py.",
        ],
    }

    json_out = args.out_dir / "subset_search_summary.json"
    json_out.write_text(
        json.dumps(_sanitize_json_obj(summary), indent=2, allow_nan=False), encoding="utf-8"
    )

    print(json.dumps(_sanitize_json_obj(summary), indent=2, allow_nan=False))
    print(f"WROTE_GRID_CSV={csv_out.resolve()}", flush=True)
    print(f"WROTE_SUMMARY_JSON={json_out.resolve()}", flush=True)

    pdet = args.out_dir / "best_subset_month_params.tsv"
    with pdet.open("w", encoding="utf-8", newline="\n") as f:
        f.write("calendar_month\tmonth_name\tactive\tentry_day\thold_days\n")
        for m in range(1, 13):
            act = bool((best_mask >> (m - 1)) & 1)
            f.write(
                f"{m}\t{_MONTH_EN[m - 1]}\t{int(act)}\t{entry_by_month[m]}\t{hold_by_month[m]}\n"
            )
    print(f"WROTE_BEST_PARAMS_TSV={pdet.resolve()}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
