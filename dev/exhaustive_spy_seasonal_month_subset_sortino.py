"""Exhaust all 2^12-1 month masks; rank by fast Sortino (rf off in extras).

Per-month entry days come from data/reports CSV. Uses BacktestRunner.run_scenario plus
calculate_optimizer_metrics_fast to align with default optimizer objective math.
"""

from __future__ import annotations

from typing import Any

# ruff: noqa: E402

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

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
from portfolio_backtester.reporting.fast_objective_metrics import (
    calculate_optimizer_metrics_fast,
)
from portfolio_backtester.reporting.risk_free import build_optional_risk_free_series
from portfolio_backtester.scenario_normalizer import ScenarioNormalizer
from portfolio_backtester.strategies.builtins.signal.seasonal_signal_strategy import (
    SeasonalSignalStrategy,
)

_MONTH_NAMES_EN = (
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


def _load_summary_csv(path: Path) -> dict[int, int]:
    """Map calendar month 1..12 -> best_entry_day from seasonal summary."""
    by_month: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            mname = str(row.get("month", "")).strip()
            if not mname:
                continue
            try:
                mi = _MONTH_NAMES_EN.index(mname) + 1
            except ValueError:
                raise ValueError(f"Unknown month label {mname!r} in {path}") from None
            by_month[mi] = int(row["best_entry_day"])
    if set(by_month.keys()) != set(range(1, 13)):
        raise ValueError(f"Expected 12 rows in {path}, got keys {sorted(by_month)}")
    return by_month


def _base_scenario_dict(
    *,
    name: str,
    start_date: str,
    end_date: str,
    hold_days: int,
    bitmask: int,
    legacy_sortino: bool,
) -> dict:
    strat_params: dict = {
        "direction": "long",
        "month_local_seasonal_windows": False,
        "hold_days": int(hold_days),
        "entry_day": 1,
    }
    for m in range(1, 13):
        active = bool((bitmask >> (m - 1)) & 1)
        strat_params[f"trade_month_{m}"] = active

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
        "extras": (
            {"is_wfo": False, "risk_free_metrics_enabled": False}
            if legacy_sortino
            else {"is_wfo": False}
        ),
        "optimization_targets": [{"name": "Sortino", "direction": "maximize"}],
        "strategy_params": strat_params,
        "costs_config": {"transaction_costs_bps": 0.0},
    }


_CTX: dict[str, object] = {}
_ORIG_GET_ENTRY = SeasonalSignalStrategy.get_entry_date_for_month


def _install_entry_patch() -> None:
    def patched(self: SeasonalSignalStrategy, date: pd.Timestamp, entry_day: int) -> pd.Timestamp:
        cal = SeasonalSignalStrategy._calendar_naive(date)
        by_m = _CTX["entry_by_month"]
        assert isinstance(by_m, dict)
        m = int(cal.month)
        eff = int(by_m[m])
        return _ORIG_GET_ENTRY(self, date, eff)

    setattr(SeasonalSignalStrategy, "get_entry_date_for_month", patched)


def _uninstall_patch() -> None:
    setattr(SeasonalSignalStrategy, "get_entry_date_for_month", _ORIG_GET_ENTRY)


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
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=_REPO / "data" / "reports" / "seasonal_month_spylong_sortino_summary.csv",
    )
    parser.add_argument("--start-date", default="2005-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--hold-days", type=int, default=10)
    parser.add_argument(
        "--legacy-sortino",
        action="store_true",
        help="Set extras risk_free_metrics_enabled=false (legacy Sortino path)",
    )
    parser.add_argument("--cache-only", action="store_true", default=True)
    parser.add_argument("--no-cache-only", dest="cache_only", action="store_false")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO / "dev" / "spy_seasonal_month_subset_grid",
    )
    parser.add_argument("--max-subsets", type=int, default=0, help="0 = all 4095 non-empty")
    parser.add_argument(
        "--only-bitmask",
        type=int,
        default=None,
        help="If set, evaluate exactly this mask (1..4095) and ignore max-subsets sweep",
    )
    args = parser.parse_args()

    entry_by_month_all = _load_summary_csv(args.summary_csv)
    _CTX["entry_by_month"] = entry_by_month_all
    _install_entry_patch()

    config_loader.load_config()
    gc = config_loader.GLOBAL_CONFIG
    if args.cache_only:
        gc.setdefault("data_source_config", {})["cache_only"] = True

    strategy_manager = StrategyManager()
    data_source = create_data_source(gc)
    data_cache = create_cache_manager()
    fetcher = DataFetcher(gc, data_source)
    runner = BacktestRunner(gc, data_cache, strategy_manager, lambda: False)
    norm = ScenarioNormalizer()

    probe = norm.normalize(
        scenario=_base_scenario_dict(
            name="subset_probe",
            start_date=args.start_date,
            end_date=args.end_date,
            hold_days=args.hold_days,
            bitmask=1,
            legacy_sortino=args.legacy_sortino,
        ),
        global_config=gc,
    )

    daily_ohlc, monthly_data, daily_closes = fetcher.prepare_data_for_backtesting(
        [probe], strategy_manager.get_strategy
    )

    _rets_cached = data_cache.get_cached_returns(daily_closes, "full_period_returns")
    if isinstance(_rets_cached, pd.Series):
        rets_full = _rets_cached.to_frame()
    else:
        rets_full = pd.DataFrame(_rets_cached)

    max_m = args.max_subsets if args.max_subsets and args.max_subsets > 0 else (1 << 12) - 1

    if args.only_bitmask is not None:
        ob = int(args.only_bitmask)
        if ob < 1 or ob >= (1 << 12):
            raise SystemExit("--only-bitmask must satisfy 1 <= mask <= 4095")
        masks_to_run = [ob]
    else:
        masks_to_run = list(range(1, max_m + 1))

    bench_ticker = "SPY"

    rows: list[dict[str, Any]] = []
    best_mask = -1
    best_sort = float("-inf")
    t0 = time.perf_counter()

    try:
        for mask in masks_to_run:
            raw = _base_scenario_dict(
                name=f"subset_{mask:04x}",
                start_date=args.start_date,
                end_date=args.end_date,
                hold_days=args.hold_days,
                bitmask=mask,
                legacy_sortino=args.legacy_sortino,
            )
            canon = norm.normalize(scenario=raw, global_config=gc)
            pr = runner.run_scenario(canon, monthly_data, daily_ohlc, rets_full, verbose=False)
            if pr is None or pr.empty:
                sort_val = float("nan")
                metrics_dict = {
                    "Sharpe": float("nan"),
                    "Calmar": float("nan"),
                    "Total Return": float("nan"),
                    "Max Drawdown": float("nan"),
                    "Ann Return": float("nan"),
                }
            else:
                bench_px = daily_ohlc.xs("Close", level="Field", axis=1)[bench_ticker]
                bench_px = bench_px.loc[pr.index]
                bench_rets = bench_px.pct_change(fill_method=None).fillna(0.0)
                rf_opt = build_optional_risk_free_series(daily_ohlc, gc, pr.index, canon)
                metric_keys = {
                    "Sortino",
                    "Sharpe",
                    "Calmar",
                    "Total Return",
                    "Max Drawdown",
                    "Ann. Return",
                }
                fast = calculate_optimizer_metrics_fast(
                    pr,
                    bench_rets,
                    bench_ticker,
                    risk_free_rets=rf_opt,
                    requested_metrics=metric_keys,
                )
                sort_val_raw = fast.get("Sortino", float("nan"))
                sort_val = (
                    float(sort_val_raw)
                    if sort_val_raw is not None and sort_val_raw == sort_val_raw
                    else float("nan")
                )
                metrics_dict = {
                    "Sharpe": float(fast.get("Sharpe", float("nan"))),
                    "Calmar": float(fast.get("Calmar", float("nan"))),
                    "Total Return": float(fast.get("Total Return", float("nan"))),
                    "Max Drawdown": float(fast.get("Max Drawdown", float("nan"))),
                    "Ann Return": float(fast.get("Ann. Return", float("nan"))),
                }
            row = {
                "bitmask": mask,
                "n_months": int(bin(mask).count("1")),
                "months_csv": ";".join(_MONTH_NAMES_EN[i] for i in range(12) if (mask >> i) & 1),
                "Sortino": sort_val,
                "Sharpe": metrics_dict["Sharpe"],
                "Calmar": metrics_dict["Calmar"],
                "Total Return": metrics_dict["Total Return"],
                "Max Drawdown": metrics_dict["Max Drawdown"],
                "Ann Return": metrics_dict["Ann Return"],
            }
            rows.append(row)
            if sort_val == sort_val and sort_val > best_sort:
                best_sort = sort_val
                best_mask = mask

    finally:
        _uninstall_patch()

    elapsed = time.perf_counter() - t0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_out = args.out_dir / "all_subsets_metrics.csv"
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)

    best_row = next((r for r in rows if r["bitmask"] == best_mask), None)
    summary = {
        "best_bitmask": best_mask,
        "best_sortino": best_sort,
        "elapsed_seconds": elapsed,
        "evaluated_subsets": len(rows),
        "summary_csv_used": str(args.summary_csv.resolve()),
        "period": {"start": args.start_date, "end": args.end_date},
        "hold_days": args.hold_days,
        "legacy_sortino": args.legacy_sortino,
        "per_month_entry_day_from_csv": {
            str(k): entry_by_month_all[k] for k in sorted(entry_by_month_all)
        },
        "best_row": best_row,
    }
    safe_summary = _sanitize_json_obj(summary)
    assert isinstance(safe_summary, dict)
    (args.out_dir / "subset_search_summary.json").write_text(
        json.dumps(safe_summary, indent=2, allow_nan=False), encoding="utf-8"
    )

    print(json.dumps(safe_summary, indent=2, allow_nan=False))
    print(f"WROTE_GRID_CSV={csv_out.resolve()}", flush=True)
    print(
        f"WROTE_SUMMARY_JSON={(args.out_dir / 'subset_search_summary.json').resolve()}", flush=True
    )

    month_bits = [(i + 1, (best_mask >> i) & 1) for i in range(12)]
    detailed = []
    for m_i, active in month_bits:
        detailed.append(
            {
                "month": m_i,
                "active": bool(active),
                "entry_day": entry_by_month_all[m_i],
            }
        )

    pdet = args.out_dir / "best_subset_entry_days.tsv"
    with pdet.open("w", encoding="utf-8", newline="\n") as f:
        f.write("calendar_month\tactive\tentry_day\n")
        for d in detailed:
            f.write(f"{d['month']}\t{int(d['active'])}\t{d['entry_day']}\n")
    print(f"WROTE_BEST_ENTRIES_TSV={pdet.resolve()}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
