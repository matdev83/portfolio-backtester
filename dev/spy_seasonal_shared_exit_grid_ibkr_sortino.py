"""Exhaustive grid: shared exit params for fixed SPY intramonth seasonal meta-calendar.

Fixed months: Mar,Apr,May,Jun,Jul,Oct,Nov,Dec with per-calendar-month Nth business-day
entries (see ENTRY_BY_MONTH). Hold 10 business days, cross-month windows (legacy).

Uses SeasonalSignalStrategy exit semantics via the standard backtester (daily path when
any exit is active). Per-month entry days are injected by monkeypatching
``get_entry_date_for_month`` (not a production API).

Cost model: omit ``costs_config.transaction_costs_bps`` so portfolio simulation uses
``commission_per_share``, ``commission_min_per_order``, ``commission_max_percent_of_trade``,
``slippage_bps``, and ``portfolio_value`` from ``GLOBAL_CONFIG`` (set in main).

Objective: maximize full-period Sortino via ``calculate_optimizer_metrics_fast`` with
optional risk-free series (inherits ``risk_free_metrics_enabled`` / ``^IRX`` from loaded
parameters unless overridden).

Grid size: 2 x 2 x 51 x 41 = 8_364 combinations (stop 0..5 step 0.1, tp 0..10 step 0.25).
At ~20+ seconds per evaluation (daily generate_signals path), the full grid is usually
overnight work; use ``--max-combos`` for smoke or subsample.
"""

from __future__ import annotations

from typing import Any, Iterable

# ruff: noqa: E402

import argparse
import itertools
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

ENTRY_BY_MONTH: dict[int, int] = {
    1: 1,
    2: 1,
    3: 16,
    4: 11,
    5: 14,
    6: 20,
    7: 7,
    8: 1,
    9: 1,
    10: 19,
    11: 15,
    12: 14,
}

TRADE_MONTHS: dict[int, bool] = {
    1: False,
    2: False,
    3: True,
    4: True,
    5: True,
    6: True,
    7: True,
    8: False,
    9: False,
    10: True,
    11: True,
    12: True,
}

_CTX: dict[str, object] = {}
_ORIG_GET_ENTRY = SeasonalSignalStrategy.get_entry_date_for_month


def _install_entry_patch(entry_by_month: dict[int, int]) -> None:
    _CTX["entry_by_month"] = entry_by_month

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


def _frange_step(a: float, b: float, step: float) -> list[float]:
    n_steps = int(round((b - a) / step))
    return [round(a + i * step, 6) for i in range(n_steps + 1)]


def _scenario_dict(
    *,
    name: str,
    start_date: str,
    end_date: str,
    hold_days: int,
    simple_sl: bool,
    simple_tp: bool,
    sl_atr: float,
    tp_atr: float,
    risk_free_off: bool,
) -> dict[str, Any]:
    strat_params: dict[str, Any] = {
        "direction": "long",
        "month_local_seasonal_windows": False,
        "hold_days": int(hold_days),
        "entry_day": 1,
        "simple_high_low_stop_loss": bool(simple_sl),
        "simple_high_low_take_profit": bool(simple_tp),
        "stop_loss_atr_multiple": float(sl_atr),
        "take_profit_atr_multiple": float(tp_atr),
    }
    for m in range(1, 13):
        strat_params[f"trade_month_{m}"] = TRADE_MONTHS[m]

    extras: dict[str, Any] = {"is_wfo": False}
    if risk_free_off:
        extras["risk_free_metrics_enabled"] = False

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
        "extras": extras,
        "optimization_targets": [{"name": "Sortino", "direction": "maximize"}],
        "strategy_params": strat_params,
    }


def _sanitize_json_obj(obj: object) -> object:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {str(k): _sanitize_json_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json_obj(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    return obj


def _iter_grid(
    max_combos: int,
) -> Iterable[tuple[bool, bool, float, float]]:
    stops = _frange_step(0.0, 5.0, 0.1)
    tps = _frange_step(0.0, 10.0, 0.25)
    n = 0
    for simple_sl, simple_tp, sl_a, tp_a in itertools.product(
        [False, True], [False, True], stops, tps
    ):
        yield simple_sl, simple_tp, sl_a, tp_a
        n += 1
        if max_combos > 0 and n >= max_combos:
            return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2005-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--hold-days", type=int, default=10)
    parser.add_argument(
        "--legacy-sortino",
        action="store_true",
        help="extras risk_free_metrics_enabled=false (Sortino on raw returns)",
    )
    parser.add_argument("--cache-only", action="store_true", default=True)
    parser.add_argument("--no-cache-only", dest="cache_only", action="store_false")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO / "dev" / "spy_seasonal_shared_exit_grid_ibkr_sortino",
    )
    parser.add_argument(
        "--max-combos",
        type=int,
        default=0,
        help="0 = full 8376 grid; else stop after N evaluations (smoke)",
    )
    args = parser.parse_args()

    grid_stops = _frange_step(0.0, 5.0, 0.1)
    grid_tps = _frange_step(0.0, 10.0, 0.25)
    expected_total = 4 * len(grid_stops) * len(grid_tps)

    _install_entry_patch(dict(ENTRY_BY_MONTH))

    config_loader.load_config()
    gc = config_loader.GLOBAL_CONFIG
    gc["portfolio_value"] = 100_000.0
    gc["commission_per_share"] = 0.005
    gc["commission_min_per_order"] = 1.0
    gc["commission_max_percent_of_trade"] = 0.005
    gc["slippage_bps"] = 0.5
    if args.cache_only:
        gc.setdefault("data_source_config", {})["cache_only"] = True

    strategy_manager = StrategyManager()
    data_source = create_data_source(gc)
    data_cache = create_cache_manager()
    fetcher = DataFetcher(gc, data_source)
    runner = BacktestRunner(gc, data_cache, strategy_manager, lambda: False)
    norm = ScenarioNormalizer()

    probe = norm.normalize(
        scenario=_scenario_dict(
            name="exit_grid_probe",
            start_date=args.start_date,
            end_date=args.end_date,
            hold_days=args.hold_days,
            simple_sl=False,
            simple_tp=False,
            sl_atr=0.0,
            tp_atr=0.0,
            risk_free_off=args.legacy_sortino,
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

    bench_ticker = "SPY"
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    max_combos = int(args.max_combos)

    try:
        for idx, (simple_sl, simple_tp, sl_a, tp_a) in enumerate(_iter_grid(max_combos)):
            raw = _scenario_dict(
                name=f"exit_{idx:05d}",
                start_date=args.start_date,
                end_date=args.end_date,
                hold_days=args.hold_days,
                simple_sl=simple_sl,
                simple_tp=simple_tp,
                sl_atr=sl_a,
                tp_atr=tp_a,
                risk_free_off=args.legacy_sortino,
            )
            canon = norm.normalize(scenario=raw, global_config=gc)
            pr = runner.run_scenario(canon, monthly_data, daily_ohlc, rets_full, verbose=False)
            if pr is None or pr.empty:
                sort_val = float("nan")
                metrics_dict = {
                    "Sharpe": float("nan"),
                    "Total Return": float("nan"),
                    "Max Drawdown": float("nan"),
                }
            else:
                bench_px = daily_ohlc.xs("Close", level="Field", axis=1)[bench_ticker]
                bench_px = bench_px.loc[pr.index]
                bench_rets = bench_px.pct_change(fill_method=None).fillna(0.0)
                rf_opt = build_optional_risk_free_series(daily_ohlc, gc, pr.index, canon)
                fast = calculate_optimizer_metrics_fast(
                    pr,
                    bench_rets,
                    bench_ticker,
                    risk_free_rets=rf_opt,
                    requested_metrics={"Sortino", "Sharpe", "Total Return", "Max Drawdown"},
                )
                sort_raw = fast.get("Sortino", float("nan"))
                sort_val = (
                    float(sort_raw)
                    if sort_raw is not None and sort_raw == sort_raw
                    else float("nan")
                )
                metrics_dict = {
                    "Sharpe": float(fast.get("Sharpe", float("nan"))),
                    "Total Return": float(fast.get("Total Return", float("nan"))),
                    "Max Drawdown": float(fast.get("Max Drawdown", float("nan"))),
                }

            rows.append(
                {
                    "idx": idx,
                    "simple_high_low_stop_loss": simple_sl,
                    "simple_high_low_take_profit": simple_tp,
                    "stop_loss_atr_multiple": sl_a,
                    "take_profit_atr_multiple": tp_a,
                    "Sortino": sort_val,
                    **metrics_dict,
                }
            )
    finally:
        _uninstall_patch()

    elapsed = time.perf_counter() - t0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_out = args.out_dir / "shared_exit_grid_metrics.csv"
    if rows:
        with csv_out.open("w", newline="", encoding="utf-8") as f:
            w = pd.DataFrame(rows)
            w.to_csv(f, index=False)

    df = pd.DataFrame(rows)
    if df.empty:
        valid_sort = df
        top_n = df
        best_row = None
    else:
        valid_sort = df[df["Sortino"].notna()].sort_values("Sortino", ascending=False)
        top_n = valid_sort.head(25)
        best_row = valid_sort.iloc[0].to_dict() if len(valid_sort) else None
    summary = {
        "best_row": best_row,
        "top_25": top_n.to_dict(orient="records"),
        "elapsed_seconds": elapsed,
        "n_evaluations": len(rows),
        "grid": {
            "simple_high_low_stop_loss": [False, True],
            "simple_high_low_take_profit": [False, True],
            "stop_loss_atr_multiple_levels": len(grid_stops),
            "take_profit_atr_multiple_levels": len(grid_tps),
            "expected_total_evaluations": expected_total,
        },
        "costs_applied_global_config": {
            "portfolio_value": gc.get("portfolio_value"),
            "commission_per_share": gc.get("commission_per_share"),
            "commission_min_per_order": gc.get("commission_min_per_order"),
            "commission_max_percent_of_trade": gc.get("commission_max_percent_of_trade"),
            "slippage_bps": gc.get("slippage_bps"),
        },
        "entry_by_month": ENTRY_BY_MONTH,
        "trade_months": TRADE_MONTHS,
        "hold_days": args.hold_days,
        "period": {"start": args.start_date, "end": args.end_date},
        "legacy_sortino": args.legacy_sortino,
        "parity_notes": [
            "Per-month entry days via monkeypatch on SeasonalSignalStrategy.get_entry_date_for_month;",
            "production class uses a single entry_day for all months.",
            "No costs_config.transaction_costs_bps: simulation uses detailed commission+slippage.",
        ],
    }
    safe_summary = _sanitize_json_obj(summary)
    assert isinstance(safe_summary, dict)
    json_path = args.out_dir / "shared_exit_grid_summary.json"
    json_path.write_text(json.dumps(safe_summary, indent=2, allow_nan=False), encoding="utf-8")

    print(json.dumps(safe_summary, indent=2, allow_nan=False))
    print(f"WROTE_GRID_CSV={csv_out.resolve()}", flush=True)
    print(f"WROTE_SUMMARY_JSON={json_path.resolve()}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
