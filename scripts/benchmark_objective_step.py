"""Synthetic micro-benchmarks for target generation, simulation, and metrics timing."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from portfolio_backtester.backtester_logic.portfolio_simulation_input import (
    build_close_and_mask_from_dataframe,
    build_portfolio_simulation_input,
)
from portfolio_backtester.reporting.performance_metrics import calculate_metrics
from portfolio_backtester.simulation.kernel import simulate_portfolio

DEFAULT_SMALL_SCENARIOS: tuple[str, ...] = ("single_asset", "fixed_10", "momentum_50")


@dataclass(frozen=True)
class BenchmarkRecord:
    scenario: str
    target_generation_ms: float
    simulation_ms: float
    metrics_ms: float
    total_ms: float
    num_assets: int
    num_days: int
    num_rebalances: int


def benchmark_record_to_dict(rec: BenchmarkRecord) -> dict[str, Any]:
    return asdict(rec)


def _zero_cost_global(portfolio_value: float) -> dict[str, Any]:
    return {
        "portfolio_value": float(portfolio_value),
        "commission_per_share": 0.0,
        "commission_min_per_order": 0.0,
        "commission_max_percent_of_trade": 0.0,
        "slippage_bps": 0.0,
    }


def _scenario_config() -> dict[str, Any]:
    return {"allocation_mode": "reinvestment"}


def _synthetic_prices(
    n_days: int, n_assets: int, *, seed: int, start: str = "2018-01-01"
) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="B")
    r = rng.normal(0.0003, 0.008, size=(n_days, n_assets))
    px = 100.0 * np.exp(np.cumsum(r, axis=0))
    cols = [f"S{i:02d}" for i in range(n_assets)]
    return dates, pd.DataFrame(px, index=dates, columns=cols)


def _rebalance_mask_every(n_days: int, step: int) -> np.ndarray:
    rb = np.zeros(n_days, dtype=np.bool_)
    rb[0] = True
    for i in range(step, n_days, step):
        rb[i] = True
    return rb


def _run_metrics(daily_returns: pd.Series) -> None:
    bench = pd.Series(0.0, index=daily_returns.index, dtype=float)
    calculate_metrics(
        daily_returns,
        bench,
        "bench",
        risk_free_rets=None,
    )


def _bench_from_workload(
    scenario: str,
    dates: pd.DatetimeIndex,
    close_df: pd.DataFrame,
    weights_daily: pd.DataFrame,
    rebalance_mask: np.ndarray,
) -> BenchmarkRecord:
    cols = list(close_df.columns)
    tg_ms = 0.0

    t0 = perf_counter()
    close_arr, close_mask = build_close_and_mask_from_dataframe(close_df, dates, cols)
    sim_in = build_portfolio_simulation_input(
        weights_daily=weights_daily,
        price_index=dates,
        valid_cols=cols,
        close_arr=close_arr,
        close_price_mask_arr=close_mask,
        rebalance_mask_arr=rebalance_mask,
        trade_execution_timing="bar_close",
    )
    out = simulate_portfolio(
        sim_in,
        global_config=_zero_cost_global(100_000.0),
        scenario_config=_scenario_config(),
    )
    sim_ms = (perf_counter() - t0) * 1000.0

    t1 = perf_counter()
    _run_metrics(out.daily_returns)
    metrics_ms = (perf_counter() - t1) * 1000.0

    total_ms = tg_ms + sim_ms + metrics_ms
    return BenchmarkRecord(
        scenario=scenario,
        target_generation_ms=tg_ms,
        simulation_ms=sim_ms,
        metrics_ms=metrics_ms,
        total_ms=total_ms,
        num_assets=len(cols),
        num_days=len(dates),
        num_rebalances=int(np.sum(rebalance_mask)),
    )


def _single_asset_targets(
    dates: pd.DatetimeIndex, close_df: pd.DataFrame, step: int
) -> tuple[pd.DataFrame, np.ndarray]:
    weights = pd.DataFrame(
        np.ones((len(dates), 1), dtype=np.float64),
        index=dates,
        columns=close_df.columns,
    )
    rb = _rebalance_mask_every(len(dates), step)
    return weights, rb


def _fixed_n_targets(
    dates: pd.DatetimeIndex, close_df: pd.DataFrame, step: int
) -> tuple[pd.DataFrame, np.ndarray]:
    n = close_df.shape[1]
    w = np.full((len(dates), n), 1.0 / float(n), dtype=np.float64)
    weights = pd.DataFrame(w, index=dates, columns=close_df.columns)
    rb = _rebalance_mask_every(len(dates), step)
    return weights, rb


def _momentum_targets(
    close_df: pd.DataFrame,
    *,
    lookback: int,
    top_k: int,
    rebalance_every: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    arr = close_df.to_numpy(dtype=np.float64, copy=False)
    n_days, n_assets = arr.shape
    weights_arr = np.zeros((n_days, n_assets), dtype=np.float64)
    rb = np.zeros(n_days, dtype=np.bool_)
    for t in range(n_days):
        if t == 0 or (t % rebalance_every == 0):
            rb[t] = True
        if not rb[t]:
            weights_arr[t] = weights_arr[t - 1]
            continue
        if t < lookback:
            weights_arr[t, :] = 1.0 / float(n_assets)
            continue
        momentum = arr[t] / arr[t - lookback] - 1.0
        pick = np.argsort(-momentum)[:top_k]
        row = np.zeros(n_assets, dtype=np.float64)
        row[pick] = 1.0 / float(top_k)
        weights_arr[t] = row
    wdf = pd.DataFrame(weights_arr, index=close_df.index, columns=close_df.columns)
    return wdf, rb


def run_benchmark_single_asset() -> BenchmarkRecord:
    tg0 = perf_counter()
    dates, close_df = _synthetic_prices(64, 1, seed=11)
    weights, rb = _single_asset_targets(dates, close_df, step=16)
    tg_ms = (perf_counter() - tg0) * 1000.0

    base = _bench_from_workload("single_asset", dates, close_df, weights, rb)
    return BenchmarkRecord(
        scenario=base.scenario,
        target_generation_ms=tg_ms,
        simulation_ms=base.simulation_ms,
        metrics_ms=base.metrics_ms,
        total_ms=tg_ms + base.simulation_ms + base.metrics_ms,
        num_assets=base.num_assets,
        num_days=base.num_days,
        num_rebalances=base.num_rebalances,
    )


def run_benchmark_fixed_10() -> BenchmarkRecord:
    tg0 = perf_counter()
    dates, close_df = _synthetic_prices(64, 10, seed=13)
    weights, rb = _fixed_n_targets(dates, close_df, step=8)
    tg_ms = (perf_counter() - tg0) * 1000.0

    base = _bench_from_workload("fixed_10", dates, close_df, weights, rb)
    return BenchmarkRecord(
        scenario=base.scenario,
        target_generation_ms=tg_ms,
        simulation_ms=base.simulation_ms,
        metrics_ms=base.metrics_ms,
        total_ms=tg_ms + base.simulation_ms + base.metrics_ms,
        num_assets=base.num_assets,
        num_days=base.num_days,
        num_rebalances=base.num_rebalances,
    )


def run_benchmark_momentum_50() -> BenchmarkRecord:
    tg0 = perf_counter()
    dates, close_df = _synthetic_prices(96, 50, seed=17)
    weights, rb = _momentum_targets(close_df, lookback=15, top_k=10, rebalance_every=8)
    tg_ms = (perf_counter() - tg0) * 1000.0

    base = _bench_from_workload("momentum_50", dates, close_df, weights, rb)
    return BenchmarkRecord(
        scenario=base.scenario,
        target_generation_ms=tg_ms,
        simulation_ms=base.simulation_ms,
        metrics_ms=base.metrics_ms,
        total_ms=tg_ms + base.simulation_ms + base.metrics_ms,
        num_assets=base.num_assets,
        num_days=base.num_days,
        num_rebalances=base.num_rebalances,
    )


def run_benchmarks(scenario_names: Sequence[str] | None = None) -> list[BenchmarkRecord]:
    names = tuple(DEFAULT_SMALL_SCENARIOS) if scenario_names is None else tuple(scenario_names)
    out: list[BenchmarkRecord] = []
    for name in names:
        if name == "single_asset":
            out.append(run_benchmark_single_asset())
        elif name == "fixed_10":
            out.append(run_benchmark_fixed_10())
        elif name == "momentum_50":
            out.append(run_benchmark_momentum_50())
        else:
            msg = f"unknown scenario {name!r}; supported: {', '.join(DEFAULT_SMALL_SCENARIOS)}"
            raise ValueError(msg)
    return out


def violations_for_enforce(
    records: Sequence[BenchmarkRecord],
    baseline_totals_ms: Mapping[str, float],
    threshold: float,
) -> list[tuple[str, float, float]]:
    thr = float(threshold)
    by_name = {r.scenario: r for r in records}
    viol: list[tuple[str, float, float]] = []
    for scen, base_raw in baseline_totals_ms.items():
        rec = by_name.get(scen)
        if rec is None:
            continue
        limit_ms = float(base_raw) * thr
        if rec.total_ms > limit_ms:
            viol.append((scen, limit_ms, rec.total_ms))
    return viol


def load_baseline_totals(path: Path) -> dict[str, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    recs = raw.get("records", [])
    out: dict[str, float] = {}
    for row in recs:
        if isinstance(row, dict) and "scenario" in row and "total_ms" in row:
            out[str(row["scenario"])] = float(row["total_ms"])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        default=None,
        help="Scenario name (repeatable). Default: built-in small set.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write JSON to this path.")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="JSON file with prior run records (total_ms per scenario).",
    )
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="Exit 1 if any scenario exceeds baseline total_ms * threshold.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Multiplier on baseline total_ms for --enforce (default: 2).",
    )
    args = parser.parse_args(argv)

    scenario_list = args.scenarios if args.scenarios else list(DEFAULT_SMALL_SCENARIOS)
    records = run_benchmarks(scenario_list)
    payload: dict[str, Any] = {"records": [benchmark_record_to_dict(r) for r in records]}

    text = json.dumps(payload, indent=2 if args.output else None)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")

    if args.enforce:
        if args.baseline is None:
            sys.stderr.write("benchmark_objective_step: --enforce requires --baseline\n")
            return 2
        baseline = load_baseline_totals(args.baseline)
        viol = violations_for_enforce(records, baseline, args.threshold)
        if viol:
            for scen, lim, act in viol:
                sys.stderr.write(
                    f"benchmark_objective_step: {scen} total_ms={act:.6f} "
                    f"exceeds limit {lim:.6f} (baseline * threshold)\n"
                )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
