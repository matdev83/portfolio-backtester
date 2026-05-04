"""Robustness/stability checks for final SPY intramonth seasonal (bitmask 4086, final TSV).

Uses BacktestRunner with generate_signal_matrix disabled for production detailed-path parity
on baseline, slippage sweeps, and (optional) extended calendar fetch. Fast first-anchor scan
simulation for parameter neighborhoods and joint Monte Carlo perturbations (cost model matches
``spy_intramonth_detailed_month_grid_and_subset``).

Artifacts: ``dev/spy_intramonth_final_robustness_run/`` (default ``--out-dir``).

Run:
  ./.venv/Scripts/python.exe dev/spy_intramonth_final_robustness_suite.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ruff: noqa: E402
from portfolio_backtester.reporting.fast_objective_metrics import calculate_optimizer_metrics_fast

_FINAL_DIR = _REPO / "dev" / "spy_intramonth_detailed_ibkr_rf_off_2005_2024_final"
_DEFAULT_TSV = _FINAL_DIR / "best_subset_month_params.tsv"
_DEFAULT_JSON = _FINAL_DIR / "pipeline_summary.json"


def _load_grid() -> Any:
    path = _REPO / "dev" / "spy_intramonth_detailed_month_grid_and_subset.py"
    spec = importlib.util.spec_from_file_location("_intramonth_grid_mod", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load grid module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_best_from_tsv(tsv: Path) -> tuple[int, dict[int, int], dict[int, int]]:
    entry: dict[int, int] = {}
    hold: dict[int, int] = {}
    bitmask = 0
    with tsv.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            m = int(row["calendar_month"])
            entry[m] = int(row["entry_day"])
            hold[m] = int(row["hold_days"])
            if int(row["active"]):
                bitmask |= 1 << (m - 1)
    if set(entry.keys()) != set(range(1, 13)):
        raise ValueError("TSV must define months 1..12")
    return bitmask, entry, hold


def _sanitize_json(obj: object) -> object:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {str(k): _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    return obj


def _slice_by_naive_date(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    idx = pd.DatetimeIndex(series.index)
    if idx.tz is not None:
        cmp = pd.DatetimeIndex(
            [pd.Timestamp(ts).tz_convert(idx.tz).replace(tzinfo=None) for ts in idx]
        )
    else:
        cmp = idx
    m = (cmp >= start) & (cmp <= end)
    return series.iloc[np.flatnonzero(np.asarray(m, dtype=bool))]


def _metrics_from_returns(
    pr: pd.Series,
    bench: pd.Series,
    *,
    extras: set[str] | None = None,
) -> dict[str, float]:
    want = {"Sortino", "Sharpe", "Calmar", "Total Return", "Max Drawdown", "Ann. Return"}
    if extras:
        want |= extras
    raw = calculate_optimizer_metrics_fast(
        pr.astype(float),
        bench.reindex(pr.index).astype(float).fillna(0.0),
        "SPY",
        risk_free_rets=None,
        requested_metrics=want,
    )
    return {str(k): float(v) for k, v in raw.items()}


def _bench_for_index(grid: Any, shared: tuple[Any, ...], pr_index: pd.Index) -> pd.Series:
    daily_ohlc = shared[0]
    close = daily_ohlc.xs("Close", level="Field", axis=1)["SPY"].reindex(pr_index).astype(float)
    return cast(pd.Series, close.pct_change(fill_method=None).fillna(0.0))


def _eval_fast(
    grid: Any,
    strategy_params: dict[str, Any],
    *,
    close_a: np.ndarray,
    rets: np.ndarray,
    ix_f: pd.DatetimeIndex,
    loc_f: pd.DatetimeIndex,
    bench_rets: pd.Series,
    slippage_bps: float,
) -> dict[str, Any]:
    prev = float(grid.SLIP)
    grid.SLIP = float(slippage_bps)
    try:
        tgt = grid.detailed_allocation_mask(strategy_params, loc_f)
        return cast(
            dict[str, Any],
            grid.eval_target_net_returns(
                tgt,
                close=close_a,
                rets=rets,
                ix=ix_f,
                bench_rets=bench_rets,
            ),
        )
    finally:
        grid.SLIP = prev


def _block_bootstrap_sortinos(
    daily: np.ndarray,
    index: pd.Index,
    *,
    block_len: int,
    n_boot: int,
    bench_aligned: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Circular block bootstrap of daily portfolio returns; Sortino each replicate."""
    n = int(daily.shape[0])
    bl = max(1, int(block_len))
    if n < bl * 3:
        return np.array([])
    n_blocks = int(math.ceil(n / bl))
    ring = np.concatenate([daily, daily[: bl - 1]])
    ring_b = np.concatenate([bench_aligned, bench_aligned[: bl - 1]])
    out = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pos = rng.integers(0, n, size=n_blocks, endpoint=False)
        syn = np.empty(n, dtype=np.float64)
        syn_b = np.empty(n, dtype=np.float64)
        w = 0
        for j in range(n_blocks):
            s = int(pos[j])
            chunk = ring[s : s + bl]
            chunk_b = ring_b[s : s + bl]
            take = min(bl, n - w)
            syn[w : w + take] = chunk[:take]
            syn_b[w : w + take] = chunk_b[:take]
            w += take
            if w >= n:
                break
        m = calculate_optimizer_metrics_fast(
            pd.Series(syn, index=index),
            pd.Series(syn_b, index=index),
            "SPY",
            risk_free_rets=None,
            requested_metrics={"Sortino"},
        )
        out[b] = float(m.get("Sortino", np.nan))
    return out


def main() -> int:
    grid = _load_grid()
    tsv = _DEFAULT_TSV
    js = _DEFAULT_JSON
    bitmask, entry0, hold0 = _load_best_from_tsv(tsv)
    pipeline_bitmask = None
    if js.is_file():
        docj = json.loads(js.read_text(encoding="utf-8"))
        pipeline_bitmask = int(docj.get("best_bitmask", bitmask))
    if pipeline_bitmask is not None and int(pipeline_bitmask) != int(bitmask):
        raise ValueError(
            f"bitmask mismatch TSV={bitmask} pipeline={pipeline_bitmask} "
            f"(use consistent final artifacts)"
        )

    out_dir = _REPO / "dev" / "spy_intramonth_final_robustness_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_date = "2005-01-01"
    end_date = "2024-12-31"
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    t0 = time.perf_counter()
    grid.config_loader.load_config()
    gc0 = grid.config_loader.GLOBAL_CONFIG
    gc0.setdefault("data_source_config", {})["cache_only"] = True
    gc0.update(
        {
            "portfolio_value": grid.PORT,
            "commission_per_share": grid.CPS,
            "commission_min_per_order": grid.CMIN,
            "commission_max_percent_of_trade": grid.CMAX,
            "slippage_bps": 0.5,
        },
    )

    fetch_start = "1995-01-01"
    fetch_end = "2025-12-31"
    probe = grid.ScenarioNormalizer().normalize(
        scenario=grid._scenario_template(
            name="robustness_probe",
            strategy_params=grid._params_subset(bitmask, entry0, hold0),
            start_date=fetch_start,
            end_date=fetch_end,
        ),
        global_config=gc0,
    )
    fetcher = grid.DataFetcher(gc0, grid.create_data_source(gc0))
    sm = grid.StrategyManager()
    daily_ohlc, _monthly_data, _daily_closes = fetcher.prepare_data_for_backtesting(
        [probe], sm.get_strategy
    )
    close_all = daily_ohlc.xs("Close", level="Field", axis=1)["SPY"]
    ix_all = pd.DatetimeIndex(close_all.index)
    loc_all = (
        pd.DatetimeIndex(
            [pd.Timestamp(t).tz_convert(ix_all.tz).replace(tzinfo=None) for t in ix_all]
        )
        if ix_all.tz is not None
        else ix_all
    )
    data_min = pd.Timestamp(loc_all.min())
    data_max = pd.Timestamp(loc_all.max())

    _gc, ix_f, loc_f, close_a, bench_rets, rets, shared = grid.load_global_and_ohlc(
        start_date=start_date,
        end_date=end_date,
        cache_only=True,
        start=start,
        end=end,
    )

    base_params = grid._params_subset(bitmask, entry0, hold0)

    fast_base = _eval_fast(
        grid,
        base_params,
        close_a=close_a,
        rets=rets,
        ix_f=ix_f,
        loc_f=loc_f,
        bench_rets=bench_rets,
        slippage_bps=0.5,
    )
    pr_run, met_run = grid.run_detailed_runner(
        global_config=gc0,
        shared=shared,
        strategy_params=base_params,
        name="robustness_baseline_detailed",
        start_date=start_date,
        end_date=end_date,
        start=start,
        end=end,
    )
    bench_run = _bench_for_index(grid, shared, pr_run.index)
    tgt_base = grid.detailed_allocation_mask(base_params, loc_f)
    lag_tgt = np.r_[0.0, tgt_base[:-1]]
    gross = lag_tgt * rets
    delta = np.abs(tgt_base - np.r_[0.0, tgt_base[:-1]])
    tv = delta * grid.PORT
    shares = np.where((tv > 0) & np.isfinite(close_a) & (close_a > 0), tv / close_a, 0.0)
    comm = np.zeros_like(tv)
    nz = shares > 0
    comm[nz] = np.minimum(np.maximum(shares[nz] * grid.CPS, grid.CMIN), tv[nz] * grid.CMAX)
    costs = (comm + tv * (0.5 / 10000.0)) / grid.PORT
    sim_net = pd.Series(gross - costs, index=ix_f)
    parity_mad = grid.parity_max_abs_diff(sim_net=sim_net, runner_net=pr_run, start=start, end=end)

    active_months = [m for m in range(1, 13) if (bitmask >> (m - 1)) & 1]

    neigh_rows: list[dict[str, Any]] = []
    for m in active_months:
        for de in range(-3, 4):
            for dh in range(-3, 4):
                if de == 0 and dh == 0:
                    continue
                ent = dict(entry0)
                hd = dict(hold0)
                ent[m] = int(np.clip(ent[m] + de, 1, 20))
                hd[m] = int(np.clip(hd[m] + dh, 3, 25))
                sp = grid._params_subset(bitmask, ent, hd)
                mm = _eval_fast(
                    grid,
                    sp,
                    close_a=close_a,
                    rets=rets,
                    ix_f=ix_f,
                    loc_f=loc_f,
                    bench_rets=bench_rets,
                    slippage_bps=0.5,
                )
                neigh_rows.append(
                    {
                        "perturb": "single_month",
                        "month": m,
                        "de": de,
                        "dh": dh,
                        "entry_new": ent[m],
                        "hold_new": hd[m],
                        **mm,
                    }
                )

    rng = np.random.default_rng(42)
    mc_rows: list[dict[str, Any]] = []
    for t in range(600):
        ent = dict(entry0)
        hd = dict(hold0)
        for m in active_months:
            ent[m] = int(np.clip(ent[m] + int(rng.integers(-3, 4)), 1, 20))
            hd[m] = int(np.clip(hd[m] + int(rng.integers(-3, 4)), 3, 25))
        sp = grid._params_subset(bitmask, ent, hd)
        mm = _eval_fast(
            grid,
            sp,
            close_a=close_a,
            rets=rets,
            ix_f=ix_f,
            loc_f=loc_f,
            bench_rets=bench_rets,
            slippage_bps=0.5,
        )
        mc_rows.append({"trial": t, **{f"e{m}": ent[m] for m in active_months}, **mm})

    slip_runner_rows: list[dict[str, Any]] = []
    for slip in (0.5, 1.0, 2.5, 5.0):
        gc0["slippage_bps"] = float(slip)
        _pr_s, met_s = grid.run_detailed_runner(
            global_config=gc0,
            shared=shared,
            strategy_params=base_params,
            name=f"robustness_slip_{slip}",
            start_date=start_date,
            end_date=end_date,
            start=start,
            end=end,
        )
        slip_runner_rows.append({"slippage_bps": slip, **met_s, "note": "BacktestRunner detailed"})
    gc0["slippage_bps"] = 0.5

    pr_train = _slice_by_naive_date(pr_run, start, pd.Timestamp("2018-12-31"))
    pr_test = _slice_by_naive_date(pr_run, pd.Timestamp("2019-01-01"), end)
    bench_tr = _bench_for_index(grid, shared, pr_train.index)
    bench_te = _bench_for_index(grid, shared, pr_test.index)
    holdout = {
        "train_2005_2018": _metrics_from_returns(pr_train, bench_tr),
        "holdout_2019_2024": _metrics_from_returns(pr_test, bench_te),
    }

    blocked: list[dict[str, Any]] = []
    for y0 in range(2005, 2022, 3):
        y1 = min(y0 + 2, 2024)
        p0 = pd.Timestamp(f"{y0}-01-01")
        p1 = pd.Timestamp(f"{y1}-12-31")
        sl = _slice_by_naive_date(pr_run, p0, p1)
        if sl.shape[0] < 50:
            continue
        bsl = _bench_for_index(grid, shared, sl.index)
        blocked.append({"block": f"{y0}_{y1}", **_metrics_from_returns(sl, bsl)})

    pr_arr = pr_run.astype(float).to_numpy()
    b_arr = bench_run.reindex(pr_run.index).astype(float).fillna(0.0).to_numpy()
    bs21 = _block_bootstrap_sortinos(
        pr_arr,
        pr_run.index,
        block_len=21,
        n_boot=2500,
        bench_aligned=b_arr,
        rng=rng,
    )
    bs63 = _block_bootstrap_sortinos(
        pr_arr,
        pr_run.index,
        block_len=63,
        n_boot=2500,
        bench_aligned=b_arr,
        rng=rng,
    )

    extended: dict[str, Any] = {
        "data_min": str(data_min.date()),
        "data_max": str(data_max.date()),
    }
    if data_min <= pd.Timestamp("2000-01-01") and data_max >= pd.Timestamp("2020-12-31"):
        ext_start = max(pd.Timestamp("2000-01-01"), data_min)
        ext_end = min(pd.Timestamp("2024-12-31"), data_max)
        ext_start_s = ext_start.strftime("%Y-%m-%d")
        ext_end_s = ext_end.strftime("%Y-%m-%d")
        shared_e: tuple[Any, ...]
        _, ix_e, loc_e, close_e, bench_e, rets_e, shared_e = grid.load_global_and_ohlc(
            start_date=ext_start_s,
            end_date=ext_end_s,
            cache_only=True,
            start=ext_start,
            end=ext_end,
        )
        fast_ext = _eval_fast(
            grid,
            base_params,
            close_a=close_e,
            rets=rets_e,
            ix_f=ix_e,
            loc_f=loc_e,
            bench_rets=bench_e,
            slippage_bps=0.5,
        )
        _, met_ext_run = grid.run_detailed_runner(
            global_config=gc0,
            shared=shared_e,
            strategy_params=base_params,
            name="robustness_extended_detailed",
            start_date=ext_start_s,
            end_date=ext_end_s,
            start=ext_start,
            end=ext_end,
        )
        extended.update(
            {
                "window": [ext_start_s, ext_end_s],
                "fast_sim": fast_ext,
                "runner": met_ext_run,
            }
        )
    else:
        extended["note"] = "skipped: cache span narrower than 2000..2020 requirement"

    sortinos_neigh = np.array([float(r["Sortino"]) for r in neigh_rows], dtype=np.float64)
    sortinos_mc = np.array([float(r["Sortino"]) for r in mc_rows], dtype=np.float64)
    summary = {
        "bitmask": int(bitmask),
        "artifacts": {"tsv": str(tsv.resolve()), "pipeline": str(js.resolve())},
        "baseline_period": [start_date, end_date],
        "parity_max_abs_daily_return_diff": float(parity_mad),
        "baseline_fast_sortino": float(fast_base["Sortino"]),
        "baseline_runner_sortino": float(met_run.get("Sortino", float("nan"))),
        "train_holdout": holdout,
        "slippage_runner": slip_runner_rows,
        "param_neighborhood": {
            "n_variants": int(len(neigh_rows)),
            "sortino_q05_q95": [
                float(np.nanpercentile(sortinos_neigh, 5)),
                float(np.nanpercentile(sortinos_neigh, 95)),
            ],
            "sortino_min": float(np.nanmin(sortinos_neigh)),
        },
        "joint_mc_perturb_600": {
            "sortino_q05_q95": [
                float(np.nanpercentile(sortinos_mc, 5)),
                float(np.nanpercentile(sortinos_mc, 95)),
            ],
            "sortino_min": float(np.nanmin(sortinos_mc)),
        },
        "blocked_3y_metrics": blocked,
        "bootstrap_sortino_daily_blocks": {
            "block_21d_n2500": {
                "q05_median_q95": [
                    float(np.nanpercentile(bs21, 5)),
                    float(np.nanpercentile(bs21, 50)),
                    float(np.nanpercentile(bs21, 95)),
                ]
            },
            "block_63d_n2500": {
                "q05_median_q95": [
                    float(np.nanpercentile(bs63, 5)),
                    float(np.nanpercentile(bs63, 50)),
                    float(np.nanpercentile(bs63, 95)),
                ]
            },
        },
        "extended_calendar": extended,
        "elapsed_seconds": time.perf_counter() - t0,
        "limitations": [
            "eval_target_net_returns / first-anchor mask in dev month_grid no longer matches production BacktestRunner P&L on identical masks (checked: Sortino ~2.15 vs ~2.66; Total Return ~8.09 vs ~10.71). Parameter neighborhood and joint MC in this suite use that fast path—non-authoritative until P&L timing is realigned.",
            "Block bootstrap is on daily net detailed returns (stationarity not guaranteed).",
            "cache_only=True; extended window needs sufficient cached OHLC.",
            "Holdout is time-slice of one full-sample BacktestRunner run (fixed params, no re-fit).",
        ],
    }

    pd.DataFrame(neigh_rows).to_csv(out_dir / "param_neighborhood_single_month.csv", index=False)
    pd.DataFrame(mc_rows).to_csv(out_dir / "param_joint_mc_600.csv", index=False)
    pd.DataFrame(slip_runner_rows).to_csv(out_dir / "slippage_sensitivity_runner.csv", index=False)
    pd.DataFrame(blocked).to_csv(out_dir / "blocked_three_year_metrics.csv", index=False)
    (out_dir / "robustness_summary.json").write_text(
        json.dumps(_sanitize_json(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    with (out_dir / "train_holdout_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(_sanitize_json(holdout), f, indent=2, allow_nan=False)

    print(json.dumps(_sanitize_json(summary), indent=2, allow_nan=False))
    print(f"WROTE={out_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
