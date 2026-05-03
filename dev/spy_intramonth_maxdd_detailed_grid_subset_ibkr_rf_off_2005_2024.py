"""SPY intramonth seasonal — ``max_dd_from_ath_pct`` grid (RoRo off), IBKR + 0.5 bps slip, RF-off.

Task A (heterogeneous): each month optimizes Sortino over entry 1..20, hold 5..20,
``max_dd_from_ath_pct`` in {0(off), 5..15}.

Task B (heterogeneous): 4095 month subsets × first-anchor semantics × DD gate tied to anchor
winner month (**not** reproducible via one YAML ``max_dd_from_ath_pct`` field).

Sweep (deployable): same DD threshold for every day; rerun Task A and Task B per threshold;
winner matches production ``SeasonalSignalStrategy``. Optional ``BacktestRunner`` parity.
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

_DD_GRID = (0.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0)


def _roro_false() -> dict[str, Any]:
    return {"use_carlos_roro": False}


def _scenario_template(
    *,
    name: str,
    strategy_params: dict[str, Any],
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    sp = {**strategy_params, **_roro_false()}
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
        **_roro_false(),
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
        "max_dd_from_ath_pct": 0.0,
    }
    for m in range(1, 13):
        sp[f"trade_month_{m}"] = m == int(calendar_month)
    return sp


def _params_subset(
    bitmask: int,
    entry_by_month: dict[int, int],
    hold_by_month: dict[int, int],
    *,
    max_dd_from_ath_pct: float,
) -> dict[str, Any]:
    max_hold = max(int(hold_by_month[m]) for m in range(1, 13))
    sp: dict[str, Any] = {
        **_roro_false(),
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
        "max_dd_from_ath_pct": float(max_dd_from_ath_pct),
    }
    for m in range(1, 13):
        sp[f"trade_month_{m}"] = bool((bitmask >> (m - 1)) & 1)
    return sp


def _nth_bday(year: int, month: int, n: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    last = (first + pd.offsets.MonthEnd(1)).normalize()
    bdays = pd.bdate_range(first, last)
    idx = n - 1 if n > 0 else n
    idx = max(min(idx, len(bdays) - 1), -len(bdays))
    return pd.Timestamp(bdays[idx])


def subset_entry_hold(
    entry_best: dict[int, int],
    hold_best: dict[int, int],
) -> tuple[dict[int, int], dict[int, int]]:
    return (
        {m: int(entry_best[m]) for m in range(1, 13)},
        {m: int(hold_best[m]) for m in range(1, 13)},
    )


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
            ed_k = int(entry_by_month[m])
            hd_k = int(hold_by_month[m])
            key = (y, m, ed_k, hd_k)
            if key not in cache:
                ent = _nth_bday(y, m, ed_k)
                cache[key] = (ent, ent + pd.tseries.offsets.BDay(hd_k - 1))
            ent, wend = cache[key]
            month_at_k[i, k] = np.int8(m)
            in_win[i, k] = bool(ent <= d < wend)
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


def allocate_from_scan_with_winner(
    month_at_k: np.ndarray,
    in_win: np.ndarray,
    bitmask: int,
) -> tuple[np.ndarray, np.ndarray]:
    n, k_max = month_at_k.shape
    active = np.zeros(n, dtype=np.bool_)
    winner = np.zeros(n, dtype=np.int8)
    bits = int(bitmask)
    for k in range(k_max):
        m_col = month_at_k[:, k].astype(np.int32, copy=False)
        allowed_m = (bits >> (m_col - 1)) & 1
        hit = (~active) & in_win[:, k] & (allowed_m != 0)
        winner = np.where(hit, m_col.astype(np.int8), winner)
        active |= hit
    return active.astype(np.float64), winner


def dd_allow_frac(close_a: np.ndarray, thr_pct: float) -> np.ndarray:
    if thr_pct <= 0.0:
        return np.ones(len(close_a), dtype=np.float64)
    ath = pd.Series(close_a.astype(float)).expanding(min_periods=1).max().to_numpy()
    cur = close_a.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = (ath - cur) / ath * 100.0
    finite_ath = np.isfinite(ath) & (ath > 0.0) & np.isfinite(cur)
    ok = finite_ath & np.isfinite(dd) & (~(dd > float(thr_pct)))
    return cast(np.ndarray[Any, Any], ok.astype(np.float64))


def apply_dd_by_anchor_month_winner(
    seasonal: np.ndarray,
    winner: np.ndarray,
    thr_by_month: dict[int, float],
    dd_frac_by_thr: dict[float, np.ndarray],
) -> np.ndarray:
    out = np.zeros_like(seasonal)
    memo: dict[float, np.ndarray] = {}
    for m in range(1, 13):
        thr_m = float(thr_by_month.get(m, 0.0))
        if thr_m not in memo:
            memo[thr_m] = dd_frac_by_thr[thr_m]
        mult = memo[thr_m]
        wm = winner == np.int8(m)
        sel = wm & (seasonal > 1e-15)
        out = np.where(sel, seasonal * mult, out)
    return out


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
    slice_m = (cmp >= start) & (cmp <= end)
    pr_d = pr.loc[slice_m].astype(float)
    close = daily_ohlc.xs("Close", level="Field", axis=1)["SPY"].reindex(pr_d.index).astype(float)
    bench = close.pct_change(fill_method=None).fillna(0.0)
    metrics_raw = calculate_optimizer_metrics_fast(
        pr_d,
        bench.astype(float),
        "SPY",
        risk_free_rets=None,
        requested_metrics={"Sortino", "Total Return", "Max Drawdown", "Ann. Return"},
    )
    metrics = {str(k): float(v) for k, v in metrics_raw.items()}
    return pr_d, metrics


def _period_mask(index: pd.Index, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        cmp_dates = pd.DatetimeIndex(
            [pd.Timestamp(ts).tz_convert(idx.tz).replace(tzinfo=None) for ts in idx]
        )
    else:
        cmp_dates = idx
    return pd.Series((cmp_dates >= start) & (cmp_dates <= end), index=index)


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
        default=_REPO / "dev" / "spy_intramonth_maxdd_detailed_ibkr_rf_off_2005_2024",
    )
    parser.add_argument(
        "--skip-runner-verify",
        action="store_true",
        help="Skip BacktestRunner parity vs fast sim on deployable shared-threshold winner.",
    )
    args = parser.parse_args()

    start = pd.Timestamp(args.start_date)
    end_spec = pd.Timestamp(args.end_date)

    print("loading_data ...", flush=True)
    gc, ix_f, loc_f, close_a, bench_rets, rets, shared = load_global_and_ohlc(
        start_date=args.start_date,
        end_date=args.end_date,
        cache_only=args.cache_only,
        start=start,
        end=end_spec,
    )
    print(f"loaded_bars n={len(ix_f)}", flush=True)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dd_frac_by_thr: dict[float, np.ndarray] = {thr: dd_allow_frac(close_a, thr) for thr in _DD_GRID}

    t0 = time.perf_counter()
    month_rows_hetero: list[dict[str, Any]] = []
    entry_best: dict[int, int] = {}
    hold_best: dict[int, int] = {}
    thr_best: dict[int, float] = {}
    seiz_cache: dict[tuple[int, int, int], np.ndarray] = {}

    print("task_a_hetero_start", flush=True)
    for month in range(1, 13):
        best_s = float("-inf")
        best_e, best_h, best_th = 1, 5, 0.0
        for ed in range(1, 21):
            for hd in range(5, 21):
                key = (month, ed, hd)
                if key not in seiz_cache:
                    seiz_cache[key] = single_month_long_mask(
                        loc_f,
                        calendar_month=month,
                        entry_day=ed,
                        hold_days=hd,
                    )
                seas = seiz_cache[key]
                for thr in _DD_GRID:
                    tgt = seas * dd_frac_by_thr[thr]
                    mets = eval_target_net_returns(
                        tgt, close=close_a, rets=rets, ix=ix_f, bench_rets=bench_rets
                    )
                    s = float(mets["Sortino"])
                    month_rows_hetero.append(
                        {
                            "calendar_month": month,
                            "month_name": _MONTH_EN[month - 1],
                            "entry_day": ed,
                            "hold_days": hd,
                            "max_dd_from_ath_pct": thr,
                            **mets,
                        }
                    )
                    if math.isfinite(s) and s > best_s:
                        best_s = s
                        best_e, best_h, best_th = ed, hd, thr
        entry_best[month] = best_e
        hold_best[month] = best_h
        thr_best[month] = best_th

    pd.DataFrame(month_rows_hetero).to_csv(out_dir / "task_a_hetero_full_grid.csv", index=False)
    het_summary = []
    for month in range(1, 13):
        sub = [r for r in month_rows_hetero if int(r["calendar_month"]) == month]
        het_summary.append(
            max(
                sub,
                key=lambda r: (
                    float(r["Sortino"]) if math.isfinite(float(r["Sortino"])) else float("-inf")
                ),
            )
        )
    pd.DataFrame(het_summary).to_csv(out_dir / "per_month_best_hetero.csv", index=False)

    print("task_b_hetero_start", flush=True)
    ent_sub, hd_sub = subset_entry_hold(entry_best, hold_best)
    month_at_k_b, in_win_b, _ = precompute_anchor_scan(loc_f, ent_sub, hd_sub)

    het_subset_rows: list[dict[str, Any]] = []
    best_mask_het = -1
    best_sort_het = float("-inf")
    for mask in range(1, (1 << 12)):
        sea, winner = allocate_from_scan_with_winner(month_at_k_b, in_win_b, mask)
        tgt = apply_dd_by_anchor_month_winner(sea, winner, thr_best, dd_frac_by_thr)
        mets = eval_target_net_returns(
            tgt, close=close_a, rets=rets, ix=ix_f, bench_rets=bench_rets
        )
        het_subset_rows.append(
            {
                "bitmask": mask,
                "n_months": int(bin(mask).count("1")),
                "months_csv": ";".join(_MONTH_EN[i] for i in range(12) if (mask >> i) & 1),
                **mets,
            }
        )
        sk = float(mets["Sortino"])
        if math.isfinite(sk) and sk > best_sort_het:
            best_sort_het = sk
            best_mask_het = mask

    pd.DataFrame(het_subset_rows).to_csv(out_dir / "task_b_hetero_all_subsets.csv", index=False)
    print("shared_dd_sweep_start", flush=True)

    shared_rows: list[dict[str, Any]] = []
    shared_month_winners: list[dict[str, Any]] = []
    best_deploy: dict[str, Any] | None = None

    for thr_shared in _DD_GRID:
        mult = dd_frac_by_thr[thr_shared]
        sh_entry: dict[int, int] = {}
        sh_hold: dict[int, int] = {}
        seiz_sh: dict[tuple[int, int, int], np.ndarray] = {}

        for month in range(1, 13):
            best_ms = float("-inf")
            be, bh = 1, 5
            for ed in range(1, 21):
                for hd in range(5, 21):
                    kk = (month, ed, hd)
                    if kk not in seiz_sh:
                        seiz_sh[kk] = single_month_long_mask(
                            loc_f, calendar_month=month, entry_day=ed, hold_days=hd
                        )
                    seas = seiz_sh[kk]
                    tgt = seas * mult
                    mets = eval_target_net_returns(
                        tgt, close=close_a, rets=rets, ix=ix_f, bench_rets=bench_rets
                    )
                    s = float(mets["Sortino"])
                    if math.isfinite(s) and s > best_ms:
                        best_ms = s
                        be, bh = ed, hd
            sh_entry[month] = be
            sh_hold[month] = bh
            shared_month_winners.append(
                {
                    "thr_shared": thr_shared,
                    "calendar_month": month,
                    "entry_day": be,
                    "hold_days": bh,
                    "Sortino_month_grid_best": best_ms,
                }
            )

        ent_su, hd_su = subset_entry_hold(sh_entry, sh_hold)
        month_at_k_s, in_win_s, _ = precompute_anchor_scan(loc_f, ent_su, hd_su)
        best_sort_sh = float("-inf")
        best_mask_sh = -1
        best_mets_row: dict[str, Any] = {}

        for mask in range(1, (1 << 12)):
            sea = allocate_from_scan(month_at_k_s, in_win_s, mask)
            tgt = sea * mult
            mets = eval_target_net_returns(
                tgt, close=close_a, rets=rets, ix=ix_f, bench_rets=bench_rets
            )
            sk = float(mets["Sortino"])
            if math.isfinite(sk) and sk > best_sort_sh:
                best_sort_sh = sk
                best_mask_sh = mask
                best_mets_row = dict(mets)

        row_agg: dict[str, Any] = {
            "thr_shared": float(thr_shared),
            "bitmask": int(best_mask_sh),
            "Sortino": best_sort_sh,
            **best_mets_row,
            **{f"m{m}_entry": sh_entry[m] for m in range(1, 13)},
            **{f"m{m}_hold": sh_hold[m] for m in range(1, 13)},
        }
        shared_rows.append(row_agg)

        cand_s = row_agg["Sortino"]
        bd_s = float(best_deploy["Sortino"]) if best_deploy else float("-inf")
        if best_deploy is None or (math.isfinite(float(cand_s)) and float(cand_s) > bd_s):
            best_deploy = dict(row_agg)

    pd.DataFrame(shared_month_winners).to_csv(
        out_dir / "shared_thr_task_a_month_winners.csv", index=False
    )
    pd.DataFrame(shared_rows).to_csv(
        out_dir / "shared_dd_thr_subset_best_one_row_each.csv", index=False
    )

    parity_runner: dict[str, Any] = {}
    if best_deploy is not None:
        thr_w = float(best_deploy["thr_shared"])
        bm_w = int(best_deploy["bitmask"])
        sh_entry_w = {m: int(best_deploy[f"m{m}_entry"]) for m in range(1, 13)}
        sh_hold_w = {m: int(best_deploy[f"m{m}_hold"]) for m in range(1, 13)}
        sp_run = _params_subset(bm_w, sh_entry_w, sh_hold_w, max_dd_from_ath_pct=thr_w)

        month_at_kw, in_win_w, _ = precompute_anchor_scan(loc_f, sh_entry_w, sh_hold_w)
        sea_w = allocate_from_scan(month_at_kw, in_win_w, bm_w)
        tgt_w = sea_w * dd_frac_by_thr[thr_w]
        lag_target = np.r_[0.0, tgt_w[:-1]]
        gross = lag_target * rets
        delta = np.abs(tgt_w - np.r_[0.0, tgt_w[:-1]])
        tv = delta * PORT
        shares = np.where((tv > 0) & np.isfinite(close_a) & (close_a > 0), tv / close_a, 0.0)
        comm = np.zeros_like(tv)
        nz = shares > 0
        comm[nz] = np.minimum(np.maximum(shares[nz] * CPS, CMIN), tv[nz] * CMAX)
        costs = (comm + tv * (SLIP / 10000.0)) / PORT
        sim_net = pd.Series(gross - costs, index=ix_f)

        if not args.skip_runner_verify:
            run_net, run_m = run_detailed_runner(
                global_config=dict(gc),
                shared=shared,
                strategy_params=sp_run,
                name="deploy_shared_thr_verify",
                start_date=args.start_date,
                end_date=args.end_date,
                start=start,
                end=end_spec,
            )
            parity_runner = {
                "max_abs_return_diff": parity_max_abs_diff(
                    sim_net=sim_net, runner_net=run_net, start=start, end=end_spec
                ),
                "sortino_runner": float(run_m.get("Sortino", np.nan)),
                "thr_shared": thr_w,
                "bitmask": bm_w,
            }

    doc = {
        "period": {"start": args.start_date, "end": args.end_date},
        "risk_free_metrics": False,
        "use_carlos_roro": False,
        "dd_grid_percent": list(_DD_GRID),
        "hetero_task_b_best": {
            "bitmask": int(best_mask_het),
            "months": ";".join(
                _MONTH_EN[i] for i in range(12) if best_mask_het >= 0 and (best_mask_het >> i) & 1
            ),
            "Sortino_fast": float(best_sort_het),
            "note": (
                "DD threshold chosen per-month from Task A; applied when first-scan anchor equals that "
                "month (deployability requires a single YAML max_dd value — see deployable_best)."
            ),
        },
        "per_month_best_heterogeneous": {
            str(m): {
                "entry_day": int(entry_best[m]),
                "hold_days": int(hold_best[m]),
                "max_dd_from_ath_pct": float(thr_best[m]),
            }
            for m in range(1, 13)
        },
        "deployable_best_shared_threshold_row": best_deploy,
        "runner_parity_deployable_shared": parity_runner,
        "wall_seconds_total": time.perf_counter() - t0,
    }

    (out_dir / "pipeline_summary.json").write_text(
        json.dumps(_sanitize_json_obj(doc), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    with (out_dir / "hetero_best_subset_month_params.tsv").open(
        "w", encoding="utf-8", newline="\n"
    ) as fh:
        fh.write("calendar_month\tmonth_name\thetero_active\tentry_day\thold_days\tmax_dd_pct\n")
        for m in range(1, 13):
            act = bool((best_mask_het >> (m - 1)) & 1)
            fh.write(
                f"{m}\t{_MONTH_EN[m - 1]}\t{int(act)}\t{entry_best[m]}\t"
                f"{hold_best[m]}\t{thr_best[m]}\n"
            )

    if best_deploy:
        bm_d = int(best_deploy["bitmask"])
        thr_d = float(best_deploy["thr_shared"])
        with (out_dir / "deploy_best_shared_month_params.tsv").open(
            "w", encoding="utf-8", newline="\n"
        ) as fh:
            fh.write("calendar_month\tmonth_name\tdeploy_active\tentry\thold\tshared_max_dd_pct\n")
            for m in range(1, 13):
                act = bool((bm_d >> (m - 1)) & 1)
                fh.write(
                    f"{m}\t{_MONTH_EN[m - 1]}\t{int(act)}\t{int(best_deploy[f'm{m}_entry'])}\t"
                    f"{int(best_deploy[f'm{m}_hold'])}\t{thr_d}\n"
                )

    print(json.dumps(_sanitize_json_obj(doc), indent=2, allow_nan=False))
    print(f"WROTE_OUT_DIR={out_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
